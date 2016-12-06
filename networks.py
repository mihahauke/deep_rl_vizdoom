# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
from tensorflow.python.ops.rnn_cell import LSTMStateTuple
from tensorflow.contrib.framework import arg_scope

from util import Record
from util.tfutil import gather_2d
import inspect
import sys


def _default_conv_layers(img_input, name_scope, activation_fn=tf.nn.relu, trainable=True):
    # TODO test tf.nn.elu
    # TODO test initialization as in the original paper (sampled from uniform)
    with arg_scope([layers.conv2d], activation_fn=activation_fn), arg_scope([layers.conv2d], data_format="NCHW",
                                                                            trainable=trainable):
        conv1 = layers.conv2d(img_input, num_outputs=32, kernel_size=[8, 8], stride=4, padding="VALID",
                              scope=name_scope + "/conv1", trainable=trainable)
        conv2 = layers.conv2d(conv1, num_outputs=64, kernel_size=[4, 4], stride=2, padding="VALID",
                              scope=name_scope + "/conv2", )
        conv3 = layers.conv2d(conv2, num_outputs=64, kernel_size=[3, 3], stride=1, padding="VALID",
                              scope=name_scope + "/conv3", )
        conv3_flat = layers.flatten(conv3)

        return conv3_flat


# A3C nets:

class _BaseACNet(object):
    def __init__(self,
                 actions_num,
                 img_shape,
                 misc_len=0,
                 entropy_beta=0.01,
                 thread="global",
                 **ignored):
        self.ops = Record()
        self.vars = Record()
        self.vars.state_img = tf.placeholder(tf.float32, [None] + list(img_shape), name="state_img")
        self.use_misc = misc_len > 0
        if self.use_misc:
            self.vars.state_misc = tf.placeholder("float", [None, misc_len], name="state_misc")
        self._actions_num = actions_num
        self._name_scope = self._get_name_scope() + "_" + str(thread)

        self._params = None
        self._entropy_beta = entropy_beta

        self.ops.sync = self._sync_op

        # TODO make it configurable from json
        self.create_architecture()
        self._prepare_loss_op()
        self._params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name_scope)

    def _sync_op(self, src_network, name=None):
        src_vars = src_network.get_params()
        dst_vars = self.get_params()

        sync_ops = []
        for src_var, dst_var in zip(src_vars, dst_vars):
            sync_op = tf.assign(dst_var, src_var)
            sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)

    def _prepare_loss_op(self):
        self.vars.a = tf.placeholder(tf.float32, [None, self._actions_num], name="action")
        self.vars.advantage = tf.placeholder(tf.float32, [None], name="advantage")
        self.vars.R = tf.placeholder(tf.float32, [None], name="R")
        # TODO add summaries for entropy, policy and value
        log_pi = tf.log(tf.clip_by_value(self.ops.pi, 1e-20, 1.0))
        entropy = -tf.reduce_sum(self.ops.pi * log_pi, reduction_indices=1)
        policy_loss = - tf.reduce_sum(
            tf.reduce_sum(tf.mul(log_pi, self.vars.a),
                          reduction_indices=1) * self.vars.advantage + entropy * self._entropy_beta)
        value_loss = 0.5 * tf.reduce_sum(tf.squared_difference(self.vars.R, self.ops.v))

        self.ops.loss = policy_loss + value_loss

    def create_architecture(self, **specs):
        raise NotImplementedError()

    def _get_standard_feed_dict(self, state):
        raise NotImplementedError()

    def get_policy_and_value(self, sess, state):
        pi_out, v_out = sess.run([self.ops.pi, self.ops.v], feed_dict=self._get_standard_feed_dict(state))
        return pi_out[0], v_out[0]

    def get_policy(self, sess, state):
        pi_out = sess.run(self.ops.pi, feed_dict=self._get_standard_feed_dict(state))
        return pi_out[0]

    def get_value(self, sess, state):
        v = sess.run(self.ops.v, feed_dict=self._get_standard_feed_dict(state))
        return v[0]

    def get_params(self):
        return self._params

    def has_state(self):
        return False

    def _get_name_scope(self):
        raise NotImplementedError()


class FFACNet(_BaseACNet):
    def __init__(self,
                 **kwargs):
        super(FFACNet, self).__init__(**kwargs)

    def _get_name_scope(self):
        return "a3c_ff_net"

    def create_architecture(self, **specs):
        conv_layers = _default_conv_layers(self.vars.state_img, self._name_scope)

        if self.use_misc:
            fc_input = tf.concat(concat_dim=1, values=[conv_layers, self.vars.state_misc])
        else:
            fc_input = conv_layers

        fc1 = layers.fully_connected(fc_input, num_outputs=256, scope=self._name_scope + "/fc1",
                                     biases_initializer=tf.constant_initializer(0.1),
                                     activation_fn=tf.nn.relu)

        self.ops.pi = layers.fully_connected(fc1, num_outputs=self._actions_num, scope=self._name_scope + "/fc_pi",
                                             activation_fn=tf.nn.softmax)

        state_value = layers.fully_connected(fc1, num_outputs=1,
                                             scope=self._name_scope + "/fc_value",
                                             activation_fn=None)

        self.ops.v = tf.reshape(state_value, [-1])

    def _get_standard_feed_dict(self, state):
        feed_dict = {self.vars.state_img: [state[0]]}
        if self.use_misc > 0:
            if len(state[1].shape) == 1:
                misc = state[1].reshape([1, -1])
            else:
                misc = state[1]
            feed_dict[self.vars.state_misc] = misc
        return feed_dict


class _BaseRcurrentACNet(_BaseACNet):
    def __init__(self,
                 recurrent_units_num=256,
                 **settings
                 ):
        self._recurrent_cells = None
        self.network_state = None
        # TODO make it configurable in jsons
        self._recurrent_units_num = recurrent_units_num
        super(_BaseRcurrentACNet, self).__init__(**settings)

    def _get_ru_class(self):
        raise NotImplementedError()

    def create_architecture(self, **specs):
        self.vars.sequence_length = tf.placeholder(tf.float32, [1], name="sequence_length")

        conv_layers = _default_conv_layers(self.vars.state_img, self._name_scope)

        if self.use_misc:
            fc_input = tf.concat(concat_dim=1, values=[conv_layers, self.vars.state_misc])
        else:
            fc_input = conv_layers

        fc1 = layers.fully_connected(fc_input, num_outputs=self._recurrent_units_num,
                                     scope=self._name_scope + "/fc1",
                                     biases_initializer=tf.constant_initializer(0.1),
                                     activation_fn=tf.nn.relu)

        fc1_reshaped = tf.reshape(fc1, [1, -1, self._recurrent_units_num])

        self._recurrent_cells = self._get_ru_class()(self._recurrent_units_num)
        state_c = tf.placeholder(tf.float32, [1, self._recurrent_cells.state_size.c], name="initial_lstm_state_c")
        state_h = tf.placeholder(tf.float32, [1, self._recurrent_cells.state_size.h], name="initial_lstm_state_h")
        self.vars.initial_network_state = LSTMStateTuple(state_c, state_h)
        lstm_outputs, self.ops.network_state = tf.nn.dynamic_rnn(self._recurrent_cells,
                                                                 fc1_reshaped,
                                                                 initial_state=self.vars.initial_network_state,
                                                                 sequence_length=self.vars.sequence_length,
                                                                 time_major=False,
                                                                 scope=self._name_scope)

        lstm_outputs = tf.reshape(lstm_outputs, [-1, self._recurrent_units_num])

        self.ops.pi = layers.fully_connected(lstm_outputs, num_outputs=self._actions_num,
                                             scope=self._name_scope + "/fc_pi",
                                             activation_fn=tf.nn.softmax)

        state_value = layers.fully_connected(lstm_outputs, num_outputs=1,
                                             scope=self._name_scope + "/fc_value",
                                             activation_fn=None)

        self.ops.v = tf.reshape(state_value, [-1])

        self.reset_state()

    def reset_state(self):
        state_c = np.zeros([1, self._recurrent_cells.state_size.c], dtype=np.float32)
        state_h = np.zeros([1, self._recurrent_cells.state_size.h], dtype=np.float32)
        self.network_state = LSTMStateTuple(state_c, state_h)

    def _get_standard_feed_dict(self, state):
        feed_dict = {self.vars.state_img: [state[0]],
                     self.vars.initial_network_state: self.network_state,
                     self.vars.sequence_length: [1]}
        if self.use_misc > 0:
            if len(state[1].shape) == 1:
                misc = state[1].reshape([1, -1])
            else:
                misc = state[1]
            feed_dict[self.vars.state_misc] = misc
        return feed_dict

    def get_policy_and_value(self, sess, state):
        # This run_policy_and_value() is used when forward propagating.
        # so the step size is 1.
        policy, value, self.network_state = sess.run([self.ops.pi, self.ops.v, self.ops.network_state],
                                                     feed_dict=self._get_standard_feed_dict(state))
        # pi_out: (1,3), v_out: (1)
        return policy[0], value[0]

    def get_policy(self, sess, state):
        policy, self.network_state = sess.run([self.ops.pi, self.ops.network_state],
                                              feed_dict=self._get_standard_feed_dict(state))
        return policy[0]

    def get_value(self, sess, state, retain_net_state=True):
        if retain_net_state:
            initial_network_sate = self.network_state
        v = super(_BaseRcurrentACNet, self).get_value(sess, state)

        if retain_net_state:
            self.network_state = initial_network_sate
        return v

    def get_current_network_state(self):
        return self.network_state

    def has_state(self):
        return True


class BasicLstmACACNet(_BaseRcurrentACNet):
    def __init__(self,
                 **settings
                 ):
        super(BasicLstmACACNet, self).__init__(**settings)

    def _get_name_scope(self):
        return "a3c_basic_lstm_net"

    def _get_ru_class(self):
        return tf.nn.rnn_cell.BasicLSTMCell


class LstmACACNet(_BaseRcurrentACNet):
    def __init__(self,
                 **settings
                 ):
        super(LstmACACNet, self).__init__(**settings)

    def _get_name_scope(self):
        return "a3c_lstm_net"

    def _get_ru_class(self):
        return tf.nn.rnn_cell.LSTMCell


# DQN nets:

class BaseDQNNet(object):
    def __init__(self,
                 actions_num,
                 img_shape,
                 gamma,
                 misc_len=0,
                 **settings):
        self.gamma = np.float32(gamma)
        self.ops = Record()
        self.vars = Record()
        self.vars.state_img = tf.placeholder(tf.float32, [None] + list(img_shape), name="state_img")
        self.vars.state2_img = tf.placeholder(tf.float32, [None] + list(img_shape), name="state2_img")
        self.use_misc = misc_len > 0
        if self.use_misc:
            self.vars.state_misc = tf.placeholder("float", [None, misc_len], name="state_misc")
            self.vars.state2_misc = tf.placeholder("float", [None, misc_len], name="state2_misc")
        else:
            self.vars.state_misc = None
            self.vars.state2_misc = None
        self._actions_num = actions_num

        self._name_scope = self._get_name_scope()

        # TODO make it configurable from json
        # self.create_architecture()
        # self._prepare_loss_op()
        self.vars.a = tf.placeholder(tf.int32, [None], "action")
        self.vars.r = tf.placeholder(tf.float32, [None], "reward")
        self.vars.terminal = tf.placeholder(tf.bool, [None], "terminal")

        frozen_name_scope = self._name_scope + "/frozen"
        q = self.create_architecture(self.vars.state_img, self.vars.state_misc, trainable=True,
                                     name_scope=self._name_scope)
        q2 = self.create_architecture(self.vars.state2_img, self.vars.state2_misc, trainable=False,
                                      name_scope=frozen_name_scope)
        net_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name_scope)
        target_net_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=frozen_name_scope)

        unfreeze_ops = [tf.assign(dst_var, src_var) for src_var, dst_var in zip(net_params, target_net_params)]

        self.ops.unfreeze = tf.group(*unfreeze_ops, name="unfreeze")
        self.ops.best_action = tf.argmax(q, axis=1)[0]

        best_q2 = tf.reduce_max(q2, axis=1)
        target_q = self.vars.r + self.gamma * best_q2 * (1 - tf.to_float(self.vars.terminal))
        active_q = gather_2d(q, self.vars.a)

        loss = 0.5 * tf.reduce_mean((target_q - active_q) ** 2)

        global_step = tf.Variable(0, trainable=False, name="global_step")
        # TODO customize learning rate decay
        if settings["constant_learning_rate"]:
            learning_rate = settings["initial_learning_rate"]
        else:
            learning_rate = tf.train.polynomial_decay(
                learning_rate=settings["initial_learning_rate"],
                end_learning_rate=settings["final_learning_rate"],
                decay_steps=settings["learning_rate_decay_steps"],
                global_step=global_step)

        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, **settings["rmsprop"])
        self.ops.train_batch = optimizer.minimize(loss)

    def _prepare_loss_op(self):
        pass
        #     self.vars.a = tf.placeholder(tf.float32, [None, self._actions_num], name="action")
        #     self.vars.R = tf.placeholder(tf.float32, [None], name="R")
        #     self.ops.loss = 0.5 * tf.reduce_sum(tf.squared_difference(self.vars.R, self.ops.v))

    def create_architecture(self, img_input, misc_input, trainable, name_scope, **specs):
        conv_layers = _default_conv_layers(img_input, name_scope, trainable=trainable)

        if self.use_misc:
            fc_input = tf.concat(concat_dim=1, values=[conv_layers, misc_input])
        else:
            fc_input = conv_layers

        fc1 = layers.fully_connected(fc_input, num_outputs=256, scope=name_scope + "/fc1",
                                     activation_fn=tf.nn.relu, trainable=trainable)
        q_op = layers.fully_connected(fc1, num_outputs=self._actions_num, scope=name_scope + "/fc_q",
                                      activation_fn=None, trainable=trainable)
        return q_op

    def get_action(self, sess, state):
        feed_dict = {self.vars.state_img: [state[0]]}
        if self.use_misc:
            feed_dict[self.vars.state_misc] = [state[1]]
        return sess.run(self.ops.best_action, feed_dict=feed_dict)

    def train_batch(self, sess, batch):
        feed_dict = {self.vars.state_img: batch["s1_img"], self.vars.a: batch["a"],
                     self.vars.r: batch["r"], self.vars.state2_img: batch["s2_img"],
                     self.vars.terminal: batch["terminal"]}
        if self.use_misc:
            feed_dict[self.vars.state_misc] = batch["s1_misc"]
            feed_dict[self.vars.state2_misc] = batch["s2_misc"]
        sess.run(self.ops.train_batch, feed_dict=feed_dict)

    def update_target_network(self, session):
        session.run(self.ops.unfreeze)

    def _get_name_scope(self):
        return "dqn"


# TODO make a module and move this methods somewhere else?
def get_available_ac_networks():
    nets = []
    for member in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(member[1]):
            member_name = member[0]
            if member_name.endswith("ACNet") and not member_name.startswith("_"):
                nets.append(member)
    return nets


def get_available_dqn_networks():
    nets = []
    for member in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(member[1]):
            member_name = member[0]
            if member_name.endswith("DQNNet") and not member_name.startswith("_"):
                nets.append(member)
    return nets


def create_ac_network(network_type, **args):
    _short_names = {FFACNet: "ff_ac", BasicLstmACACNet: "basic_lstm_ac", LstmACACNet: "lstm_ac"}
    _inv_short_names = {v: k for k, v in _short_names.items()}
    if network_type is not None:
        for mname, mclass in get_available_ac_networks():
            if network_type == mname:
                return mclass(**args)
        if network_type in _inv_short_names:
            return _inv_short_names[network_type](**args)

    raise ValueError("Unsupported net: {}".format(network_type))


def create_dqn_network(network_type, **args):
    _short_names = {BaseDQNNet: "base_dqn"}
    _inv_short_names = {v: k for k, v in _short_names.items()}
    if network_type is not None:
        for mname, mclass in get_available_ac_networks():
            if network_type == mname:
                return mclass(**args)
        if network_type in _inv_short_names:
            return _inv_short_names[network_type](**args)

    raise ValueError("Unsupported net: {}".format(network_type))
