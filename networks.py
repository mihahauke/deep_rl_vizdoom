# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import convolution2d, fully_connected, flatten
from tensorflow.python.ops.rnn_cell import LSTMStateTuple

from util import Record
import inspect
import sys


class _BaseACNet(object):
    def __init__(self,
                 actions_num,

                 resolution=(84, 84),
                 misc_len=0,
                 stack_n_frames=4,
                 entropy_beta=0.01,
                 thread="global",
                 device="/gpu:0",
                 **ignored):
        self._name_scope = self._get_name_scope() + "_" + str(thread)
        self._misc_len = misc_len
        self._resolution = resolution
        self._stack_n_frames = stack_n_frames
        self._device = device

        self._device = device
        self._actions_num = actions_num
        self._params = None
        self._entropy_beta = entropy_beta

        self.ops = Record()
        self.vars = Record()

        self.ops.sync = self._sync_op

        # TODO make it configurable from json
        self.create_architecture()
        self._prepare_loss_op()
        self._params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name_scope)

    def _sync_op(self, src_network, name=None):
        src_vars = src_network.get_params()
        dst_vars = self.get_params()

        sync_ops = []
        with tf.device(self._device):
            # TODO is this scope needed?
            with tf.name_scope(name, "BaseACNet", []) as name:
                for (src_var, dst_var) in zip(src_vars, dst_vars):
                    sync_op = tf.assign(dst_var, src_var)
                    sync_ops.append(sync_op)

                return tf.group(*sync_ops, name=name)

    def _prepare_loss_op(self):
        with tf.device(self._device):
            self.vars.a = tf.placeholder(tf.float32, [None, self._actions_num], name="action")
            self.vars.advantage = tf.placeholder(tf.float32, [None], name="advantage")
            self.vars.R = tf.placeholder(tf.float32, [None], name="R")

            log_pi = tf.log(tf.clip_by_value(self.ops.pi, 1e-20, 1.0))
            entropy = -tf.reduce_sum(self.ops.pi * log_pi, reduction_indices=1)
            # TODO maybe dacay entropy_beta?
            # TODO is this tf.mul really needed?
            policy_loss = - tf.reduce_sum(
                tf.reduce_sum(tf.mul(log_pi, self.vars.a),
                              reduction_indices=1) * self.vars.advantage + entropy * self._entropy_beta)

            # TODO is further division by 2 needed?
            value_loss = tf.nn.l2_loss(self.vars.R - self.ops.v)

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

    def _get_default_conv_layers(self):
        # TODO add initilizer options?
        # TODO test initialization as in the original paper (sampled from uniform)

        conv1 = convolution2d(self.vars.state_img, num_outputs=32, kernel_size=[8, 8], stride=4, padding="VALID",
                              scope=self._name_scope + "/conv1",
                              biases_initializer=tf.constant_initializer(0.1))
        conv2 = convolution2d(conv1, num_outputs=64, kernel_size=[4, 4], stride=2, padding="VALID",
                              scope=self._name_scope + "/conv2",
                              biases_initializer=tf.constant_initializer(0.1))
        conv3 = convolution2d(conv2, num_outputs=64, kernel_size=[3, 3], stride=1, padding="VALID",
                              scope=self._name_scope + "/conv3",
                              biases_initializer=tf.constant_initializer(0.1))
        conv3_flat = flatten(conv3)

        return conv3_flat

    def _get_name_scope(self):
        raise NotImplementedError()


class FFACNet(_BaseACNet):
    def __init__(self,
                 **kwargs):
        super(FFACNet, self).__init__(**kwargs)

    def _get_name_scope(self):
        return "a3c_ff_net"

    def create_architecture(self, **specs):
        with tf.device(self._device):
            state_shape = [None] + list(self._resolution) + [self._stack_n_frames]
            self.vars.state_img = tf.placeholder("float", state_shape, name="state_img")
            if self._misc_len > 0:
                self.vars.state_misc = tf.placeholder("float", [None, self._misc_len], name="state_misc")

            conv_layers = self._get_default_conv_layers()

            fc1 = fully_connected(conv_layers, num_outputs=256, scope=self._name_scope + "/fc1",
                                  biases_initializer=tf.constant_initializer(0.1))

            self.ops.pi = fully_connected(fc1, num_outputs=self._actions_num, scope=self._name_scope + "/fc_pi",
                                          activation_fn=tf.nn.softmax,
                                          biases_initializer=tf.constant_initializer(0.1))

            state_value = fully_connected(fc1, num_outputs=1,
                                          scope=self._name_scope + "/fc_value",
                                          activation_fn=None,
                                          biases_initializer=tf.constant_initializer(0.1))

            self.ops.v = tf.reshape(state_value, [-1])

    def _get_standard_feed_dict(self, state):
        feed_dict = {self.vars.state_img: [state[0]]}
        if self._misc_len > 0:
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
        with tf.device(self._device):
            state_shape = [None] + list(self._resolution) + [self._stack_n_frames]
            self.vars.state_img = tf.placeholder(tf.float32, state_shape, name="state_img")
            if self._misc_len > 0:
                self.vars.state_misc = tf.placeholder("float", [None, self._misc_len], name="state_misc")
            self.vars.sequence_length = tf.placeholder(tf.float32, [1], name="sequence_length")

            conv_layers = self._get_default_conv_layers()

            if self._misc_len > 0:
                fc_input = tf.concat(concat_dim=1, values=[conv_layers, self.vars.state_misc])
            else:
                fc_input = conv_layers

            fc1 = fully_connected(fc_input, num_outputs=self._recurrent_units_num, scope=self._name_scope + "/fc1",
                                  biases_initializer=tf.constant_initializer(0.1))

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

            self.ops.pi = fully_connected(lstm_outputs, num_outputs=self._actions_num,
                                          scope=self._name_scope + "/fc_pi",
                                          activation_fn=tf.nn.softmax,
                                          biases_initializer=tf.constant_initializer(0.1))

            state_value = fully_connected(lstm_outputs, num_outputs=1,
                                          scope=self._name_scope + "/fc_value",
                                          activation_fn=None,
                                          biases_initializer=tf.constant_initializer(0.1))

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
        if self._misc_len > 0:
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


# TODO make a module and move this methods somewhere else?
# TODO add short name
def get_available_networks():
    nets = []
    for member in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(member[1]):
            member_name = member[0]
            if member_name.endswith("Net") and not member_name.startswith("_"):
                nets.append(member)
    return nets


def create_ac_network(network_type, **args):
    _short_names = {FFACNet: "ff_ac", BasicLstmACACNet: "basic_lstm_ac", LstmACACNet: "lstm_ac"}
    _inv_short_names = {v: k for k, v in _short_names.items()}
    if network_type is not None:
        for mname, mclass in get_available_networks():
            if network_type == mname:
                return mclass(**args)
        if network_type in _inv_short_names:
            return _inv_short_names[network_type](**args)

    raise ValueError("Unsupported net: {}".format(network_type))
