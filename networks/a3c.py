# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers

from .common import _BaseNetwork, gather_2d


class _BaseACNet(_BaseNetwork):
    def __init__(self,
                 initial_entropy_beta=0.05,
                 final_entropy_beta=0.0,
                 entropy_beta_decay_steps=10e6,
                 thread="global",
                 **settings):

        super(_BaseACNet, self).__init__(**settings)
        self.network_state = None
        self._name_scope = self._get_name_scope() + "_" + str(thread)

        if initial_entropy_beta == final_entropy_beta:
            self._entropy_beta = initial_entropy_beta
        else:
            self._entropy_beta = tf.train.polynomial_decay(
                name="larning_rate",
                learning_rate=initial_entropy_beta,
                end_learning_rate=final_entropy_beta,
                decay_steps=entropy_beta_decay_steps,
                global_step=tf.train.get_global_step())

        with arg_scope([layers.conv2d], data_format="NCHW"), \
             arg_scope([layers.fully_connected, layers.conv2d], activation_fn=self.activation_fn):
            self.create_architecture()

        self._prepare_loss_op()
        self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name_scope)

    def policy_value_layer(self, inputs):
        pi = layers.fully_connected(inputs,
                                    num_outputs=self.actions_num,
                                    scope=self._name_scope + "/fc_pi",
                                    activation_fn=tf.nn.softmax)
        state_value = layers.linear(inputs,
                                    num_outputs=1,
                                    scope=self._name_scope + "/fc_value")
        v = tf.reshape(state_value, [-1])
        return pi, v

    def prepare_sync_op(self, global_network):
        global_params = global_network.get_params()
        local_params = self.get_params()
        sync_ops = [dst_var.assign(src_var) for dst_var, src_var, in zip(local_params, global_params)]

        self.ops.sync = tf.group(*sync_ops, name="SyncWithGlobal")

    def _prepare_loss_op(self):
        self.vars.a = tf.placeholder(tf.int32, [None], name="action")
        self.vars.advantage = tf.placeholder(tf.float32, [None], name="advantage")
        self.vars.R = tf.placeholder(tf.float32, [None], name="R")

        log_pi = tf.log(tf.clip_by_value(self.ops.pi, 1e-20, 1.0))
        entropy = -tf.reduce_sum(self.ops.pi * log_pi)
        chosen_pi_log = gather_2d(log_pi, self.vars.a)

        policy_loss = - tf.reduce_sum(chosen_pi_log * self.vars.advantage)
        value_loss = 0.5 * tf.reduce_sum(tf.squared_difference(self.vars.R, self.ops.v))

        self.ops.loss = policy_loss + value_loss - entropy * self._entropy_beta

    def create_architecture(self, **specs):
        raise NotImplementedError()

    def get_standard_feed_dict(self, state):
        feed_dict = {self.vars.state_img: [state[0]]}
        if self.use_misc:
            if len(state[1].shape) == 1:
                misc = state[1].reshape([1, -1])
            else:
                misc = state[1]
            feed_dict[self.vars.state_misc] = misc
        return feed_dict

    def get_policy_and_value(self, sess, state):
        pi, v = sess.run([self.ops.pi, self.ops.v], feed_dict=self.get_standard_feed_dict(state))
        return pi[0], v[0]

    def get_policy(self, sess, state):
        pi_out = sess.run(self.ops.pi, feed_dict=self.get_standard_feed_dict(state))
        return pi_out[0]

    def get_value(self, sess, state):
        v = sess.run(self.ops.v, feed_dict=self.get_standard_feed_dict(state))
        return v[0]

    def get_params(self):
        return self.params

    def has_state(self):
        return False

    def get_current_network_state(self):
        return self.network_state

    def _get_name_scope(self):
        raise NotImplementedError()


class ACFFNet(_BaseACNet):
    def __init__(self,
                 **kwargs):
        super(ACFFNet, self).__init__(**kwargs)

    def _get_name_scope(self):
        return "ff_ac"

    def create_architecture(self):
        fc_input = self.get_input_layers()

        fc1 = layers.fully_connected(fc_input,
                                     num_outputs=self.fc_units_num,
                                     scope=self._name_scope + "/fc1",
                                     biases_initializer=tf.constant_initializer(0.1))

        self.ops.pi, self.ops.v = self.policy_value_layer(fc1)


class _BaseACRecurrentNet(_BaseACNet):
    def __init__(self,
                 ru_class,
                 recurrent_units_num=256,
                 **settings
                 ):
        self.recurrent_cells = None
        self._recurrent_units_num = recurrent_units_num
        self.ru_class = ru_class
        super(_BaseACRecurrentNet, self).__init__(**settings)

    def create_architecture(self, **specs):
        self.vars.sequence_length = tf.placeholder(tf.int64, [1], name="sequence_length")

        fc_input = self.get_input_layers()

        fc1 = layers.fully_connected(fc_input, num_outputs=self.fc_units_num,
                                     scope=self._name_scope + "/fc1",
                                     biases_initializer=tf.constant_initializer(0.1))

        fc1_reshaped = tf.reshape(fc1, [1, -1, self.fc_units_num])
        self.recurrent_cells = self.ru_class(self._recurrent_units_num)
        state_c = tf.placeholder(tf.float32, [1, self.recurrent_cells.state_size.c], name="initial_lstm_state_c")
        state_h = tf.placeholder(tf.float32, [1, self.recurrent_cells.state_size.h], name="initial_lstm_state_h")
        self.vars.initial_network_state = LSTMStateTuple(state_c, state_h)
        rnn_outputs, self.ops.network_state = tf.nn.dynamic_rnn(self.recurrent_cells,
                                                                fc1_reshaped,
                                                                initial_state=self.vars.initial_network_state,
                                                                sequence_length=self.vars.sequence_length,
                                                                time_major=False,
                                                                scope=self._name_scope)
        reshaped_rnn_outputs = tf.reshape(rnn_outputs, [-1, self._recurrent_units_num])

        self.reset_state()
        self.ops.pi, self.ops.v = self.policy_value_layer(reshaped_rnn_outputs)

    def reset_state(self):
        state_c = np.zeros([1, self.recurrent_cells.state_size.c], dtype=np.float32)
        state_h = np.zeros([1, self.recurrent_cells.state_size.h], dtype=np.float32)
        self.network_state = LSTMStateTuple(state_c, state_h)

    def has_state(self):
        return True

    def get_standard_feed_dict(self, state):
        feed_dict = super(_BaseACRecurrentNet, self).get_standard_feed_dict(state)
        feed_dict[self.vars.initial_network_state] = self.network_state,
        feed_dict[self.vars.sequence_length] = [1]
        return feed_dict

    def get_policy_and_value(self, sess, state, update_state=True):
        pi, v, new_network_state = sess.run([self.ops.pi, self.ops.v, self.ops.network_state],
                                            feed_dict=self.get_standard_feed_dict(state))
        if update_state:
            self.network_state = new_network_state
        return pi[0], v[0]

    def get_policy(self, sess, state, update_state=True):
        pi, new_network_state = sess.run([self.ops.pi, self.ops.network_state],
                                         feed_dict=self.get_standard_feed_dict(state))
        if update_state:
            self.network_state = new_network_state
        return pi[0]

    def get_value(self, sess, state, update_state=False):
        v, new_network_state = sess.run([self.ops.v, self.ops.network_state],
                                        feed_dict=self.get_standard_feed_dict(state))
        if update_state:
            self.network_state = new_network_state
        return v[0]


class ACBacisLstmNet(_BaseACRecurrentNet):
    def __init__(self,
                 **settings
                 ):
        super(ACBacisLstmNet, self).__init__(tf.contrib.rnn.BasicLSTMCell, **settings)

    def _get_name_scope(self):
        return "basic_lstm_ac"


class ACLstmNet(_BaseACRecurrentNet):
    def __init__(self,
                 **settings
                 ):
        super(ACLstmNet, self).__init__(tf.contrib.rnn.LSTMCell, **settings)

    def _get_name_scope(self):
        return "lstm_ac"


class FigarACFFNet(_BaseACNet):
    def __init__(self,
                 dynamic_frameskips,
                 **settings
                 ):
        if dynamic_frameskips:
            if isinstance(dynamic_frameskips, (list, tuple)):
                self._frameskips_num = len(dynamic_frameskips)
            elif isinstance(dynamic_frameskips, int):
                self._frameskips_num = dynamic_frameskips
            else:
                # TODO
                raise ValueError()

        super(FigarACFFNet, self).__init__(**settings)

    def _prepare_loss_op(self):
        self.vars.a = tf.placeholder(tf.int32, [None], name="action")
        self.vars.frameskip = tf.placeholder(tf.int32, [None], name="frameskip")
        self.vars.advantage = tf.placeholder(tf.float32, [None], name="advantage")
        self.vars.R = tf.placeholder(tf.float32, [None], name="R")

        log_pi = tf.log(tf.clip_by_value(self.ops.pi, 1e-20, 1.0))
        entropy = -tf.reduce_sum(self.ops.pi * log_pi)
        chosen_pi_log = gather_2d(log_pi, self.vars.a)

        log_fs_pi = tf.log(tf.clip_by_value(self.ops.frameskip_pi, 1e-20, 1.0))
        entropy += -tf.reduce_sum(self.ops.frameskip_pi * log_fs_pi)
        chosen_fs_pi_log = gather_2d(log_fs_pi, self.vars.frameskip)

        policy_loss = - tf.reduce_sum((chosen_pi_log + chosen_fs_pi_log) * self.vars.advantage)
        value_loss = 0.5 * tf.reduce_sum(tf.squared_difference(self.vars.R, self.ops.v))

        self.ops.loss = policy_loss + value_loss - entropy * self._entropy_beta

    def policy_value_frameskip_layer(self, inputs):
        pi = layers.fully_connected(inputs,
                                    num_outputs=self.actions_num,
                                    scope=self._name_scope + "/fc_pi",
                                    activation_fn=tf.nn.softmax)
        frameskip = layers.fully_connected(inputs,
                                           num_outputs=self._frameskips_num,
                                           scope=self._name_scope + "/fc_frameskip",
                                           activation_fn=tf.nn.softmax)
        state_value = layers.linear(inputs,
                                    num_outputs=1,
                                    scope=self._name_scope + "/fc_value")

        v = tf.reshape(state_value, [-1])
        return pi, frameskip, v

    def _get_name_scope(self):
        return "ff_figar_ac"

    def create_architecture(self):
        fc_input = self.get_input_layers()

        fc1 = layers.fully_connected(fc_input,
                                     num_outputs=self.fc_units_num,
                                     scope=self._name_scope + "/fc1",
                                     biases_initializer=tf.constant_initializer(0.1))

        self.ops.pi, self.ops.frameskip_pi, self.ops.v = self.policy_value_frameskip_layer(fc1)

    def get_policy_and_value(self, sess, state):
        pi, fs, v = sess.run([self.ops.pi, self.ops.frameskip_pi, self.ops.v],
                             feed_dict=self.get_standard_feed_dict(state))
        return pi[0], fs[0], v[0]

    def get_policy(self, sess, state):
        pi_a, pi_fs = sess.run([self.ops.pi, self.ops.frameskip_pi], feed_dict=self.get_standard_feed_dict(state))
        return pi_a[0], pi_fs[0]


class FigarACLSTMNet(FigarACFFNet):
    def __init__(self,
                 ru_class=tf.contrib.rnn.LSTMCell,
                 recurrent_units_num=256,
                 **settings
                 ):

        self.recurrent_cells = None
        self.network_state = None
        self._recurrent_units_num = recurrent_units_num
        self.ru_class = ru_class

        super(FigarACLSTMNet, self).__init__(**settings)

    def _get_name_scope(self):
        return "lstm_figar_ac"

    def reset_state(self):
        state_c = np.zeros([1, self.recurrent_cells.state_size.c], dtype=np.float32)
        state_h = np.zeros([1, self.recurrent_cells.state_size.h], dtype=np.float32)
        self.network_state = LSTMStateTuple(state_c, state_h)

    def has_state(self):
        return True

    def create_architecture(self):
        self.vars.sequence_length = tf.placeholder(tf.int64, [1], name="sequence_length")

        fc_input = self.get_input_layers()

        fc1 = layers.fully_connected(fc_input, num_outputs=self.fc_units_num,
                                     scope=self._name_scope + "/fc1",
                                     biases_initializer=tf.constant_initializer(0.1))

        fc1_reshaped = tf.reshape(fc1, [1, -1, self.fc_units_num])
        self.recurrent_cells = self.ru_class(self._recurrent_units_num)
        state_c = tf.placeholder(tf.float32, [1, self.recurrent_cells.state_size.c], name="initial_lstm_state_c")
        state_h = tf.placeholder(tf.float32, [1, self.recurrent_cells.state_size.h], name="initial_lstm_state_h")
        self.vars.initial_network_state = LSTMStateTuple(state_c, state_h)
        rnn_outputs, self.ops.network_state = tf.nn.dynamic_rnn(self.recurrent_cells,
                                                                fc1_reshaped,
                                                                initial_state=self.vars.initial_network_state,
                                                                sequence_length=self.vars.sequence_length,
                                                                time_major=False,
                                                                scope=self._name_scope)
        reshaped_rnn_outputs = tf.reshape(rnn_outputs, [-1, self._recurrent_units_num])

        self.reset_state()
        self.ops.pi, self.ops.frameskip_pi, self.ops.v = self.policy_value_frameskip_layer(reshaped_rnn_outputs)

    def get_standard_feed_dict(self, state):
        feed_dict = super(FigarACLSTMNet, self).get_standard_feed_dict(state)
        feed_dict[self.vars.initial_network_state] = self.network_state,
        feed_dict[self.vars.sequence_length] = [1]
        return feed_dict

    def get_policy_and_value(self, sess, state, update_state=True):
        pi, pi_fs, v, new_network_state = sess.run([self.ops.pi, self.ops.frameskip_pi, self.ops.v,
                                                 self.ops.network_state],
                                                feed_dict=self.get_standard_feed_dict(state))
        if update_state:
            self.network_state = new_network_state
        return pi[0], pi_fs[0], v[0]

    def get_policy(self, sess, state, update_state=True):
        pi, pi_fs, new_network_state = sess.run([self.ops.pi, self.ops.frameskip_pi, self.ops.network_state],
                                                feed_dict=self.get_standard_feed_dict(state))
        if update_state:
            self.network_state = new_network_state

        return pi[0], pi_fs[0]

    def get_value(self, sess, state, update_state=False):
        v, new_network_state = sess.run([self.ops.v, self.ops.network_state],
                                        feed_dict=self.get_standard_feed_dict(state))
        if update_state:
            self.network_state = new_network_state
        return v[0]
