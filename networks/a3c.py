# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from tensorflow.python.ops.rnn_cell import LSTMStateTuple
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers

from util import Record
from util.tfutil import gather_2d

from .common import default_conv_layers


class _BaseACNet(object):
    def __init__(self,
                 actions_num,
                 img_shape,
                 misc_len=0,
                 entropy_beta=0.01,
                 thread="global",
                 activation_fn="tf.nn.relu",
                 **ignored):
        self.activation_fn = eval(activation_fn)
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

        with arg_scope([layers.conv2d], data_format="NCHW"), \
             arg_scope([layers.fully_connected, layers.conv2d], activation_fn=self.activation_fn):
            # TODO make it configurable from yaml
            self.ops.pi, self.ops.v = self.create_architecture()
        self._prepare_loss_op()
        self._params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name_scope)

    def prepare_sync_op(self, global_network):
        global_params = global_network.get_params()
        local_params = self.get_params()
        sync_ops = [tf.assign(dst_var, src_var) for src_var, dst_var in zip(global_params, local_params)]

        self.ops.sync = tf.group(*sync_ops, name="SyncWithGlobal")

    def _prepare_loss_op(self):
        # TODO use gather2d instead of one hot action
        self.vars.a = tf.placeholder(tf.float32, [None], name="action")
        self.vars.advantage = tf.placeholder(tf.float32, [None], name="advantage")
        self.vars.R = tf.placeholder(tf.float32, [None], name="R")
        # TODO add summaries for entropy, policy and value

        log_pi = tf.log(tf.clip_by_value(self.ops.pi, 1e-20, 1.0))
        entropy = -tf.reduce_sum(self.ops.pi * log_pi)
        chosen_pi_log = gather_2d(log_pi, self.vars.a)

        policy_loss = - tf.reduce_sum(chosen_pi_log * self.vars.advantage) + entropy * self._entropy_beta
        value_loss = 0.5 * tf.reduce_sum(tf.squared_difference(self.vars.R, self.ops.v))

        self.ops.loss = policy_loss + value_loss

    def create_architecture(self, **specs):
        raise NotImplementedError()

    def _get_standard_feed_dict(self, state):
        raise NotImplementedError()

    def get_policy_and_value(self, sess, state):
        pi, v = sess.run([self.ops.pi, self.ops.v], feed_dict=self._get_standard_feed_dict(state))
        return pi[0], v[0]

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
    shortname = "ff_ac"

    def __init__(self,
                 **kwargs):
        super(FFACNet, self).__init__(**kwargs)

    def _get_name_scope(self):
        return "a3c_ff_net"

    def create_architecture(self):
        conv_layers = default_conv_layers(self.vars.state_img, self._name_scope)

        if self.use_misc:
            fc_input = tf.concat(concat_dim=1, values=[conv_layers, self.vars.state_misc])
        else:
            fc_input = conv_layers

        fc1 = layers.fully_connected(fc_input, num_outputs=512, scope=self._name_scope + "/fc1",
                                     biases_initializer=tf.constant_initializer(0.1))

        pi = layers.fully_connected(fc1, num_outputs=self._actions_num, scope=self._name_scope + "/fc_pi",
                                    activation_fn=tf.nn.softmax)

        state_value = layers.linear(fc1, num_outputs=1, scope=self._name_scope + "/fc_value")

        v = tf.reshape(state_value, [-1])

        return pi, v

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
        self.recurrent_cells = None
        self.network_state = None
        # TODO make it configurable in jsons
        self._recurrent_units_num = recurrent_units_num
        super(_BaseRcurrentACNet, self).__init__(**settings)

    def _get_ru_class(self):
        raise NotImplementedError()

    def create_architecture(self, **specs):
        self.vars.sequence_length = tf.placeholder(tf.int64, [1], name="sequence_length")

        conv_layers = default_conv_layers(self.vars.state_img, self._name_scope)
        if self.use_misc:
            fc_input = tf.concat(concat_dim=1, values=[conv_layers, self.vars.state_misc])
        else:
            fc_input = conv_layers

        fc1 = layers.fully_connected(fc_input, num_outputs=self._recurrent_units_num,
                                     scope=self._name_scope + "/fc1",
                                     biases_initializer=tf.constant_initializer(0.1))

        fc1_reshaped = tf.reshape(fc1, [1, -1, self._recurrent_units_num])
        self.recurrent_cells = self._get_ru_class()(self._recurrent_units_num)
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

        pi = layers.fully_connected(reshaped_rnn_outputs, num_outputs=self._actions_num,
                                    scope=self._name_scope + "/fc_pi",
                                    activation_fn=tf.nn.softmax)
        state_value = layers.linear(reshaped_rnn_outputs, num_outputs=1, scope=self._name_scope + "/fc_value")
        v = tf.reshape(state_value, [-1])

        self.reset_state()
        return pi, v

    def reset_state(self):
        state_c = np.zeros([1, self.recurrent_cells.state_size.c], dtype=np.float32)
        state_h = np.zeros([1, self.recurrent_cells.state_size.h], dtype=np.float32)
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

    def get_policy_and_value(self, sess, state, update_state=True):
        if update_state:
            pi, v, self.network_state = sess.run([self.ops.pi, self.ops.v, self.ops.network_state],
                                                 feed_dict=self._get_standard_feed_dict(state))
        else:
            pi, v = super(_BaseRcurrentACNet, self).get_policy_and_value(sess, state)
        return pi[0], v[0]

    def get_policy(self, sess, state, update_state=True):
        if update_state:
            pi = self.network_state = sess.run([self.ops.pi, self.ops.network_state],
                                               feed_dict=self._get_standard_feed_dict(state))
        else:
            pi = super(_BaseRcurrentACNet, self).get_policy(sess, state)
        return pi[0]

    def get_value(self, sess, state, update_state=False):
        if update_state:
            v, self.network_state = sess.run([self.ops.v, self.ops.network_state],
                                             feed_dict=self._get_standard_feed_dict(state))
        else:
            v = super(_BaseRcurrentACNet, self).get_value(sess, state)
        return v

    def get_current_network_state(self):
        return self.network_state

    def has_state(self):
        return True


class BasicLstmACACNet(_BaseRcurrentACNet):
    shortname = "basic_lstm_ac"

    def __init__(self,
                 **settings
                 ):
        super(BasicLstmACACNet, self).__init__(**settings)

    def _get_name_scope(self):
        return BasicLstmACACNet.shortname

    def _get_ru_class(self):
        return tf.nn.rnn_cell.BasicLSTMCell


class LstmACACNet(_BaseRcurrentACNet):
    shortname = "lstm_ac"

    def __init__(self,
                 **settings
                 ):
        super(LstmACACNet, self).__init__(**settings)

    def _get_name_scope(self):
        return LstmACACNet.shortname

    def _get_ru_class(self):
        return tf.nn.rnn_cell.LSTMCell
