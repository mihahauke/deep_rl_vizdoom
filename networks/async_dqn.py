# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers

from util import Record

from .common import default_conv_layers, gather_2d


# TODO add dueling
class ADQNNet(object):
    def __init__(self,
                 actions_num,
                 img_shape,
                 misc_len=0,
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
        self.actions_num = actions_num
        self._name_scope = self._get_name_scope() + "_" + str(thread)

        self.params = None

        with arg_scope([layers.conv2d], data_format="NCHW"), \
             arg_scope([layers.fully_connected, layers.conv2d], activation_fn=self.activation_fn):
            # TODO make it configurable from yaml
            self.ops.q = self.create_architecture()
        self._prepare_loss_op()
        self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name_scope)

    def prepare_sync_op(self, global_network):
        global_params = global_network.get_params()
        local_params = self.get_params()
        sync_ops = [dst_var.assign(src_var) for dst_var, src_var, in zip(local_params, global_params)]
        self.ops.sync = tf.group(*sync_ops, name="SyncWithGlobal")

    def prepare_unfreeze_op(self, target_network):
        target_params = target_network.get_params()
        global_params = self.get_params()
        unfreeze_ops = [dst_var.assign(src_var) for dst_var, src_var, in zip(target_params, global_params)]
        self.ops.unfreeze = tf.group(*unfreeze_ops, name="UpdateTargetNetwork")

    def _prepare_loss_op(self):
        self.vars.a = tf.placeholder(tf.int32, [None], name="action")
        self.vars.advantage = tf.placeholder(tf.float32, [None], name="advantage")
        self.vars.target_q = tf.placeholder(tf.float32, [None], name="R")
        # TODO add summaries for entropy, policy and value

        active_q = gather_2d(self.ops.q, self.vars.a)
        self.ops.loss = 0.5 * tf.reduce_sum((active_q - self.vars.target_q) ** 2)

    def create_architecture(self):
        conv_layers = default_conv_layers(self.vars.state_img, self._name_scope)

        if self.use_misc:
            fc_input = tf.concat(values=[conv_layers, self.vars.state_misc], axis=1)
        else:
            fc_input = conv_layers

        fc1 = layers.fully_connected(fc_input, num_outputs=512, scope=self._name_scope + "/fc1")
        q = layers.linear(fc1, num_outputs=self.actions_num, scope=self._name_scope + "/q")
        return q

    def get_standard_feed_dict(self, state):
        feed_dict = {self.vars.state_img: [state[0]]}
        if self.use_misc > 0:
            if len(state[1].shape) == 1:
                misc = state[1].reshape([1, -1])
            else:
                misc = state[1]
            feed_dict[self.vars.state_misc] = misc
        return feed_dict

    def get_q_values(self, sess, state):
        q = sess.run(self.ops.q, feed_dict=self.get_standard_feed_dict(state))
        return q

    def get_params(self):
        return self.params

    def has_state(self):
        return False

    def get_current_network_state(self):
        raise NotImplementedError()

    def _get_name_scope(self):
        return "async_dqn"


class ADQNLstmNet(ADQNNet):
    def __init__(self, recurrent_units_num=256, **kwargs):
        self.network_state = None
        self.recurrent_cells = None
        self._recurrent_units_num = recurrent_units_num
        super(ADQNLstmNet, self).__init__(**kwargs)

    def has_state(self):
        return True

    def get_current_network_state(self):
        return self.network_state

    def get_standard_feed_dict(self, state):
        feed_dict = super(ADQNLstmNet, self).get_standard_feed_dict(state)
        feed_dict[self.vars.initial_network_state] = self.network_state,
        feed_dict[self.vars.sequence_length] = [1]
        return feed_dict

    def get_q_values(self, sess, state, update_state=True, initial_network_state=None):
        feed_dict = self.get_standard_feed_dict(state)
        if initial_network_state is not None:
            feed_dict[self.vars.initial_network_state] = initial_network_state

        if update_state:
            q, self.network_state = sess.run([self.ops.q, self.ops.network_state],
                                             feed_dict=feed_dict)
        else:
            q = sess.run(self.ops.q, feed_dict=feed_dict)

        return q

    def update_network_state(self, sess, state):
        self.network_state = sess.run(self.ops.network_state, feed_dict=self.get_standard_feed_dict(state))

    def reset_state(self):
        state_c = np.zeros([1, self.recurrent_cells.state_size.c], dtype=np.float32)
        state_h = np.zeros([1, self.recurrent_cells.state_size.h], dtype=np.float32)
        self.network_state = LSTMStateTuple(state_c, state_h)

    def _get_ru_class(self):
        return tf.contrib.rnn.LSTMCell

    def create_architecture(self):
        self.vars.sequence_length = tf.placeholder(tf.int64, [1], name="sequence_length")
        conv_layers = default_conv_layers(self.vars.state_img, self._name_scope)

        if self.use_misc:
            fc_input = tf.concat(values=[conv_layers, self.vars.state_misc], axis=1)
        else:
            fc_input = conv_layers
        # TODO add fc units num in settings
        fc_units_num = 256
        fc1 = layers.fully_connected(fc_input, fc_units_num, scope=self._name_scope + "/fc1")
        fc1_reshaped = tf.reshape(fc1, [1, -1, fc_units_num])

        self.recurrent_cells = self._get_ru_class()(self._recurrent_units_num)
        state_c = tf.placeholder(tf.float32, [1, self.recurrent_cells.state_size.c], name="initial_lstm_state_c")
        state_h = tf.placeholder(tf.float32, [1, self.recurrent_cells.state_size.h], name="initial_lstm_state_h")
        self.vars.initial_network_state = LSTMStateTuple(state_c, state_h)
        rnn_outputs, self.ops.network_state = tf.nn.dynamic_rnn(self.recurrent_cells,
                                                                fc1_reshaped,
                                                                initial_state=self.vars.initial_network_state,
                                                                sequence_length=self.vars.sequence_length,
                                                                scope=self._name_scope)
        reshaped_rnn_outputs = tf.reshape(rnn_outputs, [-1, self._recurrent_units_num])
        q = layers.linear(reshaped_rnn_outputs, num_outputs=self.actions_num, scope=self._name_scope + "/q")
        self.reset_state()
        return q

    def _get_name_scope(self):
        return "async_dqn_lstm"
