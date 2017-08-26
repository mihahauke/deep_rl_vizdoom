# -*- coding: utf-8 -*-

from tensorflow.contrib import layers
import tensorflow as tf
from util import Record


def create_conv_layers(img_input, name_scope, conv_layers=(32, 64, 64), conv_sizes=(8, 4, 3), conv_strides=(4, 2, 1)):
    if len(conv_layers) != len(conv_sizes) or len(conv_layers) != len(conv_strides):
        raise ValueError("Legnths of layers, kernels and strides should be equal. Got:"
                         "\tconv_layers: {}"
                         "\tkernels: {}"
                         "\tconv_strides: {}".format(conv_layers, conv_sizes, conv_strides))

    last_layer = img_input
    for i, [nout, ksize, stride] in enumerate(zip(conv_layers, conv_sizes, conv_strides)):
        last_layer = layers.conv2d(last_layer, num_outputs=nout,
                                   kernel_size=ksize, stride=stride,
                                   padding="VALID", scope=name_scope + "/conv" + str(i))

    last_layer_flattened = layers.flatten(last_layer)
    return last_layer_flattened


def gather_2d(tensor_2d, col_indices):
    """ return: tensor_2d[:, col_indices]"""
    col_indices = tf.to_int32(col_indices)
    res = tf.gather_nd(tensor_2d, tf.stack([tf.range(tf.shape(tensor_2d)[0]), col_indices], 1))
    return res


class _BaseNetwork(object):
    def __init__(self,
                 actions_num,
                 img_shape,
                 misc_len=0,
                 conv_filters_num=(32, 64, 64),
                 conv_filters_sizes=(8, 4, 3),
                 conv_strides=(4, 2, 1),
                 activation_fn="tf.nn.relu",
                 init_bias=0.1,
                 fc_units_num=256,
                 **ignored):
        self.actions_num = actions_num
        self.init_bias = init_bias
        self.conv_filters_num = conv_filters_num
        self.conv_filters_sizes = conv_filters_sizes
        self.conv_strides = conv_strides
        self.activation_fn = eval(activation_fn)
        self.fc_units_num = fc_units_num
        self.ops = Record()
        self.vars = Record()
        self.use_misc = misc_len > 0
        self.params = None

        self.vars.state_img = tf.placeholder(tf.float32, [None] + list(img_shape), name="state_img")
        if self.use_misc:
            self.vars.state_misc = tf.placeholder(tf.float32, [None, misc_len], name="state_misc")
        else:
            self.vars.state_misc = None

    def get_input_layers(self, img_input=None, misc_input=None, name_scope=None):
        if img_input is None:
            img_input = self.vars.state_img

        if name_scope is None:
            name_scope = self._name_scope

        conv_layers = create_conv_layers(img_input, name_scope, self.conv_filters_num, self.conv_filters_sizes,
                                         self.conv_strides)
        if self.use_misc:
            if misc_input is None:
                misc_input = self.vars.state_misc
            fc_input = tf.concat(values=[conv_layers, misc_input], axis=1)
        else:
            fc_input = conv_layers

        return fc_input
