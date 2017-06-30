# -*- coding: utf-8 -*-

from tensorflow.contrib import layers
import tensorflow as tf


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
                 conv_filters_num=(32, 64, 64),
                 conv_filters_sizes=(8, 4, 3),
                 conv_strides=(4, 2, 1),
                 activation_fn="tf.nn.relu",
                 fc_units_num=256,
                 **ignored):
        self.actions_num = actions_num
        self.conv_filters_num = conv_filters_num
        self.conv_filters_sizes = conv_filters_sizes
        self.conv_strides = conv_strides
        self.activation_fn = eval(activation_fn)
        self.fc_units_num = fc_units_num

    def get_conv_layers(self, img_input, name_scope):
        return create_conv_layers(img_input, name_scope, self.conv_filters_num, self.conv_filters_sizes, self.conv_strides)
