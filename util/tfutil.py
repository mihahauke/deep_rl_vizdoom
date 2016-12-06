# -*- coding: utf-8 -*-

import tensorflow as tf


# TODO remove when they resolve the issue #5342
def gather_2d(tensor_2d, col_indices):
    """ return: tensor_2d[:, col_indices]"""
    flat = tf.reshape(tensor_2d, [-1])
    nrows = tf.shape(tensor_2d)[0]
    ncols = tensor_2d.get_shape()[1]
    add = tf.range(nrows) * ncols
    idx = col_indices + add
    return tf.gather(flat, idx)
