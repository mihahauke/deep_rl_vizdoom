# -*- coding: utf-8 -*-

import tensorflow as tf


def gather_2d(tensor_2d, col_indices):
    """ return: tensor_2d[:, col_indices]"""
    return tf.gather_nd(tensor_2d, tf.stack([tf.range(tf.shape(tensor_2d)[0]), col_indices], 1))
