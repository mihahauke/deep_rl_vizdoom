# -*- coding: utf-8 -*-

from tensorflow.contrib import layers


def default_conv_layers(img_input, name_scope):
    # TODO maybe different initialization?
    conv1 = layers.conv2d(img_input, num_outputs=32, kernel_size=[8, 8], stride=4, padding="VALID",
                          scope=name_scope + "/conv1")
    conv2 = layers.conv2d(conv1, num_outputs=64, kernel_size=[4, 4], stride=2, padding="VALID",
                          scope=name_scope + "/conv2")
    conv3 = layers.conv2d(conv2, num_outputs=64, kernel_size=[3, 3], stride=1, padding="VALID",
                          scope=name_scope + "/conv3")
    conv3_flat = layers.flatten(conv3)

    return conv3_flat


def simplest_conv_layers(img_input, name_scope):
    conv1 = layers.conv2d(img_input, num_outputs=8, kernel_size=[6, 6], stride=3, padding="VALID",
                          scope=name_scope + "/conv1")
    conv2 = layers.conv2d(conv1, num_outputs=8, kernel_size=[3, 3], stride=2, padding="VALID",
                          scope=name_scope + "/conv2")
    conv2_flat = layers.flatten(conv2)

    return conv2_flat
