#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from paths import *
import ruamel.yaml
from .logger import log
from .coloring import red
import numpy as np

import tensorflow as tf


def setup_vector_summaries(name_prefix):
    placeholder = tf.placeholder(tf.float32, None)
    scores_mean = tf.reduce_mean(placeholder)
    scores_std = tf.sqrt(tf.reduce_mean((placeholder - scores_mean) ** 2))
    scores_min = tf.reduce_min(placeholder)
    scores_max = tf.reduce_max(placeholder)

    summary_mean = tf.summary.scalar(name_prefix + "/mean", scores_mean)
    summary_std = tf.summary.scalar(name_prefix + "/std", scores_std)
    summary_min = tf.summary.scalar(name_prefix + "/min", scores_min)
    summary_max = tf.summary.scalar(name_prefix + "/max", scores_max)

    summaries = [summary_mean, summary_std, summary_min, summary_max]
    return placeholder, summaries


def print_settings(settings, level=1, indent="    ", end_with_newline=True):
    if not isinstance(settings, dict):
        raise ValueError("Settings should be a dictionary, got:".format(type(settings)))

    space_len = 4
    max_key_len = max([len(k) for k in settings])
    for key in sorted(settings):
        value = settings[key]

        if isinstance(value, dict):
            log("{}{}:".format(level * indent, key))
            print_settings(value, level=level + 1, indent=indent, end_with_newline=False)
        else:
            spacing = " " * (space_len + max_key_len - len(key))
            log("{}{}:{}{}".format(level * indent, key, spacing, value))
    if end_with_newline:
        log("")


def load_settings(default_settings_file, override_settings_files):
    yaml = ruamel.yaml.YAML()
    yaml.allow_duplicate_keys = False
    try:
        log("Loading common default settings from: " + DEFAULT_COMMON_SETTINGS_FILE)
        settings = dict(yaml.load(open(DEFAULT_COMMON_SETTINGS_FILE)))
        log("Loading default settings from: " + default_settings_file)
        settings.update(yaml.load(open(default_settings_file)))

        for settings_fpath in override_settings_files:
            log("Loading settings from: " + settings_fpath)
            override_settings = yaml.load(open(settings_fpath))
            settings.update(override_settings)
        log("Loaded settings.")
    except ruamel.yaml.constructor.DuplicateKeyError as ex:
        log(red(ex))
        log(red("Aborting!"))
        exit(1)

    return settings


def gray_square(val):
    val = 232 + int(val * 23)
    return "\x1b[48;5;{}m  {}".format(val, '\x1b[0m')


def string_heatmap(mat, x_labels=None, y_labels=None):
    # TODO refactor this piec of shit code some day
    mat = mat - mat.min()
    mat /= mat.max()
    action_maxes = mat.max(1)
    action_maxes[action_maxes == 0] = 1
    anorm_mat = (mat.T / action_maxes).T
    mat_sum = np.sum(mat)
    if y_labels is None:
        y_labels = [str(i) for i in range(mat.shape[0])]
    if x_labels is None:
        x_labels = [str(i) for i in range(mat.shape[1])]

    x_labels_len = max([len(str(l)) for l in y_labels])

    str_mat = ""
    space = "    "
    for row_i in range(mat.shape[0]):
        str_mat += str(y_labels[row_i]) + " " * (x_labels_len + 1 - len(str(y_labels[row_i])))
        str_mat += " {:2.0f}%  ".format(np.sum(mat[row_i]) / mat_sum * 100)
        for column_j in range(mat.shape[1]):
            str_mat += gray_square(mat[row_i, column_j])

        str_mat += space
        for column_j in range(mat.shape[1]):
            str_mat += gray_square(anorm_mat[row_i, column_j])
        str_mat += "\n"

    str_mat += " " * (x_labels_len + 1)
    x_axis = ""
    for yl in x_labels:
        x_axis += (str(yl) + "  ")[0:2]
    str_mat += "      " + x_axis + space + x_axis

    return str_mat
