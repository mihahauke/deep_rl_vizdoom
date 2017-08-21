#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from paths import *
import ruamel.yaml as yaml
from .logger import log

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
    log("Loading common default settings from: " + DEFAULT_COMMON_SETTINGS_FILE)
    settings = yaml.safe_load(open(DEFAULT_COMMON_SETTINGS_FILE))
    log("Loading default settings from: " + default_settings_file)
    settings.update(yaml.safe_load(open(default_settings_file)))

    for settings_fpath in override_settings_files:
        log("Loading settings from: " + settings_fpath)
        override_settings = yaml.safe_load(open(settings_fpath))
        settings.update(override_settings)
    print("Loaded settings.")
    return settings
