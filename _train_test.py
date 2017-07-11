# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from time import strftime

from util.logger import setup_file_logger, log
from util.misc import load_settings, print_settings

from util.parsers import parse_train_dqn_args, parse_test_dqn_args
from util.parsers import parse_train_a3c_args, parse_test_a3c_args
from util.parsers import parse_train_adqn_args, parse_test_adqn_args

import numpy as np
from paths import *


def _test_common(args, settings):
    log("Loaded settings.")
    if args.print_settings:
        print_settings(settings)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(settings["tf_log_level"])

    settings["display"] = not args.hide_window
    settings["vizdoom_async_mode"] = not args.hide_window
    settings["smooth_display"] = not args.agent_view
    settings["fps"] = args.fps
    settings["seed"] = args.seed
    settings["write_summaries"] = False
    settings["test_only"] = True


def _train_common(settings):
    if settings["logfile"] is not None:
        setup_file_logger(settings["logfile"], add_date=True)

    log("Settings:")
    print_settings(settings)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(settings["tf_log_level"])


def train_dqn():
    args = parse_train_dqn_args()
    settings = load_settings(DEFAULT_DQN_SETTINGS_FILE, args.settings_yml)
    _train_common(settings)

    from _dqn_algo import DQN
    dqn = DQN(**settings)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.InteractiveSession(config=config)
    session.run(tf.global_variables_initializer())

    dqn.train(session)


def train_a3c():
    args = parse_train_a3c_args()
    settings = load_settings(DEFAULT_A3C_SETTINGS_FILE, args.settings_yml)
    _train_common(settings)

    from _async_algo import train_async
    train_async(q_learning=False, settings=settings)


def train_adqn():
    args = parse_train_adqn_args()
    settings = load_settings(DEFAULT_ADQN_SETTINGS_FILE, args.settings_yml)
    _train_common(settings)

    from _async_algo import train_async
    train_async(q_learning=True, settings=settings)


def test_dqn():
    args = parse_test_dqn_args()
    settings = load_settings(DEFAULT_DQN_SETTINGS_FILE, args.settings_yml)

    _test_common(args, settings)

    from _dqn_algo import DQN
    dqn = DQN(**settings)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.InteractiveSession(config=config)
    session.run(tf.global_variables_initializer())
    dqn.load_model(session, args.model)

    log("\nScores: ")
    scores = []
    for _ in range(args.episodes_num):
        reward = dqn.run_test_episode(session)
        scores.append(reward)
        print("{0:3f}".format(reward))
    print()
    log("\nMean score: {:0.3f}".format(np.mean(scores)))


def test_a3c():
    args = parse_test_a3c_args()
    settings = load_settings(DEFAULT_ADQN_SETTINGS_FILE, args.settings_yml)
    _test_common(args, settings)

    from _async_algo import test_async
    test_async(q_learning=False, settings=settings, modelfile=args.model, eps=args.episodes_num)


def test_adqn():
    args = parse_test_adqn_args()
    settings = load_settings(DEFAULT_ADQN_SETTINGS_FILE, args.settings_yml)
    _test_common(args, settings)

    from _async_algo import test_async
    test_async(q_learning=True, settings=settings, modelfile=args.model, eps=args.episodes_num)
