# -*- coding: utf-8 -*-

import os
from time import strftime

import numpy as np
import tensorflow as tf

from paths import *
from util.logger import setup_file_logger, log
from util.misc import load_settings, print_settings
from util.parsers import parse_train_a3c_args, parse_test_a3c_args
from util.parsers import parse_train_adqn_args, parse_test_adqn_args
from util.parsers import parse_train_dqn_args, parse_test_dqn_args
from util import ensure_parent_directories
import ruamel.yaml as yaml


def _test_common(args, settings):
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
    run_id_string = "{}/{}".format(settings["network_class"], strftime(settings["date_format"]))

    if settings["run_tag"] is not None:
        run_id_string = str(settings["run_tag"]) + "/" + run_id_string

    if settings["logdir"] is not None:
        logfile = os.path.join(settings["logdir"], settings["scenario_tag"] + "_" + run_id_string.replace("/", "_"))
        setup_file_logger(logfile)
    settings["run_id_string"] = run_id_string
    log("Settings:")
    print_settings(settings)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(settings["tf_log_level"])
    model_dir = os.path.join(settings["models_path"], run_id_string)
    model_file = os.path.join(model_dir, "model")
    settings_output_file = os.path.join(model_dir, "setings.yml")

    ensure_parent_directories(settings_output_file)
    log("Saving settings to: {}".format(settings_output_file))
    yaml.safe_dump(settings, open(settings_output_file, "w"))

    return model_file


def train_dqn():
    args = parse_train_dqn_args()
    settings = load_settings(DEFAULT_DQN_SETTINGS_FILE, args.settings_yml)
    if args.run_tag is not None:
        settings["run_tag"] = args.run_tag
        model_savefile = _train_common(settings)

    from _dqn_algo import DQN
    dqn = DQN(model_savefile=model_savefile, **settings)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.InteractiveSession(config=config)
    session.run(tf.global_variables_initializer())

    dqn.train(session)


def train_a3c():
    args = parse_train_a3c_args()
    settings = load_settings(DEFAULT_A3C_SETTINGS_FILE, args.settings_yml)
    if args.run_tag is not None:
        settings["run_tag"] = args.run_tag
    model_savefile = _train_common(settings)

    from _async_algo import train_async
    train_async(model_savefile=model_savefile, q_learning=False, settings=settings)


def train_adqn():
    args = parse_train_adqn_args()
    settings = load_settings(DEFAULT_ADQN_SETTINGS_FILE, args.settings_yml)
    if args.run_tag is not None:
        settings["run_tag"] = args.run_tag
    model_savefile = _train_common(settings)

    from _async_algo import train_async
    train_async(model_savefile=model_savefile, q_learning=True, settings=settings)


def test_dqn():
    args = parse_test_dqn_args()
    # TODO load settings that are with the model
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
    # TODO print scores to file


def test_a3c():
    args = parse_test_a3c_args()
    # TODO load settings that are with the model
    settings = load_settings(DEFAULT_A3C_SETTINGS_FILE, args.settings_yml)
    _test_common(args, settings)

    from _async_algo import test_async
    test_async(q_learning=False,
               settings=settings,
               modelfile=args.model,
               eps=args.episodes_num,
               deterministic=bool(args.deterministic),
               output=args.output)


def test_adqn():
    args = parse_test_adqn_args()
    # TODO load settings that are with the model
    settings = load_settings(DEFAULT_ADQN_SETTINGS_FILE, args.settings_yml)
    _test_common(args, settings)

    from _async_algo import test_async
    test_async(q_learning=True,
               settings=settings,
               modelfile=args.model,
               eps=args.episodes_num,
               deterministic=args.deterministic,
               output=args.output)
