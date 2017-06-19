#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tensorflow as tf
from util.parsers import parse_test_dqn_args
from dqn import DQN
from util.logger import log
from util.misc import load_settings, print_settings
import numpy as np
from paths import *
import os

if __name__ == "__main__":

    args = parse_test_dqn_args()
    print(args)
    settings = load_settings(DEFAULT_DQN_SETTINGS_FILE, args.settings_yml)

    log("Loaded settings.")
    if args.print_settings:
        print_settings(settings)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(settings["tf_log_level"])

    settings["display"] = not args.hide_window
    settings["async"] = not args.hide_window
    settings["smooth_display"] = not args.agent_view
    settings["fps"] = args.fps
    settings["seed"] = args.seed

    dqn = DQN(**settings)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.InteractiveSession(config=config)
    session.run(tf.global_variables_initializer())
    # TODO why is TF info log still displayed when restoring?
    dqn.load_model(session, args.model)

    log("\nScores: ")
    scores = []

    for _ in range(args.episodes_num):
        reward = dqn.run_test_episode(session)
        scores.append(reward)
        print("{0:3f}".format(reward))
    print()
    log("\nMean score: {:0.3f}".format(np.mean(scores)))
