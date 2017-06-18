#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import tensorflow as tf
from util.parsers import parse_train_dqn_args
from util.logger import log,setup_file_logger
from util.misc import load_settings,print_settings
from dqn import DQN
from paths import *

if __name__ == "__main__":

    args = parse_train_dqn_args()
    settings = load_settings(DEFAULT_DQN_SETTINGS_FILE, args.settings_yml)

    setup_file_logger(settings["logfile"], add_date=True)

    log("Loaded settings:")
    print_settings(settings)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(settings["tf_log_level"])

    dqn = DQN(**settings)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.InteractiveSession(config=config)
    session.run(tf.global_variables_initializer())

    dqn.train(session)
