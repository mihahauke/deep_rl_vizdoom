#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import ruamel.yaml as yaml
from util.parsers import parse_train_dqn_args
from util.logger import log,setup_file_logging
from util.misc import load_settings,print_settings
from dqn import DQN
from constants import *

if __name__ == "__main__":
    args = parse_train_dqn_args()
    settings = load_settings(DEFAULT_DQN_SETTINGS_FILE, args.settings_yml)

    if settings["logfile"] is not None:
        log("Setting up file logging to: {}".format(settings["logfile"]))
        setup_file_logging(settings["logfile"],add_date=True)

    log("Loaded settings:")
    print_settings(settings)

    dqn = DQN(**settings)
    dqn.train()
