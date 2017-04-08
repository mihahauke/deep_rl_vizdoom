#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import ruamel.yaml as yaml
from util.parsers import parse_train_dqn_args
import os
from dqn import DQN
from util.misc import print_settings
from constants import DEFAULT_DQN_SETTINGS_FILE

if __name__ == "__main__":
    # TODO make tqdm work when stderr is redirected
    # TODO print setup info on stderr and stdout
    args = parse_train_dqn_args()

    default_settings_filepath = DEFAULT_DQN_SETTINGS_FILE
    print("Loading default settings from:", default_settings_filepath)
    dqn_settings = yaml.safe_load(open(default_settings_filepath))

    for settings_fpath in args.settings_yml:
        print("Loading settings from:", settings_fpath)
        override_settings = yaml.safe_load(open(settings_fpath))
        dqn_settings.update(override_settings)

    print("Loaded settings:")
    print_settings(dqn_settings)

    if not os.path.isdir(dqn_settings["models_path"]):
        os.makedirs(dqn_settings["models_path"])
    if not os.path.isdir(dqn_settings["logdir"]):
        os.makedirs(dqn_settings["logdir"])

    dqn = DQN(**dqn_settings)
    dqn.train()
