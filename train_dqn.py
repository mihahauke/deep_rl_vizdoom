#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from util.parsers import parse_train_dqn_args
import os
from dqn import DQN

if __name__ == "__main__":
    # TODO make tqdm work when stderr is redirected
    # TODO print setup info on stderr and stdout
    args = parse_train_dqn_args()
    # TODO override settings according to args

    default_settings_filepath = "settings/dqn/defaults.json"
    override_settings_filepath = args.settings_json
    dqn_settings = json.load(open(default_settings_filepath))
    override_settings = json.load(open(override_settings_filepath))
    dqn_settings.update(override_settings)

    if not os.path.isdir(dqn_settings["models_path"]):
        os.makedirs(dqn_settings["models_path"])
    if not os.path.isdir(dqn_settings["logdir"]):
        os.makedirs(dqn_settings["logdir"])

    dqn = DQN(**dqn_settings)
    dqn.train()
