#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from train_async import train_async

from paths import DEFAULT_ADQN_SETTINGS_FILE
from util.parsers import parse_train_adqn_args
import os

from util.misc import print_settings, load_settings
from util.logger import setup_file_logger, log

if __name__ == "__main__":
    args = parse_train_adqn_args()

    settings = load_settings(DEFAULT_ADQN_SETTINGS_FILE, args.settings_yml)

    if settings["logfile"] is not None:
        log("Setting up file logging to: {}".format(settings["logfile"]))
        setup_file_logger(settings["logfile"], add_date=True)

    log("Settings:")
    print_settings(settings)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(settings["tf_log_level"])

    train_async(q_learning=True, settings=settings)
