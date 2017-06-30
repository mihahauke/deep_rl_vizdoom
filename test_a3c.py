#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from async import test_async

from paths import DEFAULT_ADQN_SETTINGS_FILE
from util.parsers import parse_test_a3c_args
import os

from util.misc import print_settings, load_settings
from util.logger import log

if __name__ == "__main__":
    args = parse_test_a3c_args()

    settings = load_settings(DEFAULT_ADQN_SETTINGS_FILE, args.settings_yml)

    log("Loaded settings.")
    if args.print_settings:
        print_settings(settings)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(settings["tf_log_level"])

    settings["display"] = not args.hide_window
    settings["async"] = not args.hide_window
    settings["smooth_display"] = not args.agent_view
    settings["fps"] = args.fps
    settings["seed"] = args.seed
    settings["write_summaries"] = False
    settings["test_only"] = True

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(settings["tf_log_level"])
    test_async(q_learning=False, settings=settings, modelfile=args.model, eps=args.episodes_num)
