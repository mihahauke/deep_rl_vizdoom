#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from constants import *
import ruamel.yaml as yaml
from .logger import log


def print_settings(settings, level=1, tab="    ", end_with_newline=True):
    if not isinstance(settings, dict):
        raise ValueError("Settings should be a dictionary, got:".format(type(settings)))
    for k, v in settings.items():
        if isinstance(v, dict):
            log("{}{}:".format(level * tab, k))
            print_settings(v, level=level + 1, tab=tab, end_with_newline=False)
        else:
            log("{}{}: {}".format(level * tab, k, v))
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

    return settings
