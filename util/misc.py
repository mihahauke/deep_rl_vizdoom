#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def print_settings(settings, level=1, tab="    ", end_with_newline=True):
    if not isinstance(settings, dict):
        raise ValueError("Settings should be a dictionary, got:".format(type(settings)))
    for k, v in settings.items():
        if isinstance(v, dict):
            print("{}{}:".format(level * tab, k))
            print_settings(v, level=level + 1, tab=tab, end_with_newline=False)
        else:
            print("{}{}: {}".format(level * tab, k, v))
    if end_with_newline:
        print()
