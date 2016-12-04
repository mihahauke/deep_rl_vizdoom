#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from util.coloring import green, blue
from time import time
from random import randint, choice
from vizdoom_wrapper import VizdoomWrapper
import vizdoom as vzd
from tqdm import trange
import itertools as it


def measure(name, iters=5000, **settings):
    print(name)
    for k, v in settings.items():
        print("\t{}: {}".format(k, v))

    # Vizdoom wrapper
    doom_wrapper = VizdoomWrapper(**settings)
    start = time()
    for _ in trange(iters, leave=False):
        current_img, current_misc = doom_wrapper.get_current_state()
        action_index = randint(0, doom_wrapper.actions_num - 1)
        doom_wrapper.make_action(action_index)

        if doom_wrapper.is_terminal():
            doom_wrapper.reset()
    end = time()
    wrapper_t = (end - start)

    # Vanilla vizdoom:
    doom = vzd.DoomGame()
    if "scenarios_path" not in settings:
        scenarios_path = vzd.__path__[0] + "/scenarios"
    else:
        scenarios_path = settings["scenarios_path"]
    config_file = scenarios_path + "/" + settings["config_file"]
    doom.load_config(config_file)
    doom.set_window_visible(False)
    doom.set_screen_format(vzd.ScreenFormat.GRAY8)
    doom.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    doom.init()
    actions = [list(a) for a in it.product([0, 1], repeat=len(doom.get_available_game_variables()))]
    start = time()
    frame_skip = settings["frame_skip"]
    for _ in trange(iters, leave=False):
        if doom.is_episode_finished():
            doom.new_episode()
        doom.make_action(choice(actions), frame_skip)

    end = time()
    vanilla_t = end - start
    print(green("\twrapper: {:0.2f} steps/s".format(iters / wrapper_t)))
    print(green("\twrapper: {:0.2f} s/1000 steps".format(wrapper_t / iters * 1000)))
    print(blue("\tvanilla: {:0.2f} steps/s".format(iters / vanilla_t)))
    print(blue("\tvanilla: {:0.2f} s/1000 steps\n".format(vanilla_t / iters * 1000)))


hg_full = {"config_file": "health_gathering.cfg",
           "frame_skip": 10, "stack_n_frames": 4,
           "input_n_last_actions": 4,
           "use_misc": True,
           }
hg_min = {"config_file": "health_gathering.cfg",
          "frame_skip": 10, "stack_n_frames": 1,
          "input_n_last_actions": None,
          "use_misc": False,
          }
hg_no_skip = {"config_file": "health_gathering.cfg",
              "frame_skip": 1, "stack_n_frames": 4,
              "input_n_last_actions": 4,
              "use_misc": True,
              }
hg_highres = {"config_file": "health_gathering.cfg",
              "frame_skip": 10, "stack_n_frames": 4,
              "input_n_last_actions": 4,
              "use_misc": True,
              "resuolution": (160, 120)
              }
pacman_full = {"config_file": "pacman.cfg",
               "frame_skip": 10, "stack_n_frames": 4,
               "input_n_last_actions": 4,
               "use_misc": True,
               "scenarios_path": "../benchmark_scenarios/scenarios",
               }
pacman_min = {"config_file": "pacman.cfg",
              "frame_skip": 10, "stack_n_frames": 1,
              "input_n_last_actions": 0,
              "use_misc": False,
              "scenarios_path": "../benchmark_scenarios/scenarios"
              }

measure("PACMAN_FULL", **pacman_full)
measure("PACMAN_MIN", **pacman_min)
measure("HG_FULL", **hg_full)
measure("HG_MINIMAL", **hg_min)
measure("HG_NO_SKIP", **hg_no_skip)
measure("HG_HIGHRES", **hg_highres)
