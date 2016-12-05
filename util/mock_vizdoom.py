#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class MockDoomState(object):
    def __init__(self, buffer):
        self.screen_buffer = buffer
        self.game_variables = None


class MockDoomGame(object):
    def __init__(self, **ignore):
        self.total_reward = None
        self.terminal = None
        self.img = None
        self.steps = None
        self.strip_width = 40
        self.new_episode()

    def get_state(self):
        return MockDoomState(self.img.copy())

    def new_episode(self):
        self.img = np.zeros([120, 160], dtype=np.uint8)
        x = np.random.randint(0, 160 - self.strip_width)
        self.img[:, x:x + self.strip_width] = 255
        self.total_reward = 0.0
        self.terminal = False
        self.steps = 0

    def make_action(self, action, skip):
        reward = -1.0
        if action[0] and action[1]:
            self.img = np.roll(self.img, -1)
        elif action[1] and action[0]:
            self.img = np.roll(self.img, 1)
        elif action[1] and action[1]:
            if self.img[0, 80] == 255:
                reward += 100.0
            else:
                reward -= 5.0

        self.total_reward += reward
        self.steps += 1
        if self.steps >= 75 or reward == 99.0:
            self.terminal = True
        return reward

    def is_episode_finished(self):
        return self.terminal

    def get_total_reward(self):
        return self.total_reward

    def get_available_buttons(self):
        return ["one", "two"]

    def get_available_game_variables(self):
        return []
