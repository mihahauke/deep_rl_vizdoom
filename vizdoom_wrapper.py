# -*- coding: utf-8 -*-
import numpy as np
import itertools as it
import vizdoom as vzd
import skimage.transform
import os
from util import threadsafe_print


class VizdoomWrapper():
    def __init__(self,
                 config_file,
                 frame_skip,
                 display=False,
                 resolution=(84, 84),
                 stack_n_frames=4,
                 reward_scale=1.0,
                 noinit=False,
                 use_freedoom=False,
                 input_n_last_actions=False,
                 use_misc=True,
                 misc_scale=None,
                 scenarios_path=os.path.join(vzd.__path__[0], "scenarios"),
                 **kwargs):

        doom = vzd.DoomGame()
        if use_freedoom:
            doom.set_doom_game_path(vzd.__path__[0] + "/freedoom2.wad")
        self.doom = doom
        doom.load_config(os.path.join(scenarios_path, str(config_file)))
        doom.set_window_visible(display)
        # TODO support for colors
        doom.set_screen_format(vzd.ScreenFormat.GRAY8)
        doom.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        if not noinit:
            doom.init()

        self._stack_n_frames = stack_n_frames
        self._resolution = resolution
        self._frame_skip = frame_skip
        self._reward_scale = reward_scale

        self._img_channels = stack_n_frames
        self._img_shape = (resolution[0], resolution[1], stack_n_frames)
        # TODO allow continuous actions
        self._actions = [list(a) for a in it.product([0, 1], repeat=len(doom.get_available_buttons()))]
        self.actions_num = len(self._actions)
        self._current_screen = None
        self._current_stacked_screen = np.zeros(self._img_shape, dtype=np.float32)
        self._last_reward = None
        self._terminal = None

        self._current_stacked_misc = None
        self.input_n_last_actions = None
        self.misc_scale = False
        self.last_n_actions = None

        if use_misc:
            gvars_misc_len = len(doom.get_available_game_variables())
            if misc_scale:
                self.misc_scale = np.ones(gvars_misc_len, dtype=np.float32)
                self.misc_scale[0:len(misc_scale)] = misc_scale

            self.misc_len = gvars_misc_len * self._stack_n_frames
            if input_n_last_actions:
                self.input_n_last_actions = input_n_last_actions
                self.last_n_actions = np.zeros(self.input_n_last_actions * self.actions_num, dtype=np.float32)
                self.misc_len += len(self.last_n_actions)

            self._current_stacked_misc = np.zeros(self.misc_len, dtype=np.float32)
            self.use_misc = True
        else:
            self.misc_len = 0
            self.use_misc = False

        if not noinit:
            self.reset()

    # # TODO efficiency?
    def _update_screen(self):
        self._current_screen = self.preprocess(self.doom.get_state().screen_buffer)
        self._current_stacked_screen = np.append(self._current_stacked_screen[:, :, 1:], self._current_screen,
                                                 axis=2)

    # TODO efficiency?
    def _update_misc(self):
        game_vars = self.doom.get_state().game_variables
        if self.misc_scale:
            game_vars *= self.misc_scale
        self._current_stacked_misc[0:len(game_vars) * (self._stack_n_frames - 1)] = self._current_stacked_misc[
                                                                                    len(game_vars):-len(
                                                                                        self.last_n_actions)]
        self._current_stacked_misc[len(game_vars) * (self._stack_n_frames - 1):-len(self.last_n_actions)] = game_vars
        if self.input_n_last_actions:
            self._current_stacked_misc[-len(self.last_n_actions):] = self.last_n_actions

    def preprocess(self, img):
        img = skimage.transform.resize(img, self._resolution)
        img = img.astype(np.float32)
        # TODO somehow get rid of this reshape
        img = img.reshape(list(self._resolution) + [1])
        return img

    def reset(self):
        self.doom.new_episode()

        self._last_reward = 0
        self._terminal = False

        self._current_stacked_screen.fill(0)
        self._update_screen()

        if self.use_misc:
            if self.input_n_last_actions:
                self.last_n_actions.fill(0)
            self._current_stacked_misc.fill(0)
            self._update_misc()

    def make_action(self, action_index):
        action = self._actions[action_index]
        self._last_reward = self.doom.make_action(action, self._frame_skip) * self._reward_scale
        self._terminal = self.doom.is_episode_finished()

        if not self._terminal:
            if self.input_n_last_actions:
                self.last_n_actions[0:-self.actions_num] = self.last_n_actions[self.actions_num:]
                last_action = np.zeros(self.actions_num, dtype=np.int8)
                last_action[action_index] = 1
                self.last_n_actions[-self.actions_num:] = last_action

            self._update_screen()
            if self.use_misc:
                self._update_misc()

        return self._last_reward

    def get_current_state(self):
        return self._current_stacked_screen, self._current_stacked_misc

    def get_total_reward(self):
        return self.doom.get_total_reward() * self._reward_scale

    def is_terminal(self):
        return self.doom.is_episode_finished()

    def get_last_reward(self):
        return self._last_reward

    def close(self):
        self.doom.close()
