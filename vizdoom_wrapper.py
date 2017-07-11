# -*- coding: utf-8 -*-
import numpy as np
import itertools as it
import vizdoom as vzd
import os
import cv2
import time


class VizdoomWrapper(object):
    def __init__(self,
                 config_file,
                 frameskip=4,
                 display=False,
                 async=False,
                 smooth_display=False,
                 fps=35,
                 resolution=(80, 60),
                 vizdoom_resolution="RES_160X120",
                 stack_n_frames=4,
                 reward_scale=1.0,
                 noinit=False,
                 force_freedoom=False,
                 input_n_last_actions=False,
                 use_misc=True,
                 misc_scale=None,
                 hide_hood=False,
                 scenarios_path=os.path.join(vzd.__path__[0], "scenarios"),
                 seed=None,
                 **kwargs):
        doom = vzd.DoomGame()

        if force_freedoom:
            doom.set_doom_game_path(vzd.__path__[0] + "/freedoom2.wad")

        doom.load_config(os.path.join(scenarios_path, str(config_file)))
        if hide_hood:
            doom.set_render_hud(not hide_hood)

        doom.set_window_visible(display)
        if display and smooth_display:
            doom.add_game_args("+viz_render_all 1")
        # TODO support for colors
        doom.set_screen_format(vzd.ScreenFormat.GRAY8)
        if async:
            doom.set_mode(vzd.Mode.ASYNC_PLAYER)
            doom.set_ticrate(int(fps))
        else:
            doom.set_mode(vzd.Mode.PLAYER)

        if seed is not None:
            doom.set_seed(seed)

        # TODO if eval fails, show some warning
        doom.set_screen_resolution(eval("vzd.ScreenResolution." + vizdoom_resolution))
        if not noinit:
            doom.init()
        self.doom = doom

        self._stack_n_frames = stack_n_frames
        assert len(resolution) == 2
        self._resolution = tuple(resolution)
        self._frameskip = frameskip
        self._reward_scale = reward_scale

        self._img_channels = stack_n_frames
        self.img_shape = (stack_n_frames, resolution[1], resolution[0])
        # TODO allow continuous actions
        self._actions = [list(a) for a in it.product([0, 1], repeat=len(doom.get_available_buttons()))]
        self.actions_num = len(self._actions)
        self._current_screen = None
        self._current_stacked_screen = np.zeros(self.img_shape, dtype=np.float32)

        self._current_stacked_misc = None
        self.input_n_last_actions = 0
        self.misc_scale = None
        self.last_n_actions = None

        gvars_misc_len = len(doom.get_available_game_variables())
        if use_misc and (gvars_misc_len or input_n_last_actions):
            if misc_scale is not None:
                assert len(misc_scale) <= gvars_misc_len
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

    def _update_screen(self):
        self._current_screen = self.preprocess(self.doom.get_state().screen_buffer)
        self._current_stacked_screen = np.append(self._current_stacked_screen[1:], self._current_screen, axis=0)

    def _update_misc(self):
        # TODO add support for input_n_actions without game variables
        game_vars = self.doom.get_state().game_variables
        if self.misc_scale is not None:
            game_vars *= self.misc_scale
        if self.input_n_last_actions:
            game_vars_end_i = -len(self.last_n_actions)
        else:
            game_vars_end_i = len(self._current_stacked_misc)
        self._current_stacked_misc[0:len(game_vars) * (self._stack_n_frames - 1)] = self._current_stacked_misc[
                                                                                    len(game_vars):game_vars_end_i]
        self._current_stacked_misc[len(game_vars) * (self._stack_n_frames - 1):game_vars_end_i] = game_vars
        if self.input_n_last_actions:
            self._current_stacked_misc[-len(self.last_n_actions):] = self.last_n_actions

    def preprocess(self, img):
        # TODO check what's the difference in practice
        # img = cv2.resize(img, self._resolution, interpolation=cv2.INTER_CUBIC)
        # img = cv2.resize(img, self._resolution, interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, self._resolution, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = img.reshape([1] + list(img.shape))
        return img

    def reset(self):
        self.doom.new_episode()
        self._current_stacked_screen = np.zeros_like(self._current_stacked_screen)
        self._update_screen()

        if self.use_misc:
            if self.input_n_last_actions:
                self.last_n_actions.fill(0)
            self._current_stacked_misc = np.zeros_like(self._current_stacked_misc)
            self._update_misc()

    def make_action(self, action_index, frameskip=None):
        if frameskip is None:
            frameskip = self._frameskip
        action = self._actions[action_index]

        reward = self.doom.make_action(action, frameskip) * self._reward_scale

        if not self.doom.is_episode_finished():
            if self.input_n_last_actions:
                self.last_n_actions[0:-self.actions_num] = self.last_n_actions[self.actions_num:]
                last_action = np.zeros(self.actions_num, dtype=np.int8)
                last_action[action_index] = 1
                self.last_n_actions[-self.actions_num:] = last_action

            self._update_screen()
            if self.use_misc:
                self._update_misc()

        return reward

    def get_current_state(self):
        if self.doom.is_episode_finished():
            return None
        return self._current_stacked_screen, self._current_stacked_misc

    def get_total_reward(self):
        return self.doom.get_total_reward() * self._reward_scale

    def is_terminal(self):
        return self.doom.is_episode_finished()

    def close(self):
        self.doom.close()
