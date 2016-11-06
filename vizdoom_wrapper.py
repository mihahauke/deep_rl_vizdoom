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
                 last_action_input=False,
                 ignore_misc=False,
                 scenarios_path=os.path.join(vzd.__path__[0], "scenarios"),
                 **kwargs):

        if ignore_misc:
            pass
            # TODO implement misc support
        if last_action_input:
            # TODO
            raise NotImplementedError("Last_action_input support not implemented yet.")

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
        self._actions = [list(a) for a in it.product([0, 1], repeat=doom.get_available_buttons_size())]
        self.actions_num = len(self._actions)
        self._current_screen = None
        self._current_stacked_screen = np.zeros(self._img_shape, dtype=np.float32)
        self._last_reward = None
        self._terminal = None

        if not noinit:
            self.reset()

    def _set_current_screen(self):
        self._current_screen = self.preprocess(self.doom.get_state().screen_buffer)

    def preprocess(self, img):
        img = skimage.transform.resize(img, self._resolution)
        img = img.astype(np.float32)
        # TODO somehow get rid of this reshape
        img = img.reshape(list(self._resolution) + [1])
        return img

    def reset(self):
        self.doom.new_episode()

        self._set_current_screen()
        self._last_reward = 0
        self._terminal = False
        self._current_stacked_screen.fill(0)
        self._current_stacked_screen[:, :, -1] = self._current_screen[:, :, 0]

    def make_action(self, action_index):
        action = self._actions[action_index]
        self._last_reward = self.doom.make_action(action, self._frame_skip) * self._reward_scale
        self._terminal = self.doom.is_episode_finished()

        if not self._terminal:
            self._set_current_screen()
            # TODO check how efficient it is
            self._current_stacked_screen = np.append(self._current_stacked_screen[:, :, 1:], self._current_screen,
                                                     axis=2)

        return self._last_reward

    def get_current_state(self):
        return self._current_stacked_screen

    def get_total_reward(self):
        return self.doom.get_total_reward() * self._reward_scale

    def is_terminal(self):
        return self.doom.is_episode_finished()

    def get_last_reward(self):
        return self._last_reward

    def close(self):
        self.doom.close()
