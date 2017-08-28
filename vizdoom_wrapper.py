# -*- coding: utf-8 -*-
import numpy as np
import itertools as it
import vizdoom as vzd
import os
import cv2
from enum import Enum
from random import sample


class VizdoomWrapper(object):
    def __init__(self,
                 config_file,
                 frameskip=4,
                 display=False,
                 vizdoom_async_mode=False,
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
        if vizdoom_async_mode:
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
        self.frameskip = frameskip
        self._reward_scale = reward_scale

        self._img_channels = stack_n_frames
        self.img_shape = (stack_n_frames, resolution[1], resolution[0])
        # TODO allow continuous actions
        self.actions = [list(a) for a in it.product([0, 1], repeat=len(doom.get_available_buttons()))]
        self.actions_num = len(self.actions)
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
        img = cv2.resize(img, self._resolution, interpolation=cv2.INTER_NEAREST)
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
            frameskip = self.frameskip
        action = self.actions[action_index]

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


class Actions(Enum):
    LEFT = "0"
    RIGHT = "1"
    SCORE = "2"
    PULL = "3"


class FakeVizdoomWrapper(object):
    def __init__(self,
                 frameskip=1,
                 display=False,
                 smooth_display=False,
                 resolution=(80, 60),
                 stack_n_frames=4,
                 reward_scale=1.0,
                 input_n_last_actions=False,
                 seed=None,
                 max_steps=300,
                 map_len=160,
                 map_height=6,
                 fov=10,
                 living_reward=-0.0001,
                 box_span=1,
                 boxes_num=10,
                 miss_penalty=-1,
                 pull_penalty=0,
                 edge_death=False,
                 edge_penalty=-0.1,
                 **kwargs):
        assert resolution[1] % map_height == 0
        assert resolution[0] % fov == 0

        self.visible = display
        self.step = 0

        self.total_reward = 0
        self.terminal = False
        self.boxes = dict()
        self.free_spaces = set()

        # CONFIGURATION params:
        self.box_span = box_span
        self.boxes_num = boxes_num
        self.miss_penalty = miss_penalty
        self.pull_penalty = pull_penalty
        self.edge_death = edge_death
        self.max_steps = max_steps
        self.living_reward = living_reward
        self.edge_penalty = edge_penalty
        ###################################

        if display and smooth_display:
            pass
            # TODO ?
        self._stack_n_frames = stack_n_frames
        assert len(resolution) == 2
        self._resolution = tuple(resolution)
        self.frameskip = frameskip
        self._reward_scale = reward_scale

        self._img_channels = stack_n_frames
        self.img_shape = (stack_n_frames, resolution[1], resolution[0])
        self.actions = list(Actions)
        self.actions_num = len(self.actions)

        self.x = None
        self.x_scale = resolution[0] // fov
        self.y_scale = resolution[1] // map_height
        self.map_len = map_len
        self.map_height = map_height
        self.map = np.zeros([self.map_len, self.map_height], dtype=np.float32)
        self.fov = fov
        self._current_stacked_screen = np.zeros(self.img_shape, dtype=np.float32)

        self.input_n_last_actions = 0
        self.last_n_actions = None

        if input_n_last_actions:
            self.use_misc = True
            self.input_n_last_actions = input_n_last_actions
            self.misc_len = self.actions_num*self.input_n_last_actions
            self.last_n_actions = np.zeros(self.misc_len, dtype=np.float32)
            self._current_stacked_misc = np.zeros(self.misc_len, dtype=np.float32)

        else:
            self.misc_len = 0
            self.use_misc = False
            self.input_n_last_actions = False
        self.reset()

    def _update_screen(self):
        current_screen = np.ones((self.fov, self.map_height))
        left = max(0, self.x - self.fov // 2)
        right = min(self.map_len, self.x + self.fov // 2)
        seen_fragment = self.map[left:right]
        loff = max(0, self.fov // 2 - self.x)
        roff = min(self.fov, self.fov + self.map_len - self.x - self.fov // 2)
        current_screen[loff:roff] = seen_fragment
        # current_screen = cv2.resize(current_screen.T, self._resolution, interpolation=cv2.INTER_CUBIC)
        current_screen = np.repeat(np.repeat(current_screen.T, self.y_scale, axis=0), self.x_scale, axis=1)
        current_screen = current_screen.reshape([-1] + list(reversed(self._resolution)))
        self._current_stacked_screen = np.append(self._current_stacked_screen[1:], current_screen, axis=0)

    def create_new_box(self):

        if self.box_span != 1:
            raise NotImplemented()
        else:
            x = sample(self.free_spaces, 1)[0]
            y = np.random.randint(0, self.map_height - 1)
            self.free_spaces.remove(x)
            self.boxes[x] = y
            self.map[x, y] = 1

    def pull_box(self):
        if self.box_span != 1:
            raise NotImplementedError()
        else:
            y = self.boxes[self.x]
            if y == 0:
                del self.boxes[self.x]
                self.free_spaces.add(self.x)
                self.map[self.x, y] = 0
                self.create_new_box()
            else:
                self.boxes[self.x] -= 1
                self.map[self.x, y] = 0
                self.map[self.x, y - 1] = 1

    def reset(self):
        self.step = 0
        self.total_reward = 0
        self.terminal = False
        self.boxes = dict()
        self.x = (self.map_len - 1) // 2
        self.map[:] = 0
        self.free_spaces = set(range(self.map_len))
        for _ in range(self.boxes_num):
            self.create_new_box()

        self._current_stacked_screen = np.zeros_like(self._current_stacked_screen)
        self._update_screen()

    def make_action(self, action_index, frameskip=None):
        if frameskip is None:
            frameskip = self.frameskip
        if self.terminal:
            raise ValueError()

        action = self.actions[action_index]
        reward = 0

        for _ in range(frameskip):
            self.step += 1
            reward += self.living_reward
            if action == Actions.LEFT:
                if self.x > 0:
                    self.x -= 1
                else:
                    if self.edge_death:
                        self.terminal = True
                    reward += self.edge_penalty
            elif action == Actions.RIGHT:
                if self.x < self.map_len - 1:
                    self.x += 1
                else:
                    if self.edge_death:
                        self.terminal = True
                    reward += self.edge_penalty
            elif action == Actions.SCORE:
                if self.map[self.x, 0] > 0:
                    reward += self.map[self.x, 0]
                    self.map[self.x, 0] = 0
                    del self.boxes[self.x]
                    self.create_new_box()
                else:
                    reward += self.miss_penalty
            elif action == Actions.PULL:
                if self.x in self.boxes:
                    self.pull_box()
                else:
                    reward += self.pull_penalty
            self.terminal = self.terminal or self.step >= self.max_steps
            if self.terminal:
                break

        if not self.terminal:
            if self.input_n_last_actions:
                self.last_n_actions[0:-self.actions_num] = self.last_n_actions[self.actions_num:]
                last_action = np.zeros(self.actions_num, dtype=np.int8)
                last_action[action_index] = 1
                self.last_n_actions[-self.actions_num:] = last_action

            self._update_screen()

        reward *= self._reward_scale
        self.total_reward += reward
        return reward

    def get_current_state(self):
        if self.is_terminal():
            return None
        return self._current_stacked_screen, self.last_n_actions

    def get_total_reward(self):
        return self.total_reward

    def is_terminal(self):
        return self.terminal

    def close(self):
        pass
