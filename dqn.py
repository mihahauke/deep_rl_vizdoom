# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from time import strftime

from vizdoom_wrapper import VizdoomWrapper
from tqdm import trange
from random import random, randint
from replay_memory import ReplayMemory
from time import time
from util.coloring import red, green, blue
from util import sec_to_str
import networks


class DQN(object):
    def __init__(self,
                 network_type="DQNNet",
                 write_summaries=True,
                 epochs=100,
                 train_steps_per_epoch=1000000,
                 test_episodes_per_epoch=100,
                 run_tests=True,
                 initial_epsilon=1.0,
                 final_epsilon=0.0000,
                 epsilon_decay_steps=10e07,
                 epsilon_decay_start_step=2e05,
                 frozen_steps=5000,
                 batchsize=32,
                 memory_capacity=10000,
                 update_pattern=(4, 4),
                 prioritized_memory=False,
                 **settings):

        if prioritized_memory:
            raise NotImplementedError("Prioritized memory not implemented")
            # TODO
            pass

        self.update_pattern = update_pattern
        self.write_summaries = write_summaries
        self._settings = settings
        date_string = strftime("%d.%m.%y-%H:%M")
        self._run_string = "{}/{}/{}".format(settings["base_tag"], network_type, date_string)
        self.train_steps_per_epoch = train_steps_per_epoch
        self._run_tests = test_episodes_per_epoch > 0 and run_tests
        self.test_episodes_per_epoch = test_episodes_per_epoch
        self._epochs = np.float32(epochs)

        self.doom_wrapper = VizdoomWrapper(**settings)
        self.doom_wrapper = VizdoomWrapper(**settings)
        misc_len = self.doom_wrapper.misc_len
        img_shape = self.doom_wrapper.img_shape
        self.use_misc = self.doom_wrapper.use_misc
        self.actions_num = self.doom_wrapper.actions_num
        self.replay_memory = ReplayMemory(img_shape, misc_len, batch_size=batchsize, capacity=memory_capacity)
        self.network = eval("networks." + network_type)(actions_num=self.actions_num, img_shape=img_shape,
                                                        misc_len=misc_len,
                                                        **settings)

        self.batchsize = batchsize
        self.frozen_steps = frozen_steps

        self._train_writer = None
        self._test_writer = None
        self._summaries = None
        self._saver = None

        self._model_savefile = settings["models_path"] + "/" + self._run_string
        if self.write_summaries:
            self.score = tf.placeholder(tf.float32)
            score_summary = tf.summary.scalar(self._run_string + "/mean_score", self.score)
            self._summaries = tf.summary.merge([score_summary])

        self.steps = 0
        # TODO epoch as tf variable?
        self._epoch = 1

        # Epsilon
        self.epsilon_decay_rate = (initial_epsilon - final_epsilon) / epsilon_decay_steps
        self.epsilon_decay_start_step = epsilon_decay_start_step
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon

    def get_current_epsilon(self):
        eps = self.initial_epsilon - (self.steps - self.epsilon_decay_start_step) * self.epsilon_decay_rate
        return np.clip(eps, self.final_epsilon, 1.0)

    def print_epoch_log(self, prefix, scores, steps, epoch_time):
        mean_score = np.mean(scores)
        score_std = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        episodes = len(scores)

        steps_per_sec = steps / epoch_time
        mil_steps_per_hour = steps_per_sec * 3600 / 1000000.0
        print(
            "{}: Episodes: {}, mean: {}, min: {}, max: {}, "
            " Speed: {:.0f} STEPS/s, {:.2f}M STEPS/hour, time: {}".format(
                prefix,
                episodes,
                green("{:0.3f}±{:0.2f}".format(mean_score, score_std)),
                red("{:0.3f}".format(min_score)),
                blue("{:0.3f}".format(max_score)),
                steps_per_sec,
                mil_steps_per_hour,
                sec_to_str(epoch_time)
            ))

    def train(self):
        # Maybe make use of the fact that it's interactive
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.InteractiveSession(config=config)
        session.run(tf.global_variables_initializer())

        # Prefill replay memory:
        for _ in trange(self.replay_memory.capacity, leave=False, desc="Filling replay memory."):
            if self.doom_wrapper.is_terminal():
                self.doom_wrapper.reset()
            s1 = self.doom_wrapper.get_current_state()
            action_index = randint(0, self.actions_num - 1)
            reward = self.doom_wrapper.make_action(action_index)
            terminal = self.doom_wrapper.is_terminal()
            s2 = self.doom_wrapper.get_current_state()
            self.replay_memory.add_transition(s1, action_index, s2, reward, terminal)

        overall_start_time = time()
        self.network.update_target_network(session)
        while self._epoch <= self._epochs:
            self.doom_wrapper.reset()
            train_scores = []
            test_scores = []
            train_start_time = time()

            for _ in trange(self.train_steps_per_epoch, leave=False, desc="Training, epoch {}".format(self._epoch)):
                self.steps += 1
                s1 = self.doom_wrapper.get_current_state()

                if random() <= self.get_current_epsilon():
                    action_index = randint(0, self.actions_num - 1)
                else:
                    action_index = self.network.get_action(session, s1)
                reward = self.doom_wrapper.make_action(action_index)
                terminal = self.doom_wrapper.is_terminal()
                s2 = self.doom_wrapper.get_current_state()
                self.replay_memory.add_transition(s1, action_index, s2, reward, terminal)

                if self.steps % self.update_pattern[0] == 0:
                    for _ in range(self.update_pattern[1]):
                        self.network.train_batch(session, self.replay_memory.get_sample())

                if terminal:
                    train_scores.append(self.doom_wrapper.get_total_reward())
                    self.doom_wrapper.reset()
                if self.steps % self.frozen_steps == 0:
                    self.network.update_target_network(session)

            train_time = time() - train_start_time

            print("Epoch", self._epoch)
            print("Training steps:", self.steps, ", epsilon:", self.get_current_epsilon())
            self.print_epoch_log("TRAIN", train_scores, self.train_steps_per_epoch, train_time)
            test_start_time = time()
            test_steps = 0
            for _ in trange(self.test_episodes_per_epoch, leave=False, desc="Testing, epoch {}".format(self._epoch)):
                self.doom_wrapper.reset()
                while not self.doom_wrapper.is_terminal():
                    test_steps += 1
                    state = self.doom_wrapper.get_current_state()
                    a = self.network.get_action(session, state)
                    self.doom_wrapper.make_action(a)

                test_scores.append(self.doom_wrapper.get_total_reward())

            test_time = time() - test_start_time
            overall_time = time() - overall_start_time

            self.print_epoch_log("TEST", test_scores, test_steps, test_time)
            print("Total elapsed time:", sec_to_str(overall_time))
            print()
            self._epoch += 1
