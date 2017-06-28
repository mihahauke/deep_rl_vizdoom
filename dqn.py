# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from time import strftime

from vizdoom_wrapper import VizdoomWrapper
from tqdm import trange
from random import random, randint
import os
from replay_memory import ReplayMemory
from time import time
from util.coloring import red, green, blue
from util import sec_to_str
from util.logger import log
from util.misc import setup_vector_summaries
import sys
import networks


class DQN(object):
    def __init__(self,
                 scenario_tag,
                 network_type="networks.DQNNet",
                 write_summaries=True,
                 tf_logdir="tensorboard_logs",
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
                 enable_progress_bar=True,
                 save_interval=1,
                 writer_max_queue=10,
                 writer_flush_secs=120,
                 **settings):

        if prioritized_memory:
            raise NotImplementedError("Prioritized memory not implemented. Maybe some day.")
            # TODO maybe some day ...
            pass

        self.update_pattern = update_pattern
        self.write_summaries = write_summaries
        self._settings = settings
        date_string = strftime(settings["tag_date_format"])
        network_name = network_type.split(".")[-1]
        self._run_string = "{}/{}/{}".format(scenario_tag, network_name, date_string)
        self.train_steps_per_epoch = train_steps_per_epoch
        self._run_tests = test_episodes_per_epoch > 0 and run_tests
        self.test_episodes_per_epoch = test_episodes_per_epoch
        self._epochs = np.float32(epochs)

        self.doom_wrapper = VizdoomWrapper(**settings)
        misc_len = self.doom_wrapper.misc_len
        img_shape = self.doom_wrapper.img_shape
        self.use_misc = self.doom_wrapper.use_misc
        self.actions_num = self.doom_wrapper.actions_num
        self.replay_memory = ReplayMemory(img_shape, misc_len, batch_size=batchsize, capacity=memory_capacity)
        self.network = eval(network_type)(actions_num=self.actions_num, img_shape=img_shape,
                                          misc_len=misc_len,
                                          **settings)

        self.batchsize = batchsize
        self.frozen_steps = frozen_steps

        self.save_interval = save_interval

        self._model_savefile = settings["models_path"] + "/" + self._run_string
        if self.write_summaries:
            self.scores_placeholder, summaries = setup_vector_summaries(scenario_tag + "/scores")
            self._summaries = tf.summary.merge(summaries)
            self._train_writer = tf.summary.FileWriter("{}/{}/{}".format(tf_logdir, self._run_string, "train"),
                                                           flush_secs=writer_flush_secs,max_queue=writer_max_queue)
            self._test_writer = tf.summary.FileWriter("{}/{}/{}".format(tf_logdir, self._run_string, "test"),
                                                           flush_secs=writer_flush_secs,max_queue=writer_max_queue)
        else:
            self._train_writer = None
            self._test_writer = None
            self._summaries = None
        self.steps = 0
        # TODO epoch as tf variable?
        self._epoch = 1

        # Epsilon
        self.epsilon_decay_rate = (initial_epsilon - final_epsilon) / epsilon_decay_steps
        self.epsilon_decay_start_step = epsilon_decay_start_step
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon

        self.enable_progress_bar = enable_progress_bar

    def get_current_epsilon(self):
        eps = self.initial_epsilon - (self.steps - self.epsilon_decay_start_step) * self.epsilon_decay_rate
        return np.clip(eps, self.final_epsilon, 1.0)

    @staticmethod
    def print_epoch_log(prefix, scores, steps, epoch_time):
        mean_score = np.mean(scores)
        score_std = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        episodes = len(scores)

        steps_per_sec = steps / epoch_time
        mil_steps_per_hour = steps_per_sec * 3600 / 1000000.0
        log(
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

    def save_model(self, session, savefile=None):
        if savefile is None:
            savefile = self._model_savefile
        savedir = os.path.dirname(savefile)
        if not os.path.exists(savedir):
            log("Creating directory: {}".format(savedir))
            os.makedirs(savedir)
        log("Saving model to: {}".format(savefile))
        saver = tf.train.Saver()
        saver.save(session, savefile)

    def load_model(self, session, savefile):
        saver = tf.train.Saver()
        log("Loading model from: {}".format(savefile))
        saver.restore(session, savefile)
        log("Loaded model.")

    def train(self, session):

        # Prefill replay memory:
        for _ in trange(self.replay_memory.capacity, desc="Filling replay memory",
                        leave=False, disable=not self.enable_progress_bar, file=sys.stdout):
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

        log(green("Starting training.\n"))
        while self._epoch <= self._epochs:
            self.doom_wrapper.reset()
            train_scores = []
            test_scores = []
            train_start_time = time()

            for _ in trange(self.train_steps_per_epoch, desc="Training, epoch {}".format(self._epoch),
                            leave=False, disable=not self.enable_progress_bar, file=sys.stdout):
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

            log("Epoch {}".format(self._epoch))
            log("Training steps: {}, epsilon: {}".format(self.steps, self.get_current_epsilon()))
            self.print_epoch_log("TRAIN", train_scores, self.train_steps_per_epoch, train_time)
            test_start_time = time()
            test_steps = 0
            # TESTING
            for _ in trange(self.test_episodes_per_epoch, desc="Testing, epoch {}".format(self._epoch),
                            leave=False, disable=not self.enable_progress_bar, file=sys.stdout):
                self.doom_wrapper.reset()
                while not self.doom_wrapper.is_terminal():
                    test_steps += 1
                    state = self.doom_wrapper.get_current_state()
                    a = self.network.get_action(session, state)
                    self.doom_wrapper.make_action(a)

                test_scores.append(self.doom_wrapper.get_total_reward())

            test_time = time() - test_start_time

            self.print_epoch_log("TEST", test_scores, test_steps, test_time)

            if self.write_summaries:
                log("Writing summaries.")
                train_summary = session.run(self._summaries, {self.scores_placeholder: train_scores})
                self._train_writer.add_summary(train_summary, self.steps)
                if self._run_tests:
                    test_summary = session.run(self._summaries, {self.scores_placeholder: test_scores})
                    self._test_writer.add_summary(test_summary, self.steps)

            # Save model
            if self._epoch % self.save_interval == 0:
                savedir = os.path.dirname(self._model_savefile)
                if not os.path.exists(savedir):
                    log("Creating directory: {}".format(savedir))
                    os.makedirs(savedir)
                log("Saving model to: {}".format(self._model_savefile))
                saver = tf.train.Saver()
                saver.save(session, self._model_savefile)

            overall_time = time() - overall_start_time
            log("Total elapsed time: {}\n".format(sec_to_str(overall_time)))
            self._epoch += 1

    def run_test_episode(self, session):
        self.doom_wrapper.reset()
        while not self.doom_wrapper.is_terminal():
            state = self.doom_wrapper.get_current_state()
            a = self.network.get_action(session, state)
            self.doom_wrapper.make_action(a)
        reward = self.doom_wrapper.get_total_reward()
        return reward
