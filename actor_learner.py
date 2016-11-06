# -*- coding: utf-8 -*-
from __future__ import print_function
from vizdoom import SignalException, ViZDoomUnexpectedExitException
from util import sec_to_str, threadsafe_print
import numpy as np
import random
import time
from vizdoom_wrapper import VizdoomWrapper
from networks import create_network
from tqdm import trange
from threading import Thread
from util.coloring import red, green, blue
from time import strftime
import tensorflow as tf


class ActorLearner(Thread):
    def __init__(self,
                 thread_index,
                 global_network,
                 # TODO somehow move global train step somewhere else
                 global_train_step,
                 optimizer,
                 **settings):
        super(ActorLearner, self).__init__()

        print("Spawning actor-learner #{}.".format(thread_index))
        self.index = thread_index
        self._settings = settings
        date_string = strftime("%d.%m.%y-%H:%M")
        self._run_string = "{}/{}_{}/{}".format(settings["base_tag"], settings["network_type"], settings["threads_num"],
                                                date_string)

        self.steps_per_epoch = settings["local_steps_per_epoch"]
        self._run_tests = settings["test_episodes_per_epoch"] > 0 and settings["run_tests"]
        self.test_episodes_per_epoch = settings["test_episodes_per_epoch"]
        self._epochs = np.float32(settings["epochs"])
        self.max_remembered_steps = settings["max_remembered_steps"]
        self.gamma = np.float32(settings["gamma"])

        self.doom_wrapper = VizdoomWrapper(**settings)
        self.actions_num = self.doom_wrapper.actions_num

        self.local_network = create_network(actions_num=self.actions_num, thread=thread_index, **settings)

        # TODO check gate_gradients != Optimizer.GATE_OP
        grads_and_vars = optimizer.compute_gradients(self.local_network.ops.loss,
                                                     var_list=self.local_network.get_params())
        grads, local_vars = zip(*grads_and_vars)
        grads_and_global_vars = zip(grads, global_network.get_params())

        self.train_op = optimizer.apply_gradients(grads_and_global_vars, global_step=global_train_step)
        self.sync_op = self.local_network.ops.sync(global_network)

        self.local_steps = 0
        self._epoch = 1
        self.train_scores = []

        self._train_writer = None
        self._test_writer = None
        if self.index == 0:
            self._model_savefile = settings["models_path"] + "/" + self._run_string
            self.test_score = tf.placeholder(tf.float32)
            self.score_summary = tf.placeholder(tf.float32)
            tf.scalar_summary(self._run_string + "/score", self.score_summary)
            self._summaries = tf.merge_all_summaries()
        else:
            self._model_savefile = None

            self._summaries = None

        self._saver = None
        self._session = None
        self._global_steps_counter = None

    @staticmethod
    def choose_action_index(policy, deterministic=False):
        if deterministic:
            return np.argmax(policy)

        # TODO the sum should be 1 so ...
        r = random.random() * np.sum(policy)
        cummulative_sum = 0.0
        for i, p in enumerate(policy):
            cummulative_sum += p
            if r <= cummulative_sum:
                return i

        return len(policy) - 1

    def make_training_step(self):
        states = []
        actions = []
        rewards = []
        values = []

        terminal_end = False

        self._session.run(self.sync_op)

        if self.local_network.has_state():
            initial_network_state = self.local_network.get_current_network_state()

        start_local_steps = self.local_steps

        iteration = 0
        while True:
            if iteration >= self.max_remembered_steps:
                break
            iteration += 1
            current_state = self.doom_wrapper.get_current_state()
            policy, state_value = self.local_network.get_policy_and_value(self._session, current_state)
            action_index = ActorLearner.choose_action_index(policy)
            states.insert(0, current_state)
            actions.insert(0, action_index)
            values.insert(0, state_value)

            reward = self.doom_wrapper.make_action(action_index)
            terminal = self.doom_wrapper.is_terminal()

            rewards.insert(0, reward)
            self.local_steps += 1

            if terminal:
                terminal_end = True
                if self.index == 0:
                    self.train_scores.append(self.doom_wrapper.get_total_reward())
                self.doom_wrapper.reset()
                if self.local_network.has_state():
                    self.local_network.reset_state()
                break

        R = 0.0
        if not terminal_end:
            R = self.local_network.get_value(self._session, self.doom_wrapper.get_current_state())

        batch_s = []
        batch_a = []
        batch_td = []
        batch_R = []

        # TODO try to make it more efficient
        for (ai, ri, s, Vi) in zip(actions, rewards, states, values):
            R = ri + self.gamma * R
            td = R - Vi
            a = np.zeros([self.actions_num])
            a[ai] = 1

            batch_s.insert(0, s)
            batch_a.insert(0, a)
            batch_td.insert(0, td)
            batch_R.insert(0, R)

        train_op_feed_dict = {
            self.local_network.vars.state: batch_s,
            self.local_network.vars.a: batch_a,
            self.local_network.vars.td: batch_td,
            self.local_network.vars.r: batch_R
        }

        if self.local_network.has_state():
            train_op_feed_dict[self.local_network.vars.initial_network_state] = initial_network_state
            train_op_feed_dict[self.local_network.vars.sequence_length] = [len(batch_a)]

        self._session.run(self.train_op, feed_dict=train_op_feed_dict)
        steps_performed = self.local_steps - start_local_steps

        return steps_performed

    def test(self, sess, episodes):
        test_start_time = time.time()
        test_rewards = []
        for _ in trange(self.test_episodes_per_epoch):
            self.doom_wrapper.reset()
            if self.local_network.has_state():
                self.local_network.reset_state()
            while not self.doom_wrapper.is_terminal():
                current_state = self.doom_wrapper.get_current_state()
                policy = self.local_network.get_policy(sess, current_state)
                action_index = self.choose_action_index(policy, deterministic=True)
                self.doom_wrapper.make_action(action_index)

            total_reward = self.doom_wrapper.get_total_reward()
            test_rewards.append(total_reward)

        self.doom_wrapper.reset()
        if self.local_network.has_state():
            self.local_network.reset_state()

        test_end_time = time.time()
        test_duration = test_end_time - test_start_time
        min_score = np.min(test_rewards)
        max_score = np.max(test_rewards)
        mean_score = np.mean(test_rewards)
        score_std = np.std(test_rewards)
        print(
            "TEST: mean: {}, min: {}, max: {} ,test time: {}".format(
                green("{:0.3f}±{:0.2f}".format(mean_score, score_std)),
                red("{:0.3f}".format(min_score)),
                blue("{:0.3f}".format(max_score)),
                sec_to_str(test_duration)))
        return mean_score

    def _print_log(self, scores, overall_start_time, last_log_time, steps):
        current_time = time.time()
        mean_score = np.mean(scores)
        score_std = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)

        elapsed_time = time.time() - overall_start_time
        global_steps = self._global_steps_counter.get()
        local_steps_per_sec = steps / (current_time - last_log_time)
        global_steps_per_sec = global_steps / elapsed_time
        global_mil_steps_per_hour = global_steps_per_sec * 3600 / 1000000.0
        print(
            "TRAIN: mean: {}, min: {}, max:{}, "
            "GlobalSteps: {}, LocalSpd: {:.0f} STEPS/s GlobalSpd: "
            "{} STEPS/s, {:.2f}M STEPS/hour, overall time: {}".format(
                green("{:0.3f}±{:0.2f}".format(mean_score, score_std)),
                red("{:0.3f}".format(min_score)),
                blue("{:0.3f}".format(max_score)),
                global_steps,
                local_steps_per_sec,
                blue("{:.0f}".format(
                    global_steps_per_sec)),
                global_mil_steps_per_hour,
                sec_to_str(elapsed_time)
            ))

    def run(self):
        try:
            overall_start_time = time.time()
            last_log_time = overall_start_time
            local_steps_for_log = 0
            while self._epoch <= self._epochs:
                steps = self.make_training_step()
                local_steps_for_log += steps
                global_steps = self._global_steps_counter.inc(steps)

                # Logs & tests
                if self.steps_per_epoch * self._epoch <= self.local_steps:
                    self._epoch += 1

                    if self.index == 0:
                        self._print_log(self.train_scores, overall_start_time, last_log_time, local_steps_for_log)
                        mean_train_scre = np.mean(self.train_scores)
                        self.train_scores = []

                        train_summary = self._session.run(self._summaries, {self.score_summary: mean_train_scre})
                        self._train_writer.add_summary(train_summary, global_steps)
                        if self._run_tests:
                            test_score = self.test(self._session, self.test_episodes_per_epoch)
                            test_summary = self._session.run(self._summaries, {self.score_summary: test_score})
                            self._test_writer.add_summary(test_summary, global_steps)

                    last_log_time = time.time()
                    local_steps_for_log = 0

        except (SignalException, ViZDoomUnexpectedExitException):
            threadsafe_print(red("Thread #{} aborting(ViZDoom killed).".format(self.index)))

    def run_training(self, session, global_steps_counter):
        if self.index == 0:
            # TODO make including sesion.graph optional
            logdir = self._settings["logdir"]
            if self._settings["run"] is not None:
                logdir += "/" + str(self._settings["run"])

            self._train_writer = tf.train.SummaryWriter(logdir + "/train")
            if self._run_tests:
                self._test_writer = tf.train.SummaryWriter(logdir + "/test")
            else:
                self._test_writer = None
            # TODO create saver
            self._saver = None
        self._session = session
        self._global_steps_counter = global_steps_counter
        self.start()
