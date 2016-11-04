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
from termcolor import colored


class ActorLearner(Thread):
    def __init__(self,
                 thread_index,
                 global_network,
                 # TODO somehow move global train step somewhere else
                 global_train_step,
                 optimizer,
                 silent=False,
                 **settings):
        super(ActorLearner, self).__init__()

        self._session = None
        self._global_steps_counter = None

        self.index = thread_index
        self._silent = silent
        if not self._silent:
            print("Spawning actor-learner #{}.".format(thread_index))

        self.steps_per_epoch = settings["steps_per_epoch"]
        self._print_logs = not silent and (settings["all_threads_log"] or self.index == 0)
        self._run_tests = settings["test_episodes_per_epoch"] > 0 and (
            settings["all_threads_test"] or self.index == 0) and settings["run_tests"]
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

        self.sync = self.local_network.ops.sync(global_network)

        self.local_steps = 0
        self._epoch = 1

        self.train_scores = []

    @staticmethod
    def choose_action_index(policy, deterministic=True):
        if deterministic:
            return np.argmax(policy)

        r = random.random() * np.sum(policy)
        cummulative_sum = 0.0
        for i, p in enumerate(policy):
            cummulative_sum += p
            if r <= cummulative_sum:
                return i

        return len(policy) - 1

    def train_step(self):
        states = []
        actions = []
        rewards = []
        values = []

        terminal_end = False

        self._session.run(self.sync)

        if self.local_network.has_state():
            initial_network_state = self.local_network.get_current_network_state()

        start_local_t = self.local_steps

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
                if self._print_logs:
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
        steps_performed = self.local_steps - start_local_t

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
        score_mean = np.mean(test_rewards)
        score_std = np.std(test_rewards)
        if not self._silent:
            print(
                "th#{} Test: {:0.2f}±{:0.2f}, test time: {}".format(self.index, score_mean, score_std,
                                                                    sec_to_str(test_duration)))
        return

    def _print_log(self, overall_start_time, last_log_time, steps):
        current_time = time.time()
        mean_score = np.mean(self.train_scores)
        self.train_scores = []

        elapsed_time = time.time() - overall_start_time
        global_steps = self._global_steps_counter.get()
        local_steps_per_sec = steps / (current_time - last_log_time)
        global_steps_per_sec = global_steps / elapsed_time
        global_mil_steps_per_hour = global_steps_per_sec * 3600 / 1000000.0
        # TODO compute local speed without tests
        print(
            "th#{} {}:, Score: {:.2f}, GlobalSteps: {}, LocalSpd: {:.0f} STEPS/s GlobalSpd: {:.0f} STEPS/s,"
            " {:.2f}M STEPS/hour, overall time: {}".format(
                self.index, self.local_steps, mean_score, global_steps, local_steps_per_sec,
                global_steps_per_sec,
                global_mil_steps_per_hour,
                sec_to_str(elapsed_time)
            ))

    def run(self):
        try:
            overall_start_time = time.time()
            last_log_time = overall_start_time
            local_steps_for_log = 0
            while self._epoch <= self._epochs:
                steps = self.train_step()
                self._global_steps_counter.inc(steps)
                local_steps_for_log += steps

                # Logs & tests
                if self.steps_per_epoch * self._epoch <= self.local_steps:
                    self._epoch += 1

                    if self._print_logs:
                        self._print_log(overall_start_time, last_log_time, local_steps_for_log)

                    if self._run_tests:
                        self.test(self._session, self.test_episodes_per_epoch)

                    last_log_time = time.time()
                    local_steps_for_log = 0

        except (SignalException, ViZDoomUnexpectedExitException):
            threadsafe_print(colored("Thread #{} aborting(ViZDoom killed).".format(self.index), "red"))

    def run_training(self, session, global_steps_counter):
        self._session = session
        self._global_steps_counter = global_steps_counter
        self.start()
