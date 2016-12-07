# -*- coding: utf-8 -*-
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


class A3CLearner(Thread):
    def __init__(self,
                 thread_index,
                 global_network,
                 optimizer,
                 write_summaries=True,
                 **settings):
        super(A3CLearner, self).__init__()

        print("Creating actor-learner #{}.".format(thread_index))
        self.thread_index = thread_index
        self.write_summaries = write_summaries
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
        misc_len = self.doom_wrapper.misc_len
        img_shape = self.doom_wrapper.img_shape
        self.use_misc = self.doom_wrapper.use_misc

        self.actions_num = self.doom_wrapper.actions_num

        self.local_network = create_network(actions_num=self.actions_num, img_shape=img_shape, misc_len=misc_len,
                                            thread=thread_index, **settings)

        # TODO check gate_gradients != Optimizer.GATE_OP
        grads_and_vars = optimizer.compute_gradients(self.local_network.ops.loss,
                                                     var_list=self.local_network.get_params())
        grads, local_vars = zip(*grads_and_vars)
        grads_and_global_vars = zip(grads, global_network.get_params())

        self.train_op = optimizer.apply_gradients(grads_and_global_vars, global_step=tf.train.get_global_step())
        self.local_network.prepare_sync_op(global_network)

        self.local_steps = 0
        # TODO epoch as tf variable?
        self._epoch = 1
        self.train_scores = []

        self._train_writer = None
        self._test_writer = None
        self._summaries = None

        if self.thread_index == 0:
            # TODO get std, min and max - use lists instead od scalars
            self._model_savefile = settings["models_path"] + "/" + self._run_string
            if self.write_summaries:
                self.score = tf.placeholder(tf.float32)
                tf.scalar_summary(self._run_string + "/mean_score", self.score)
                self._summaries = tf.merge_all_summaries()
        else:
            self._model_savefile = None

        self._saver = None
        self._session = None
        self._global_steps_counter = None

    @staticmethod
    def choose_action_index(policy, deterministic=False):
        if deterministic:
            return np.argmax(policy)

        r = random.random()
        cummulative_sum = 0.0
        for i, p in enumerate(policy):
            cummulative_sum += p
            if r <= cummulative_sum:
                np.set_printoptions(precision=3)
                return i

        return len(policy) - 1

    def make_training_step(self):
        states_img = []
        states_misc = []
        actions = []
        rewards_reversed = []
        values_reversed = []
        advantages = []
        Rs = []

        terminal_end = False

        self._session.run(self.local_network.ops.sync)

        initial_network_state = None
        # TODO add remember state or sumtin
        if self.local_network.has_state():
            initial_network_state = self.local_network.get_current_network_state()

        terminal = None
        steps_performed = 0
        for _ in range(self.max_remembered_steps):
            steps_performed += 1
            current_img, current_misc = self.doom_wrapper.get_current_state()
            policy, state_value = self.local_network.get_policy_and_value(self._session, [current_img, current_misc])
            action_index = A3CLearner.choose_action_index(policy)
            values_reversed.insert(0, state_value)
            states_img.append(current_img)
            states_misc.append(current_misc)
            actions.append(action_index)
            reward = self.doom_wrapper.make_action(action_index)
            terminal = self.doom_wrapper.is_terminal()

            rewards_reversed.insert(0, reward)
            self.local_steps += 1
            if terminal:
                if self.thread_index == 0:
                    self.train_scores.append(self.doom_wrapper.get_total_reward())
                self.doom_wrapper.reset()
                if self.local_network.has_state():
                    self.local_network.reset_state()
                break

        if terminal:
            R = 0.0
        else:
            R = self.local_network.get_value(self._session, self.doom_wrapper.get_current_state())

        for ri, Vi in zip(rewards_reversed, values_reversed):
            R = ri + self.gamma * R
            advantages.insert(0, R - Vi)
            Rs.insert(0, R)

        # TODO delegate this to the network as train_batch(session, ...)
        train_op_feed_dict = {
            self.local_network.vars.state_img: states_img,
            self.local_network.vars.a: actions,
            self.local_network.vars.advantage: advantages,
            self.local_network.vars.R: Rs
        }
        if self.use_misc:
            train_op_feed_dict[self.local_network.vars.state_misc] = states_misc

        if self.local_network.has_state():
            train_op_feed_dict[self.local_network.vars.initial_network_state] = initial_network_state
            train_op_feed_dict[self.local_network.vars.sequence_length] = [len(actions)]

        self._session.run(self.train_op, feed_dict=train_op_feed_dict)

        return steps_performed

    def test(self, sess):
        test_start_time = time.time()
        test_rewards = []
        for _ in trange(self.test_episodes_per_epoch, leave=False):
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
            "TRAIN: mean: {}, min: {}, max: {}, "
            "GlobalSteps: {}, LocalSpd: {:.0f} STEPS/s GlobalSpd: "
            "{} STEPS/s, {:.2f}M STEPS/hour, total elapsed time: {}".format(
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

                    if self.thread_index == 0:
                        self._print_log(self.train_scores, overall_start_time, last_log_time, local_steps_for_log)
                        mean_train_score = np.mean(self.train_scores)
                        self.train_scores = []

                        test_score = None
                        if self._run_tests:
                            test_score = self.test(self._session)

                        if self.write_summaries:
                            train_summary = self._session.run(self._summaries, {self.score: mean_train_score})
                            self._train_writer.add_summary(train_summary, global_steps)
                            if self._run_tests:
                                test_summary = self._session.run(self._summaries, {self.score: test_score})
                                self._test_writer.add_summary(test_summary, global_steps)

                        last_log_time = time.time()
                        local_steps_for_log = 0
                        print()

        except (SignalException, ViZDoomUnexpectedExitException):
            threadsafe_print(red("Thread #{} aborting(ViZDoom killed).".format(self.thread_index)))

    def run_training(self, session, global_steps_counter):
        if self.thread_index == 0:
            # TODO make including sesion.graph optional
            logdir = self._settings["logdir"]
            if self._settings["run_tag"] is not None:
                logdir += "/" + str(self._settings["run_tag"])

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
