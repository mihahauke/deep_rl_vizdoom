# -*- coding: utf-8 -*-

import logging
import sys
import time
from threading import Thread

import numpy as np
import tensorflow as tf
from tqdm import trange
from vizdoom import SignalException, ViZDoomUnexpectedExitException

import networks
import random
from util import sec_to_str, threadsafe_print, ensure_parent_directories, create_directory
from util.coloring import red, green, blue, yellow
from util.logger import log
from util.misc import setup_vector_summaries
from vizdoom_wrapper import VizdoomWrapper


class A3CLearner(Thread):
    def __init__(self,
                 thread_index=0,
                 model_savefile=None,
                 network_class="ACLstmNet",
                 global_steps_counter=None,
                 scenario_tag=None,
                 run_id_string=None,
                 session=None,
                 tf_logdir=None,
                 global_network=None,
                 optimizer=None,
                 learning_rate=None,
                 test_only=False,
                 test_interval=1,
                 write_summaries=True,
                 enable_progress_bar=True,
                 deterministic_testing=True,
                 save_interval=1,
                 writer_max_queue=10,
                 writer_flush_secs=120,
                 gamma_compensation=False,
                 figar_gamma=False,
                 gamma=0.99,
                 **settings):
        super(A3CLearner, self).__init__()

        log("Creating actor-learner #{}.".format(thread_index))
        self.thread_index = thread_index

        self._global_steps_counter = global_steps_counter
        self.write_summaries = write_summaries
        self.save_interval = save_interval
        self.enable_progress_bar = enable_progress_bar
        self._model_savefile = None
        self._train_writer = None
        self._test_writer = None
        self._summaries = None
        self._session = session
        self.deterministic_testing = deterministic_testing
        self.local_steps = 0
        # TODO epoch as tf variable?
        self._epoch = 1
        self.train_scores = []
        self.train_actions = []
        self.train_frameskips = []

        self.test_interval = test_interval

        self.local_steps_per_epoch = settings["local_steps_per_epoch"]
        self._run_tests = settings["test_episodes_per_epoch"] > 0 and settings["run_tests"]
        self.test_episodes_per_epoch = settings["test_episodes_per_epoch"]
        self._epochs = np.float32(settings["epochs"])
        self.max_remembered_steps = settings["max_remembered_steps"]

        assert not (gamma_compensation and figar_gamma)

        gamma = np.float32(gamma)

        if gamma_compensation:
            self.scale_gamma = lambda fskip: ((1-gamma**fskip)/(1-gamma), gamma ** fskip)
        elif figar_gamma:
            self.scale_gamma = lambda fskip: (1.0, gamma ** fskip)
        else:
            self.scale_gamma = lambda _: (1.0, gamma)

        if self.write_summaries and thread_index == 0 and not test_only:
            assert tf_logdir is not None
            self.run_id_string = run_id_string
            self.tf_models_path = settings["models_path"]
            create_directory(tf_logdir)

            if self.tf_models_path is not None:
                create_directory(self.tf_models_path)

        self.doom_wrapper = VizdoomWrapper(**settings)
        misc_len = self.doom_wrapper.misc_len
        img_shape = self.doom_wrapper.img_shape
        self.use_misc = self.doom_wrapper.use_misc

        self.actions_num = self.doom_wrapper.actions_num
        self.local_network = getattr(networks, network_class)(actions_num=self.actions_num, img_shape=img_shape,
                                                              misc_len=misc_len,
                                                              thread=thread_index, **settings)

        if not test_only:
            self.learning_rate = learning_rate
            # TODO check gate_gradients != Optimizer.GATE_OP
            grads_and_vars = optimizer.compute_gradients(self.local_network.ops.loss,
                                                         var_list=self.local_network.get_params())
            grads, local_vars = zip(*grads_and_vars)

            grads_and_global_vars = zip(grads, global_network.get_params())
            self.train_op = optimizer.apply_gradients(grads_and_global_vars, global_step=tf.train.get_global_step())

            self.global_network = global_network
            self.local_network.prepare_sync_op(global_network)

        if self.thread_index == 0 and not test_only:
            self._model_savefile = model_savefile
            if self.write_summaries:
                self.actions_placeholder = tf.placeholder(tf.int32, None)
                self.frameskips_placeholder = tf.placeholder(tf.int32, None)
                self.scores_placeholder, summaries = setup_vector_summaries(scenario_tag + "/scores")

                # TODO remove scenario_tag from histograms
                a_histogram = tf.summary.histogram(scenario_tag + "/actions", self.actions_placeholder)
                fs_histogram = tf.summary.histogram(scenario_tag + "/frameskips", self.frameskips_placeholder)
                score_histogram = tf.summary.histogram(scenario_tag + "/scores", self.scores_placeholder)
                lr_summary = tf.summary.scalar(scenario_tag + "/learning_rate", self.learning_rate)
                summaries.append(lr_summary)
                summaries.append(a_histogram)
                summaries.append(fs_histogram)
                summaries.append(score_histogram)
                self._summaries = tf.summary.merge(summaries)
                self._train_writer = tf.summary.FileWriter("{}/{}/{}".format(tf_logdir, self.run_id_string, "train"),
                                                           flush_secs=writer_flush_secs, max_queue=writer_max_queue)
                self._test_writer = tf.summary.FileWriter("{}/{}/{}".format(tf_logdir, self.run_id_string, "test"),
                                                          flush_secs=writer_flush_secs, max_queue=writer_max_queue)

    @staticmethod
    def choose_best_index(policy, deterministic=True):
        if deterministic:
            return np.argmax(policy)

        r = random.random()
        cummulative_sum = 0.0
        for i, p in enumerate(policy):
            cummulative_sum += p
            if r <= cummulative_sum:
                return i

        return len(policy) - 1

    def make_training_step(self):
        states_img = []
        states_misc = []
        actions = []
        rewards_reversed = []
        Rs = []

        self._session.run(self.local_network.ops.sync)

        initial_network_state = None
        if self.local_network.has_state():
            initial_network_state = self.local_network.get_current_network_state()

        terminal = None
        steps_performed = 0
        for _ in range(self.max_remembered_steps):
            steps_performed += 1
            current_state = self.doom_wrapper.get_current_state()
            policy = self.local_network.get_policy(self._session, current_state)
            action_index = A3CLearner.choose_best_index(policy, deterministic=False)
            states_img.append(current_state[0])
            states_misc.append(current_state[1])
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

        self.train_actions += actions
        self.train_frameskips += [self.doom_wrapper.frameskip] * len(actions)

        if terminal:
            R = 0.0
        else:
            R = self.local_network.get_value(self._session, self.doom_wrapper.get_current_state())

        # #TODO this could be handles smarter ....
        for ri in rewards_reversed:
            scale, gamma = self.scale_gamma(self.doom_wrapper.frameskip)
            R = scale * ri + gamma * R
            Rs.insert(0, R)

        train_op_feed_dict = {
            self.local_network.vars.state_img: states_img,
            self.local_network.vars.a: actions,
            self.local_network.vars.R: Rs
        }
        if self.use_misc:
            train_op_feed_dict[self.local_network.vars.state_misc] = states_misc

        if self.local_network.has_state():
            train_op_feed_dict[self.local_network.vars.initial_network_state] = initial_network_state
            train_op_feed_dict[self.local_network.vars.sequence_length] = [len(actions)]

        self._session.run(self.train_op, feed_dict=train_op_feed_dict)

        return steps_performed

    def run_episode(self, deterministic=True, return_stats=False):
        self.doom_wrapper.reset()
        if self.local_network.has_state():
            self.local_network.reset_state()
        actions = []
        frameskips = []
        rewards = []
        while not self.doom_wrapper.is_terminal():
            current_state = self.doom_wrapper.get_current_state()
            action_index, frameskip = self._get_best_action(self._session, current_state, deterministic=deterministic)
            reward = self.doom_wrapper.make_action(action_index, frameskip)
            if return_stats:
                actions.append(action_index)
                if frameskip is None:
                    frameskip = self.doom_wrapper.frameskip
                frameskips.append(frameskip)
                rewards.append(reward)

        total_reward = self.doom_wrapper.get_total_reward()
        if return_stats:
            return total_reward, actions, frameskips, rewards
        else:
            return total_reward

    def test(self, episodes_num=None, deterministic=True):
        if episodes_num is None:
            episodes_num = self.test_episodes_per_epoch

        test_start_time = time.time()
        test_rewards = []
        test_actions = []
        test_frameskips = []
        for _ in trange(episodes_num, desc="Testing", file=sys.stdout,
                        leave=False, disable=not self.enable_progress_bar):
            total_reward, actions, frameskips, _ = self.run_episode(deterministic=deterministic, return_stats=True)
            test_rewards.append(total_reward)
            test_actions += actions
            test_frameskips += frameskips

        self.doom_wrapper.reset()
        if self.local_network.has_state():
            self.local_network.reset_state()

        test_end_time = time.time()
        test_duration = test_end_time - test_start_time
        min_score = np.min(test_rewards)
        max_score = np.max(test_rewards)
        mean_score = np.mean(test_rewards)
        score_std = np.std(test_rewards)
        log(
            "TEST: mean: {}, min: {}, max: {}, test time: {}".format(
                green("{:0.3f}±{:0.2f}".format(mean_score, score_std)),
                red("{:0.3f}".format(min_score)),
                blue("{:0.3f}".format(max_score)),
                sec_to_str(test_duration)))
        return test_rewards, test_actions, test_frameskips

    def _print_train_log(self, scores, overall_start_time, last_log_time, steps):
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
        log(
            "TRAIN: {}(GlobalSteps), {} episodes, mean: {}, min: {}, max: {}, "
            "\nLocalSpd: {:.0f} STEPS/s GlobalSpd: "
            "{} STEPS/s, {:.2f}M STEPS/hour, total elapsed time: {}".format(
                global_steps,
                len(scores),
                green("{:0.3f}±{:0.2f}".format(mean_score, score_std)),
                red("{:0.3f}".format(min_score)),
                blue("{:0.3f}".format(max_score)),
                local_steps_per_sec,
                blue("{:.0f}".format(
                    global_steps_per_sec)),
                global_mil_steps_per_hour,
                sec_to_str(elapsed_time)
            ))

    def run(self):
        # TODO this method is ugly, make it nicer
        try:
            overall_start_time = time.time()
            last_log_time = overall_start_time
            local_steps_for_log = 0
            while self._epoch <= self._epochs:
                steps = self.make_training_step()
                local_steps_for_log += steps
                global_steps = self._global_steps_counter.inc(steps)
                # Logs & tests
                if self.local_steps_per_epoch * self._epoch <= self.local_steps:
                    self._epoch += 1

                    if self.thread_index == 0:
                        self._print_train_log(self.train_scores, overall_start_time, last_log_time, local_steps_for_log)

                        reun_test_this_epoch = (self._epoch % self.test_interval) == 0
                        if self._run_tests and reun_test_this_epoch:
                            test_scores, test_actions, test_frameskips = self.test(
                                deterministic=self.deterministic_testing)

                        if self.write_summaries:
                            train_summary = self._session.run(self._summaries,
                                                              {self.scores_placeholder: self.train_scores,
                                                               self.actions_placeholder: self.train_actions,
                                                               self.frameskips_placeholder: self.train_frameskips})
                            self._train_writer.add_summary(train_summary, global_steps)
                            if self._run_tests and reun_test_this_epoch:
                                test_summary = self._session.run(self._summaries,
                                                                 {self.scores_placeholder: test_scores,
                                                                  self.actions_placeholder: test_actions,
                                                                  self.frameskips_placeholder: test_frameskips})
                                self._test_writer.add_summary(test_summary, global_steps)

                        last_log_time = time.time()
                        local_steps_for_log = 0
                        log("Learning rate: {}".format(self._session.run(self.learning_rate)))

                        # Saves model
                        if self._epoch % self.save_interval == 0:
                            self.save_model()
                        log("")
                    self.train_scores = []
                    self.train_actions = []
                    self.train_frameskips = []

        except (SignalException, ViZDoomUnexpectedExitException):
            threadsafe_print(red("Thread #{} aborting(ViZDoom killed).".format(self.thread_index)))

    def run_training(self, session):
        self._session = session
        self.start()

    def save_model(self):
        ensure_parent_directories(self._model_savefile)
        log("Saving model to: {}".format(self._model_savefile))
        saver = tf.train.Saver(self.local_network.get_params())
        saver.save(self._session, self._model_savefile)

    def load_model(self, session, savefile):
        saver = tf.train.Saver(self.local_network.get_params())
        log("Loading model from: {}".format(savefile))
        saver.restore(session, savefile)
        log("Loaded model.")

    def _get_best_action(self, sess, state, deterministic=True):
        policy = self.local_network.get_policy(sess, state)
        action_index = self.choose_best_index(policy, deterministic=deterministic)
        frameskip = None
        return action_index, frameskip


class ADQNLearner(A3CLearner):
    def __init__(self,
                 network_class="ADQNLstmNet",
                 global_target_network=None,
                 unfreeze_thread=False,
                 frozen_global_steps=40000,
                 initial_epsilon=1.0,
                 final_epsilon=0.1,
                 epsilon_decay_steps=10e06,
                 epsilon_decay_start_step=0,
                 **args):
        super(ADQNLearner, self).__init__(network_class=network_class, **args)
        self.global_target_network = global_target_network
        self.unfreeze_thread = unfreeze_thread
        if unfreeze_thread:
            self.frozen_global_steps = frozen_global_steps
        else:
            self.frozen_global_steps = None
            # Epsilon
            # TODO randomize epsilon somehow
        self.epsilon_decay_rate = (initial_epsilon - final_epsilon) / epsilon_decay_steps
        self.epsilon_decay_start_step = epsilon_decay_start_step
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon

    def get_current_epsilon(self):
        eps = self.initial_epsilon - (self.local_steps - self.epsilon_decay_start_step) * self.epsilon_decay_rate
        return np.clip(eps, self.final_epsilon, 1.0)

    def make_training_step(self):
        states_img = []
        states_misc = []
        actions = []
        rewards_reversed = []
        target_qs = []

        self._session.run(self.local_network.ops.sync)
        initial_network_state = None
        if self.local_network.has_state():
            initial_network_state = self.local_network.get_current_network_state()

        terminal = None
        steps_performed = 0
        for _ in range(self.max_remembered_steps):
            steps_performed += 1
            current_img, current_misc = self.doom_wrapper.get_current_state()

            if random.random() <= self.get_current_epsilon():
                action_index = random.randint(0, self.actions_num - 1)
                if self.local_network.has_state():
                    self.local_network.update_network_state(self._session, [current_img, current_misc])
            else:
                q_values = self.local_network.get_q_values(self._session, [current_img, current_misc]).flatten()
                action_index = q_values.argmax()

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
            target_q = 0.0
        else:
            if self.global_network.has_state():
                q2 = self.global_target_network.get_q_values(self._session,
                                                             self.doom_wrapper.get_current_state(),
                                                             False,
                                                             self.local_network.get_current_network_state())
            else:
                q2 = self.global_target_network.get_q_values(self._session,
                                                             self.doom_wrapper.get_current_state())

            target_q = q2.max()

        for ri in rewards_reversed:
            target_q = ri + self.gamma * target_q
            target_qs.insert(0, target_q)

        # TODO delegate this to the network as train_batch(session, ...), maybe?
        train_op_feed_dict = {
            self.local_network.vars.state_img: states_img,
            self.local_network.vars.a: actions,
            self.local_network.vars.target_q: target_qs
        }
        if self.use_misc:
            train_op_feed_dict[self.local_network.vars.state_misc] = states_misc

        if self.local_network.has_state():
            train_op_feed_dict[self.local_network.vars.initial_network_state] = initial_network_state
            train_op_feed_dict[self.local_network.vars.sequence_length] = [len(actions)]

        self._session.run(self.train_op, feed_dict=train_op_feed_dict)
        return steps_performed

    def run(self):
        # TODO this method is ugly, make it nicer ...and it's the same as above.... really TODO!!
        # Basically code copied from base class with unfreezing
        try:
            overall_start_time = time.time()
            last_log_time = overall_start_time
            local_steps_for_log = 0
            next_target_update = self.frozen_global_steps
            while self._epoch <= self._epochs:
                steps = self.make_training_step()
                local_steps_for_log += steps
                global_steps = self._global_steps_counter.inc(steps)

                # Updating target network:
                if self.unfreeze_thread:
                    # TODO this check is dangerous
                    if global_steps >= next_target_update:
                        next_target_update += self.frozen_global_steps
                        if next_target_update <= global_steps:
                            # TODO use warn from the logger
                            logging.warning(yellow("Global steps ({}) <= next target update ({}).".format(
                                global_steps, next_target_update)))

                        self._session.run(self.global_network.ops.unfreeze)
                # Logs & tests
                if self.local_steps_per_epoch * self._epoch <= self.local_steps:
                    self._epoch += 1

                    if self.thread_index == 0:
                        self._print_train_log(self.train_scores, overall_start_time, last_log_time, local_steps_for_log)

                        if self._run_tests:
                            test_scores, actions, frameskips = self.test(deterministic=self.deterministic_testing)

                        if self.write_summaries:
                            train_summary = self._session.run(self._summaries,
                                                              {self.scores_placeholder: self.train_scores})
                            self._train_writer.add_summary(train_summary, global_steps)
                            if self._run_tests:
                                test_summary = self._session.run(self._summaries,
                                                                 {self.scores_placeholder: test_scores})
                                self._test_writer.add_summary(test_summary, global_steps)

                        last_log_time = time.time()
                        local_steps_for_log = 0

                        log("Learning rate: {}".format(self._session.run(self.learning_rate)))

                        # Saves model
                        if self._epoch % self.save_interval == 0:
                            self.save_model()

                        log("")

                    self.train_scores = []
                    self.train_actions = []
                    self.train_frameskips = []

        except (SignalException, ViZDoomUnexpectedExitException):
            threadsafe_print(red("Thread #{} aborting(ViZDoom killed).".format(self.thread_index)))

    def _get_best_action(self, sess, state, deterministic=True):
        q = self.local_network.get_q_values(sess, state).flatten()
        action_index = q.argmax()
        frameskip = None
        return action_index, frameskip


class FigarA3CLearner(A3CLearner):
    def __init__(self,
                 dynamic_frameskips=None,
                 multi_frameskip=False,
                 cfigar=False,
                 **args):
        if dynamic_frameskips is not None:
            self.binomial_frameskip = False
            if cfigar:
                raise ValueError()
            if isinstance(dynamic_frameskips, (list, tuple)):
                self.frameskips = list(dynamic_frameskips)
            elif isinstance(dynamic_frameskips, int):
                self.frameskips = list(range(1, dynamic_frameskips + 1))
            self.frameskips_indices = {f: i for i, f in enumerate(self.frameskips)}
        elif not cfigar:
            raise ValueError()
        else:
            self.binomial_frameskip = True
            self.multi_frameskip = multi_frameskip
        super(FigarA3CLearner, self).__init__(dynamic_frameskips=dynamic_frameskips,
                                              multi_frameskip=multi_frameskip,
                                              **args)

    def make_training_step(self):
        # TODO mostly coppied, just added frameskip, a bit wasteful (maybe merge it with basic learner after all...)
        states_img = []
        states_misc = []
        actions = []
        frameskips = []
        rewards_reversed = []
        Rs = []

        self._session.run(self.local_network.ops.sync)

        initial_network_state = None
        if self.local_network.has_state():
            initial_network_state = self.local_network.get_current_network_state()

        terminal = None
        steps_performed = 0
        # TODO changed compared to base:
        for _ in range(self.max_remembered_steps):
            steps_performed += 1
            current_state = self.doom_wrapper.get_current_state()
            action_index, frameskip = self._get_best_action(self._session, current_state, deterministic=False)
            if self.binomial_frameskip:
                frameskips.append(frameskip)
            else:
                frameskips.append(self.frameskips_indices[frameskip])
            self.train_actions.append(action_index)
            self.train_frameskips.append(frameskip)

            reward = self.doom_wrapper.make_action(action_index, frameskip)
            terminal = self.doom_wrapper.is_terminal()

            rewards_reversed.insert(0, reward)
            states_img.append(current_state[0])
            states_misc.append(current_state[1])
            actions.append(action_index)
            # TODO end (and frameskip in feeddict)
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

        for ri, fs in zip(rewards_reversed, reversed(frameskips)):
            scale, gamma = self.scale_gamma(fs)
            R = scale * ri + gamma * R
            Rs.insert(0, R)

        train_op_feed_dict = {
            self.local_network.vars.state_img: states_img,
            self.local_network.vars.a: actions,
            self.local_network.vars.frameskip: frameskips,
            self.local_network.vars.R: Rs
        }
        if self.use_misc:
            train_op_feed_dict[self.local_network.vars.state_misc] = states_misc

        if self.local_network.has_state():
            train_op_feed_dict[self.local_network.vars.initial_network_state] = initial_network_state
            train_op_feed_dict[self.local_network.vars.sequence_length] = [len(actions)]

        self._session.run(self.train_op, feed_dict=train_op_feed_dict)
        return steps_performed

    @staticmethod
    def choose_best_frameskip_binomial(n, p, deterministic=True):
        # Binomial test:
        if deterministic:

            frameskip = int(n * p) + 1
        else:
            frameskip = int(np.random.binomial(n, p)) + 1
        return frameskip

        # if deterministic:
        #     frameskip_float = mu
        # else:
        #     frameskip_float = np.random.normal(mu, sigma)
        # frameskip = int(max(1, round(frameskip_float)))
        # return frameskip

    def _get_best_action(self, sess, state, deterministic=True):
        policy, frameskip_policy = self.local_network.get_policy(sess, state)
        action_index = self.choose_best_index(policy, deterministic=deterministic)
        if self.binomial_frameskip:
            n, p = frameskip_policy
            if self.multi_frameskip:
                n = n[action_index]
                p = p[action_index]
            frameskip = self.choose_best_frameskip_binomial(n, p, deterministic=deterministic)
        else:
            frameskip_index = self.choose_best_index(frameskip_policy, deterministic=deterministic)
            frameskip = self.frameskips[frameskip_index]
        return action_index, frameskip
