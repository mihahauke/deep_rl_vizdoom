#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from util.parsers import parse_train_dqn_args
import os


def train_a3c(settings):
    import tensorflow as tf
    from actor_learner import ActorLearner
    from vizdoom_wrapper import VizdoomWrapper
    from util import ThreadsafeCounter
    from util.optimizers import DQNRMSPropOptimizer

    actions_num = VizdoomWrapper(noinit=True, **settings).actions_num

    # This global step counts gradient applications not performed actions.
    with tf.name_scope("global"):
        with tf.device(settings["device"]):
            global_train_step = tf.Variable(0, trainable=False, name="GlobalStep")
            global_learning_rate = tf.train.polynomial_decay(
                learning_rate=settings["initial_learning_rate"],
                end_learning_rate=settings["final_learning_rate"],
                decay_steps=settings["learning_rate_decay_steps"],
                global_step=global_train_step,
                name="LearningRateDecay")
            optimizer = DQNRMSPropOptimizer(learning_rate=global_learning_rate, **settings["rmsprop"])




if __name__ == "__main__":
    # TODO make tqdm work when stderr is redirected
    # TODO print setup info on stderr and stdout
    args = parse_train_dqn_args()
    # TODO override settings according to args

    default_settings_filepath = "settings/dqn/defaults.json"
    override_settings_filepath = args.settings_json
    dqn_settings = json.load(open(default_settings_filepath))
    override_settings = json.load(open(override_settings_filepath))
    dqn_settings.update(override_settings)

    if not os.path.isdir(dqn_settings["models_path"]):
        os.makedirs(dqn_settings["models_path"])
    if not os.path.isdir(dqn_settings["logdir"]):
        os.makedirs(dqn_settings["logdir"])

    train_a3c(dqn_settings)

