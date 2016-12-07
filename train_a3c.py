#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ruamel.yaml as yaml
from util.parsers import parse_train_a3c_args
from util.coloring import green

import os


def train_a3c(settings):
    import tensorflow as tf
    from actor_learner import ActorLearner
    from vizdoom_wrapper import VizdoomWrapper
    from networks import create_ac_network
    from util import ThreadsafeCounter
    from util.optimizers import ClippingRMSPropOptimizer

    tmpVizdoomWrapper = VizdoomWrapper(noinit=True, **settings)
    actions_num = tmpVizdoomWrapper.actions_num
    misc_len = tmpVizdoomWrapper.misc_len
    img_shape = tmpVizdoomWrapper.img_shape
    tmpVizdoomWrapper = None
    global_network = create_ac_network(actions_num=actions_num, misc_len=misc_len, img_shape=img_shape, **settings)

    # This global step counts gradient applications not performed actions.
    global_train_step = tf.Variable(0, trainable=False, name="global_step")
    global_learning_rate = tf.train.polynomial_decay(
        learning_rate=settings["initial_learning_rate"],
        end_learning_rate=settings["final_learning_rate"],
        decay_steps=settings["learning_rate_decay_steps"],
        global_step=global_train_step)
    optimizer = ClippingRMSPropOptimizer(learning_rate=global_learning_rate, **settings["rmsprop"])

    actor_learners = []
    for i in range(settings["threads_num"]):
        actor_learner = ActorLearner(thread_index=i, global_network=global_network, optimizer=optimizer, **settings)
        actor_learners.append(actor_learner)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    print("Initializing variables...")
    session.run(tf.global_variables_initializer())
    print("Initialization finished.")
    global_steps_counter = ThreadsafeCounter()

    print(green("Launching training."))
    for l in actor_learners:
        l.run_training(session=session, global_steps_counter=global_steps_counter)
    for l in actor_learners:
        l.join()


if __name__ == "__main__":
    # TODO make tqdm work when stderr is redirected
    # TODO print setup info on stderr and stdout
    args = parse_train_a3c_args()

    default_settings_filepath = "settings/a3c_defaults.yml"
    a3c_settings = yaml.safe_load(open(default_settings_filepath))
    for settings_fpath in args.settings_yml:
        override_settings = yaml.safe_load(open(settings_fpath))
        a3c_settings.update(override_settings)

    if not os.path.isdir(a3c_settings["models_path"]):
        os.makedirs(a3c_settings["models_path"])
    if not os.path.isdir(a3c_settings["logdir"]):
        os.makedirs(a3c_settings["logdir"])

    train_a3c(a3c_settings)
