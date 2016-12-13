#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ruamel.yaml as yaml
import os
from util.parsers import parse_train_async_args
from util.coloring import green
from async_learner import A3CLearner, ADQNLearner


def train_async(q_learning, settings):
    import tensorflow as tf

    from vizdoom_wrapper import VizdoomWrapper
    from networks import create_network
    from util import ThreadsafeCounter
    from util.optimizers import ClippingRMSPropOptimizer

    tmp_vizdoom_wrapper = VizdoomWrapper(noinit=True, **settings)
    actions_num = tmp_vizdoom_wrapper.actions_num
    misc_len = tmp_vizdoom_wrapper.misc_len
    img_shape = tmp_vizdoom_wrapper.img_shape
    del tmp_vizdoom_wrapper

    # TODO target global network
    # This global step counts gradient applications not performed actions.
    global_train_step = tf.Variable(0, trainable=False, name="global_step")
    global_learning_rate = tf.train.polynomial_decay(
        learning_rate=settings["initial_learning_rate"],
        end_learning_rate=settings["final_learning_rate"],
        decay_steps=settings["learning_rate_decay_steps"],
        global_step=global_train_step)
    optimizer = ClippingRMSPropOptimizer(learning_rate=global_learning_rate, **settings["rmsprop"])

    actor_learners = []
    global_network = create_network(actions_num=actions_num, misc_len=misc_len, img_shape=img_shape, **settings)
    if q_learning:
        global_target_network = None
        for i in range(settings["threads_num"]):
            actor_learner = ADQNLearner(thread_index=i, global_network=global_network,
                                        global_target_network=global_target_network, optimizer=optimizer,
                                        **settings)
            actor_learners.append(actor_learner)
    else:
        for i in range(settings["threads_num"]):
            actor_learner = A3CLearner(thread_index=i, global_network=global_network, optimizer=optimizer,
                                       **settings)
            actor_learners.append(actor_learner)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    print("Initializing variables...")
    session.run(tf.global_variables_initializer())
    print("Initialization finished.")
    global_steps_counter = ThreadsafeCounter()

    # TODO print settings
    print(green("Launching training."))
    for l in actor_learners:
        l.run_training(session, global_steps_counter=global_steps_counter)
    for l in actor_learners:
        l.join()


if __name__ == "__main__":
    # TODO make tqdm work when stderr is redirected
    # TODO print setup info on stderr and stdout
    args = parse_train_async_args()

    if args.q:
        default_settings_filepath = "settings/adqn_defaults.yml"
    else:
        default_settings_filepath = "settings/a3c_defaults.yml"

    print("Loading default settings from:", default_settings_filepath)
    settings = yaml.safe_load(open(default_settings_filepath))
    for settings_fpath in args.settings_yml:
        print("Loading settings from:", settings_fpath)
        override_settings = yaml.safe_load(open(settings_fpath))
        settings.update(override_settings)

    if not os.path.isdir(settings["models_path"]):
        os.makedirs(settings["models_path"])
    if not os.path.isdir(settings["logdir"]):
        os.makedirs(settings["logdir"])

    train_async(args.q, settings)
