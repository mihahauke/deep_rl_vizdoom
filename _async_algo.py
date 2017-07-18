#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from util.coloring import green
from async_learner import A3CLearner, ADQNLearner
from util.logger import log
from vizdoom_wrapper import VizdoomWrapper
from util import ThreadsafeCounter
from util.optimizers import ClippingRMSPropOptimizer
import networks
import tensorflow as tf
import numpy as np


def train_async(q_learning, settings):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    proto_vizdoom = VizdoomWrapper(noinit=True, **settings)
    actions_num = proto_vizdoom.actions_num
    misc_len = proto_vizdoom.misc_len
    img_shape = proto_vizdoom.img_shape
    del proto_vizdoom

    # TODO target global network
    # This global step counts gradient applications not performed actions.
    global_train_step = tf.Variable(0, trainable=False, name="global_step")
    global_learning_rate = tf.train.polynomial_decay(
        name="larning_rate",
        learning_rate=settings["initial_learning_rate"],
        end_learning_rate=settings["final_learning_rate"],
        decay_steps=settings["learning_rate_decay_steps"],
        global_step=global_train_step)
    optimizer = ClippingRMSPropOptimizer(learning_rate=global_learning_rate, **settings["rmsprop"])

    learners = []
    network_class = eval(settings["network_type"])

    global_network = network_class(actions_num=actions_num, misc_len=misc_len, img_shape=img_shape,
                                   **settings)

    global_steps_counter = ThreadsafeCounter()
    if q_learning:
        global_target_network = network_class(thread="global_target", actions_num=actions_num,
                                              misc_len=misc_len,
                                              img_shape=img_shape, **settings)
        global_network.prepare_unfreeze_op(global_target_network)
        unfreeze_thread = min(1, settings["threads_num"] - 1)
        for i in range(settings["threads_num"]):
            learner = ADQNLearner(thread_index=i, global_network=global_network,
                                  unfreeze_thread=i == unfreeze_thread,
                                  global_target_network=global_target_network,
                                  optimizer=optimizer,
                                  learning_rate=global_learning_rate,
                                  global_steps_counter=global_steps_counter,
                                  **settings)
            learners.append(learner)
    else:
        for i in range(settings["threads_num"]):
            learner = A3CLearner(thread_index=i, global_network=global_network,
                                 optimizer=optimizer, learning_rate=global_learning_rate,
                                 global_steps_counter=global_steps_counter,
                                 **settings)
            learners.append(learner)



    log("Initializing variables...")
    session.run(tf.global_variables_initializer())
    log("Initialization finished.\n")

    if q_learning:
        session.run(global_network.ops.unfreeze)

    log(green("Starting training.\n"))

    for l in learners:
        l.run_training(session)
    for l in learners:
        l.join()


def test_async(q_learning, settings, modelfile, eps, deterministic=True):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.InteractiveSession(config=config)

    # TODO: it's a workaround polynomial decays use global step, remove it somehow
    tf.Variable(0, trainable=False, name="global_step")

    if q_learning:
        agent = ADQNLearner(thread_index=0, session=session, **settings)
    else:
        agent = A3CLearner(thread_index=0, session=session, **settings)

    log("Initializing variables...")
    session.run(tf.global_variables_initializer())
    log("Initialization finished.\n")

    agent.load_model(session, modelfile)

    log("\nScores: ")
    scores = []

    for _ in range(eps):
        reward = agent.run_episode(deterministic=deterministic)
        scores.append(reward)
        print("{0:3f}".format(reward))
    print()
    log("\nMean score: {:0.3f}".format(np.mean(scores)))
