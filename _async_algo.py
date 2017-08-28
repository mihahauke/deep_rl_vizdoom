#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle

import numpy as np
import tensorflow as tf

import async_learner
import networks
from util import ThreadsafeCounter
from util import ensure_parent_directories
from util.coloring import green
from util.logger import log
from util.optimizers import ClippingRMSPropOptimizer
from vizdoom_wrapper import VizdoomWrapper, FakeVizdoomWrapper


def train_async(model_savefile, q_learning, settings):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    if settings["fake_game"]:
        Game = FakeVizdoomWrapper
    else:
        Game = VizdoomWrapper
    proto_vizdoom = Game(noinit=True, **settings)
    actions_num = proto_vizdoom.actions_num
    misc_len = proto_vizdoom.misc_len
    img_shape = proto_vizdoom.img_shape
    del proto_vizdoom

    # This global step counts gradient applications not performed actions.
    global_train_step = tf.Variable(0, trainable=False, name="global_step")
    global_learning_rate = tf.train.polynomial_decay(
        name="larning_rate",
        learning_rate=settings["initial_learning_rate"],
        end_learning_rate=settings["final_learning_rate"],
        decay_steps=settings["decay_steps"],
        global_step=global_train_step)
    optimizer = ClippingRMSPropOptimizer(learning_rate=global_learning_rate, **settings["rmsprop"])
    learners = []
    NetworkClass = getattr(networks, settings["network_class"])

    global_network = NetworkClass(
        actions_num=actions_num,
        misc_len=misc_len,
        img_shape=img_shape,
        **settings)
    global_steps_counter = ThreadsafeCounter()

    if q_learning:
        global_target_network = NetworkClass(
            thread="global_target",
            actions_num=actions_num,
            misc_len=misc_len,
            img_shape=img_shape, **settings)
        global_network.prepare_unfreeze_op(global_target_network)
        unfreeze_thread = min(1, settings["threads_num"] - 1)
        for i in range(settings["threads_num"]):
            learner = async_learner.ADQNLearner(
                game=Game(**settings),
                model_savefile=model_savefile,
                thread_index=i, global_network=global_network,
                unfreeze_thread=i == unfreeze_thread,
                global_target_network=global_target_network,
                optimizer=optimizer,
                learning_rate=global_learning_rate,
                global_steps_counter=global_steps_counter,
                **settings)
            learners.append(learner)
    else:

        for i in range(settings["threads_num"]):
            LearnerClass = getattr(async_learner, settings["learner_class"])
            learner = LearnerClass(
                game=Game(**settings),
                model_savefile=model_savefile,
                thread_index=i,
                global_network=global_network,
                optimizer=optimizer,
                learning_rate=global_learning_rate,
                global_steps_counter=global_steps_counter,
                **settings)
            learners.append(learner)

    log("Initializing variables...")
    session.run(tf.global_variables_initializer())
    log("Initialization finished.\n")

    if q_learning:
        session.run(global_network.ops.unfreeze)

    log(green("Started training.\n"))

    for l in learners:
        l.run_training(session)
    for l in learners:
        l.join()


def test_async(q_learning, settings, modelfile, eps, deterministic=True, output=None):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.InteractiveSession(config=config)

    # TODO: it's a workaround polynomial decays use global step, remove it somehow
    tf.Variable(0, trainable=False, name="global_step")

    if q_learning:
        agent = async_learner.ADQNLearner(thread_index=0, session=session, **settings)
    else:
        LearnerClass = getattr(async_learner, settings["learner_class"])
        agent = LearnerClass(thread_index=0, session=session, **settings)

    log("Initializing variables...")
    session.run(tf.global_variables_initializer())
    log("Initialization finished.\n")

    agent.load_model(session, modelfile)

    log("\nScores: ")
    scores = []

    stats = {"episodes": [],
             "actions": [],
             "frameskips": []}

    for _ in range(eps):
        score, actions, frameskips, rewards = agent.run_episode(deterministic=deterministic, return_stats=True)
        scores.append(score)
        print("{0:3f}".format(score))
        if output is not None:
            episode_stats = {"score": score,
                             "rewards": rewards,
                             "actions": actions,
                             "frameskips": frameskips}
            stats["actions"] += actions
            stats["frameskips"] += frameskips
            stats["episodes"].append(episode_stats)

    print()
    log("\nMean score: {:0.3f}".format(np.mean(scores)))

    if output is not None:
        stats["actions"] = np.array(stats["actions"], dtype=np.int16)
        stats["frameskips"] = np.array(stats["frameskips"], dtype=np.int16)

        ensure_parent_directories(output)
        pickle.dump(stats, open(output, "wb"))
