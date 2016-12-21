#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import trange
from time import time
from replay_memory import ReplayMemory


def test_memory(insertions, samples, img_shape, misc_len, batch_size, capacity, img_dtype=np.float32):
    print("image shape:", img_shape)
    print("misc vector lenght:", misc_len)
    print("batchsize:", batch_size)
    print("capacity:", capacity)
    print("image data type:", img_dtype.__name__)
    memory = ReplayMemory(img_shape, misc_len, capacity, batch_size)
    if img_dtype != np.float32:
        s = [(np.random.random(img_shape) * 255).astype(img_dtype), np.random.random(misc_len).astype(np.float32)]
        s2 = [(np.random.random(img_shape) * 255).astype(img_dtype), np.random.random(misc_len).astype(np.float32)]
    else:
        s = [np.random.random(img_shape).astype(img_dtype), np.random.random(misc_len).astype(np.float32)]
        s2 = [np.random.random(img_shape).astype(img_dtype), np.random.random(misc_len).astype(np.float32)]
    a = 0
    r = 1.0
    terminal = False
    for _ in trange(capacity, leave=False, desc="Prefilling memory."):
        memory.add_transition(s, a, s2, r, terminal)

    start = time()
    for _ in trange(insertions, leave=False, desc="Testing insertions speed"):
        memory.add_transition(s, a, s2, r, terminal)
    inserts_time = time() - start

    start = time()
    for _ in trange(samples, leave=False, desc="Testing sampling speed"):
        sample = memory.get_sample()
    sample_time = time() - start

    print("\t{:0.1f} insertions/s. 1k insertions in: {:0.2f}s".format(insertions / inserts_time,
                                                                      inserts_time / insertions * 1000))
    print("\t{:0.1f} samples/s. 1k samples in: {:0.2f}s".format(samples / sample_time, sample_time / samples * 1000))
    print()


baseline = {"img_shape": (4, 60, 80), "misc_len": 1, "batch_size": 64, "capacity": 10000}
bigger_capacity = {"img_shape": (4, 60, 80), "misc_len": 1, "batch_size": 64, "capacity": 50000}
small_state = {"img_shape": (4, 30, 40), "misc_len": 1, "batch_size": 64, "capacity": 10000}
uint_img = {"img_shape": (4, 30, 40), "misc_len": 1, "batch_size": 64, "capacity": 10000, "img_dtype": np.uint8}
insertions_num = 100000
samples_num = 1000

test_memory(insertions=insertions_num, samples=samples_num, **baseline)
test_memory(insertions=insertions_num, samples=samples_num, **bigger_capacity)
test_memory(insertions=insertions_num, samples=samples_num, **small_state)
test_memory(insertions=insertions_num, samples=samples_num, **uint_img)

