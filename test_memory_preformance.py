#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import trange
from time import time
from replay_memory import ReplayMemory


def test_memory(inserts, samples, img_shape, misc_len, batch_size, capacity):
    memory = ReplayMemory(img_shape, misc_len, capacity, batch_size)
    # TODO prepare data
    # TODO check how trange affects measurements
    start = time()
    for _ in trange(inserts):
        # TODO
        pass
    inserts_time = time() - start

    start = time()
    for _ in trange(samples):
        sample = memory.get_sample()
    sample_time = time() - start

    print("{:0f} insertion")


baseline = {"img_shape": (4, 60, 80), "misc_len": 0, "batch_size": 64, "capacity": 10000}
test_memory(inserts=1000, samples=1000, **baseline)
