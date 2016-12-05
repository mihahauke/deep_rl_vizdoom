#!/usr/bin/env python3

import random
import numpy as np


class ReplayMemory:
    def __init__(self, format, capacity=10000, batch_size=32):
        sim_shape = list(format["s_img"])
        sim_shape.insert(0, capacity)
        self._s1_img = np.zeros(sim_shape, dtype=np.float32)
        self._s2_img = np.zeros(sim_shape, dtype=np.float32)
        self._a = np.zeros(capacity, dtype=np.int32)
        self._r = np.zeros(capacity, dtype=np.float32)
        self._nonterminal = np.zeros(capacity, dtype=np.bool_)

        sim_shape[0] = batch_size
        self._s1_img_buf = np.zeros(sim_shape, dtype=np.float32)
        self._s2_img_buf = np.zeros(sim_shape, dtype=np.float32)
        self._a_buf = np.zeros(batch_size, dtype=np.int32)
        self._r_buf = np.zeros(batch_size, dtype=np.float32)
        self._nonterminal_buf = np.zeros(batch_size, dtype=np.bool_)

        if format["s_misc"] > 0:
            self._s1_misc = np.zeros((capacity, format["s_misc"]), dtype=np.float32)
            self._s2_misc = np.zeros((capacity, format["s_misc"]), dtype=np.float32)
            self._s1_misc_buf = np.zeros((batch_size, format["s_misc"]), dtype=np.float32)
            self._s2_misc_buf = np.zeros((batch_size, format["s_misc"]), dtype=np.float32)
            self._misc = True
        else:
            self._s1_misc = None
            self._s2_misc = None
            self._s1_misc_buf = None
            self._s2_misc_buf = None
            self._misc = False

        self._capacity = capacity
        self.size = 0
        self._oldest_index = 0
        self._batch_size = batch_size

        ret = dict()
        ret["s1_img"] = self._s1_img_buf
        ret["s1_misc"] = self._s1_misc_buf
        ret["a"] = self._a_buf
        ret["s2_img"] = self._s2_img_buf
        ret["s2_misc"] = self._s2_misc_buf
        ret["r"] = self._r_buf
        ret["nonterminal"] = self._nonterminal_buf

        self._ret_dict = ret.copy()

    def add_transition(self, s1, a, s2, r, terminal=False):
        if self.size < self._capacity:
            self.size += 1

        self._s1_img[self._oldest_index] = s1[0]

        if not terminal:
            self._s2_img[self._oldest_index] = s2[0]

        if self._misc:
            self._s1_misc[self._oldest_index] = s1[1]
            if not terminal:
                self._s2_misc[self._oldest_index] = s2[1]

        self._a[self._oldest_index] = a
        self._r[self._oldest_index] = r
        self._nonterminal[self._oldest_index] = not terminal

        self._oldest_index = (self._oldest_index + 1) % self._capacity

    def get_sample(self):
        if self._batch_size > self.size:
            raise Exception("Transition bank doesn't contain " + str(self._batch_size) + " entries.")

        indexes = random.sample(range(0, self.size), self._batch_size)
        self._s1_img_buf[:] = self._s1_img[indexes]
        self._s2_img_buf[:] = self._s2_img[indexes]
        if self._misc:
            self._s1_misc_buf[:] = self._s1_misc[indexes]
            self._s2_misc_buf[:] = self._s2_misc[indexes]
        self._a_buf[:] = self._a[indexes]
        self._r_buf[:] = self._r[indexes]
        self._nonterminal_buf[:] = self._nonterminal[indexes]
        return self._ret_dict
