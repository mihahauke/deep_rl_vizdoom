# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from vizdoom_wrapper import VizdoomWrapper

class DQN(object):
    def __init__(self,
                 write_summaries=True,
                 **settings):

        self.doom_wrapper = VizdoomWrapper(**settings)
        self.write_summaries = write_summaries
        pass

    def train(self):
        pass