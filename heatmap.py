#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools as it
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from argparse import ArgumentParser
from util.misc import string_heatmap

plt.style.use("ggplot")
np.set_printoptions(precision=2)

parser = ArgumentParser()
parser.add_argument(dest="input_filename")
parser.add_argument("--cli", default=False, action="store_true", dest="cli")

args = parser.parse_args()

stats = pickle.load(open(args.input_filename, "br"))

frameskips = stats["frameskips"]
actions = stats["actions"]

min_frameskip = frameskips.min()
max_frameskip = frameskips.max()
fs_values = range(min_frameskip, max_frameskip + 1)
a_values = range(max(actions) + 1)
buttons_num = max(int(np.ceil(np.log2(len(a_values)))), 3)

a_labels = [str(l) for l in it.product([0, 1], repeat=buttons_num)]

data = np.zeros((len(fs_values), len(a_values)))

for f, a in zip(frameskips, actions):
    data[f - min_frameskip, a] += 1

s = data.sum(0)
s[s == 0] = 1
action_normalized_data = data / s
data /= data.sum()

if args.cli:
    print(string_heatmap(data.T, fs_values, a_labels))
    print()
    print("Action-wise normalized")
    print(string_heatmap(action_normalized_data.T, fs_values, a_labels))

else:
    fig, axes = plt.subplots(1, 2)
    fig.canvas.set_window_title(args.input_filename)

    axes[0].set_title("")
    axes[1].set_title("Action-wise normalized")

    for i, d in enumerate([data, action_normalized_data]):
        a = sns.heatmap(d.T,
                        ax=axes[i],
                        square=True
                        )
        a.set_yticklabels(a_labels, rotation=0, fontsize=12)
        a.set_xticklabels(fs_values, rotation=0, fontsize=12)
        a.set_xlabel('frameskip', )
        a.set_ylabel('action')
    plt.show()
