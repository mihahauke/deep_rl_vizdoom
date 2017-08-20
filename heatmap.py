#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools as it
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use("ggplot")
np.set_printoptions(precision=2)

filename = "stats/cfigar_stoch.txt"
if len(sys.argv) > 1:
    filename = sys.argv[1]

stats = pickle.load(open(filename, "br"))

frameskips = stats["frameskips"]
actions = stats["actions"]
# print(np.unique(frameskips))
# print(frameskips)
# exit(0)
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

fig, axes = plt.subplots(1, 2)
fig.canvas.set_window_title(filename)

axes[0].set_title("")
axes[1].set_title("Action-wise normalized")

for i, d in enumerate([data, action_normalized_data]):
    a = sns.heatmap(d.T,
                    ax=axes[i],
                    square=True
                    )
    a.set_yticklabels(a_labels, rotation=0, fontsize=12)
    a.set_xticklabels(fs_values, rotation=0, fontsize=12)
    # a.xaxis.set_title("asd")
    a.set_xlabel('frameskip', )
    a.set_ylabel('action')

plt.show()
