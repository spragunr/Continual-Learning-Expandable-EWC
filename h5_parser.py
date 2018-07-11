import h5py
import numpy as np
from copy import deepcopy
import plot_utils

f = h5py.File('55tasks_lam150_50hidden_1layer_mnist_ewc_expansion.hdf5', 'r')

avg_accs = []
expansions = []


for data in f["avg_acc_on_all_tasks"]:

    avg_accs.append(data)

for data in f["expansion_before_tasks"]:

    expansions.append(data)

f.close()

expansion_indices = []

for i in range(len(expansions)):
    if expansions[i] == 1:
        expansion_indices.append(i)

avg_accs = avg_accs[:56]


f = h5py.File('55tasks_lam150_50hidden_1layer_no_expansion.hdf5', 'r')

avg_accs2 = []
single_task_accs2 = []

for data in f["avg_acc_on_all_tasks"]:

    avg_accs2.append(data)

for data in f["final_task_accs"]:
    single_task_accs2.append(data)

f.close()



plot_utils.plot_line_avg_acc(avg_accs, expansion_indices, 90, "expansion", avg_accs2, "fixed")

# forgot to save these, so had to reconstruct from terminal output
single_task_accs = [95,95,93,89,94,94,92,89,94,92,90,91,94,94,92,91,88,93,94,92,90,87,87,94,92,93,92,90,90,85,85,91,94,
                    93,92,85,90,91,84,86,81,76,92,95,91,90,92,92,89,89,80,83,83,78,97]

plot_utils.plot_bar_each_task_acc(single_task_accs, "expansion", single_task_accs2, "fixed")