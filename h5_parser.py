import h5py
import numpy as np
from copy import deepcopy

f = h5py.File('55tasks_lam150_50hidden_1layer_mnist_ewc_expansion.hdf5', 'r')

datasets = []

for key in f.keys():
    dataset = []

    for data in f[key]:
        dataset.append(data)

    datasets.append(dataset)

f.close()

for dataset in datasets:
    print(dataset)