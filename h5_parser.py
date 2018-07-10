import h5py
import numpy as np
from copy import deepcopy

f = h5py.File('test_results.hdf5', 'r')

datasets = []

for key in f.keys():
    datasets.append(deepcopy(f[key]))

f.close()

for dataset in datasets:
    print(dataset)