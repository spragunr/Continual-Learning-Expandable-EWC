import h5py
import numpy as np

f = h5py.File('test_results.hdf5', 'r')

datasets = []

for key in f.keys():
    datasets.append(f[key])

f.close()

for dataset in datasets:
    print(dataset)