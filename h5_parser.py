import h5py
import numpy as np

f = h5py.File('test_results.hdf5', 'r')

for key in f.keys():
    print(f[key])

f.close()