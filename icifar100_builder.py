# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import subprocess
import pickle
import torch
import os
import argparse
import os.path

cifar_path = "cifar-100-python.tar.gz"

# URL from: https://www.cs.toronto.edu/~kriz/cifar.html
if not os.path.exists(cifar_path):
    subprocess.call("wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz", shell=True)

subprocess.call("tar xzfv cifar-100-python.tar.gz", shell=True)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cifar100_train = unpickle('cifar-100-python/train')
cifar100_test = unpickle('cifar-100-python/test')

x_tr = torch.from_numpy(cifar100_train[b'data'])
y_tr = torch.LongTensor(cifar100_train[b'fine_labels'])
x_te = torch.from_numpy(cifar100_test[b'data'])
y_te = torch.LongTensor(cifar100_test[b'fine_labels'])

torch.save((x_tr, y_tr, x_te, y_te), 'cifar100.pt')

parser = argparse.ArgumentParser()

# todo arg
parser.add_argument('--i', default='raw/cifar100.pt', help='input directory')

# todo arg
parser.add_argument('--o', default='cifar100.pt', help='output file')

# todo connect to arg
parser.add_argument('--n_tasks', default=20, type=int, help='number of tasks')

# todo connect to arg
parser.add_argument('--seed', default=1, type=int, help='random seed')

args = parser.parse_args()

# todo remove- this should already have happened
torch.manual_seed(args.seed)

tasks_tr = []
tasks_te = []

x_tr, y_tr, x_te, y_te = torch.load(os.path.join(args.i))
x_tr = x_tr.float().view(x_tr.size(0), -1) / 255.0
x_te = x_te.float().view(x_te.size(0), -1) / 255.0

cpt = int(100 / args.n_tasks)

for t in range(args.n_tasks):
    c1 = t * cpt
    c2 = (t + 1) * cpt
    i_tr = ((y_tr >= c1) & (y_tr < c2)).nonzero().view(-1)
    i_te = ((y_te >= c1) & (y_te < c2)).nonzero().view(-1)
    tasks_tr.append([(c1, c2), x_tr[i_tr].clone(), y_tr[i_tr].clone()])
    tasks_te.append([(c1, c2), x_te[i_te].clone(), y_te[i_te].clone()])

torch.save([tasks_tr, tasks_te], args.o)

def load_datasets(args):
    d_tr, d_te = torch.load(args.data_path + '/' + args.data_file)
    n_inputs = d_tr[0][1].size(1)
    n_outputs = 0
    for i in range(len(d_tr)):
        n_outputs = max(n_outputs, d_tr[i][2].max())
        n_outputs = max(n_outputs, d_te[i][2].max())
    return d_tr, d_te, n_inputs, n_outputs + 1, len(d_tr)



