import argparse
import torch
import numpy as np
import scipy as sp
import random
from VanillaMLP import VanillaMLP
from VanillaCNN import VanillaCNN
from EWCMLP import EWCMLP
from EWCCNN import EWCCNN
import h5py
from pathlib import Path
import subprocess
import pickle
import torch
import os
import os.path



def parse_arguments():

    parser = argparse.ArgumentParser(description='Variable Capacity Network for Continual Learning')

    parser.add_argument('--experiment', type=str, default='custom', metavar='EXPERIMENT',
                        help='preconfigured experiment to run: mnist or cifar (defaults to custom)')

    parser.add_argument('--batch-size', type=int, default=100, metavar='BS',
                        help='input batch size for training')

    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='TBS',
                        help='input batch size for testing')

    parser.add_argument('--epochs', type=int, default=1, metavar='E',
                        help='number of epochs to train')

    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate')

    parser.add_argument('--l2-reg-penalty', type=float, default=0.0, metavar='L2',
                        help='l2 regularization penalty (weight decay) (default: 0.0)')

    parser.add_argument('--lam', type=float, default=15, metavar='LAM',
                        help='ewc lambda value (fisher multiplier) (default: 15)')

    parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                        help='SGD momentum (default: 0.0)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='seed for RNGs')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status (default 10)')

    # [train dataset size] = [full MNIST train set (60,000)] - [validation set size]
    parser.add_argument('--train-dataset-size', type=int, default=59800, metavar='TDS',
                        help='number of images in the training dataset')

    parser.add_argument('--validation-dataset-size', type=int, default=200, metavar='VDS',
                        help='number of images in the validation dataset')

    # size of hidden layer in MLP in neurons OR initial number of filters in conv network
    parser.add_argument('--hidden-size', type=int, default=20, metavar='HS',
                        help='# neurons in each hidden layer of MLP OR # filters in conv resnet')

    # 28 x 28 pixels = 784 pixels per MNIST image, 32 x 32 = 1024 for CIFAR 10
    parser.add_argument('--input-size', type=int, default=784, metavar='IS',
                        help='size of each input data sampe to the network (default 784 (28 * 28))')

    # 10 classes - digits 0-9 for MNIST, 100 for CIFAR 100
    parser.add_argument('--output-size', type=int, default=10, metavar='OS',
                        help='size of the output of the network (default 10)')

    # e.g. 2 to double the size of the network when expansion occurs
    parser.add_argument('--scale-factor', type=int, default=2, metavar='ES',
                        help='the factor by which to scale the size of network layers upon expansion')

    # accuracy threshold (minimum) required on most recent task- if not met, network will reset to pre-training state
    # and expand - set to 0 to disable automatic network expansion
    parser.add_argument('--accuracy-threshold', type=int, default=0, metavar='AT',
                        help='accuracy threshold (minimum) required on all tasks')

    # dataset on which to train/test model
    parser.add_argument('--dataset', type=str, default='mnist', metavar='DS',
                        help='dataset on which to train/test model (cifar100 or mnist)')

    # number of tasks
    parser.add_argument('--tasks', type=int, default=100, metavar='T',
                        help='number of tasks')

    parser.add_argument('--output-file', type=str, default='failure2', metavar='OUTPUT FILE',
                        help='h5 file for storage of experimental results')

    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='overwrite old result data files without asking (will prompt otherwise)')

    parser.add_argument('--nets', nargs='+', type=str, default=['EWCMLP'], metavar='NETS',
                        help='neural net classes to train')

    parser.add_argument('--run', type=int, default=0, metavar='RUN',
                        help='if running repeatedly, use this arg to signal from bash script what run # this is')
    
    parser.add_argument('--perm', type=int, default=100, metavar='PERM',
                        help='percent permutation to be applied to mnist images')

    args = parser.parse_args()

    if args.experiment == 'mnist':

        print('|-----[CONDUCTING PERMUTED MNIST EXPERIMENT]-----|')

        args.batch_size = 100 # TODO change back to 100
        args.test_batch_size = 1000 # TODO change back to 1000
        args.epochs = 1 # TODO change back to 10
        args.lr = 0.1 # TODO change back to 0.01
        args.l2_reg_penalty = 0.0
        args.lam = 15 # TODO change back to 150
        args.momentum = 0.0
        args.no_cuda = False
        args.seed = 1
        args.log_interval = 10
        args.train_dataset_size = 59800
        args.validation_dataset_size = 200
        args.hidden_size = 20
        args.input_size = 784
        args.output_size = 10
        args.scale_factor = 2
        args.accuracy_threshold = 0
        args.dataset = 'mnist'
        args.tasks = 100
        args.output_file = 'perm{}'.format(args.perm) 
        args.nets = ['EWCMLP']
        

        args_dict = vars(args)

        for k in args_dict.keys():
            if k != 'nets':
                print("{:_<30}{:_>30}".format(k, args_dict.get(k)))
            else:
                print("{:_<30}{:_>30}".format(k, ", ".join(args_dict.get(k))))

    elif args.experiment == 'cifar':

        print('|-----[CONDUCTING INCREMENTAL CIFAR 100 EXPERIMENT]-----|')

        args.batch_size = 4
        args.test_batch_size = 4
        args.epochs =  10
        args.lr = 0.01
        args.l2_reg_penalty = 0.0
        args.lam = 1500000
        args.momentum = 0.0
        args.no_cuda = False
        args.seed = 1
        args.log_interval = 10
        args.hidden_size = 64 # todo note changes in output layer sizes as a result of this
        args.validation_dataset_size = 40 # in THIS case, this is the validation data from each individual CLASS
        args.input_size = 1024
        args.output_size = 100
        args.scale_factor = 1 # in this case, we ADD this many filters to first convolutional layer...
        args.accuracy_threshold = 0 # todo figure out what this should be...
        args.dataset = 'cifar'
        args.tasks = 20
        args.output_file = 'nonexpanding_increm_cifar_lam_{}_all_fil_{}_512_cw_scale_2'.format(args.lam, args.hidden_size)
        args.nets = ['EWCCNN'] # todo change to EWCCNN
        #args.samples_per_task = -1 # todo add this to the arg parser
        #args.shuffle_tasks = 'no' # todo add this to the arg parser

        # per-class train dataset size * classes per task
        args.train_dataset_size = (500 - args.validation_dataset_size) * (100 // args.tasks)

        # per-class validation data size * classes per task
        args.validation_dataset_size *= (100 // args.tasks)

        args_dict = vars(args)

        for k in args_dict.keys():
            if k != 'nets':
                print("{:_<30}{:_>30}".format(k, args_dict.get(k)))
            else:
                print("{:_<30}{:_>30}".format(k, ", ".join(args_dict.get(k))))

    elif args.experiment == 'custom':

        print('|-----[CUSTOM EXPERIMENT- DEFAULT HYPERPARAMETERS USED WHERE NOT SPECIFIED]-----|')

        if args.dataset == "cifar":
            # per-class train dataset size * classes per task
            args.train_dataset_size = (500 - args.validation_dataset_size) * (100 // args.tasks)

            # per-class validation data size * classes per task
            args.validation_dataset_size *= (100 // args.tasks)


        args_dict = vars(args)

        for k in args_dict.keys():
            if k != 'nets':
                print("{:_<30}{:_>30}".format(k, args_dict.get(k)))
            else:
                print("{:_<30}{:_>30}".format(k, ", ".join(args_dict.get(k))))

    else:

        raise ValueError("Invalid experiment type selected: {}!\n".format(args.experiment))

    return args

def seed_rngs(args):

    # set a manual seed for PyTorch CPU random number generation
    torch.manual_seed(args.seed)

    # set a manual seed for PyTorch GPU random number generation
    torch.cuda.manual_seed_all(args.seed)

    # set a manual seed for NumPy random number generation
    np.random.seed(args.seed)

    # set a manual seed for SciPy random number generation
    sp.random.seed(args.seed)

    # set a manual seed for python random number generation
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True

def set_gpu_options(args):

    # determines if CUDA should be used - only if available AND not disabled via arguments
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # arguments specific to CUDA computation
    # num_workers: how many subprocesses to use for data loading - if set to 0, data will be loaded in the main process
    # pin_memory: if True, the DataLoader will copy tensors into CUDA pinned memory before returning them
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # set the device on which to perform computations - later calls to .to(device) will move tensors to GPU or CPU
    # based on the value determined here
    device = torch.device("cuda" if use_cuda else "cpu")

    return kwargs, device

def build_models(args, device):

    models = []

    for net in args.nets:

        if net == "VanillaMLP":

            models.append(VanillaMLP(
                args.hidden_size,
                args.input_size,
                args.output_size,
                device,
            ).to(device))

        elif net == "VanillaCNN":

            models.append(VanillaCNN(
                args.hidden_size,
                args.input_size,
                args.output_size,
                device,
            ).to(device))

        elif net == "EWCMLP":

            models.append(
                EWCMLP(
                    args.hidden_size,
                    args.input_size,
                    args.output_size,
                    device,
                    lam=args.lam  # the lambda (fisher multiplier) value to be used in the EWC loss formula
                ).to(device))

        elif net == "EWCCNN":

            models.append(
                EWCCNN(
                    args.hidden_size,
                    args.input_size,
                    args.output_size,
                    device,
                    lam=args.lam  # the lambda (fisher multiplier) value to be used in the EWC loss formula
                ).to(device))

        else:
            raise TypeError("Invalid Neural Network Type Specified: {}\n".format(net))

    return models


def setup_h5_file(args, models):
    
    DIR = "final"

    files = []
    expansions_list = []
    avg_acc_list = []
    task_acc_list = []

    # used to store variable length unicode strings in h5 format with Python 3.x
    dt = h5py.special_dtype(vlen=str)

    for model in models:

        model_type = str(type(model))
        model_type = model_type[model_type.index("'") + 1:model_type.rindex('.')]
        
        filename = DIR + "/" + model_type + "_" + args.output_file + "_run_{}.h5".format(args.run)

        path = Path(filename)

        mode = "x" # throw an exception if the file already exists

        # file already exists - ask if we want to get rid of the old one
        if path.is_file():

            if not args.overwrite:

                if input("\nWOULD YOU LIKE TO OVERWRITE OLD DATA FILE {}?[y/n]:  ".format(filename)).strip().lower() == "y":

                    mode = "w" # truncate and write over the file if it exists

            else:

                mode = "w"

        f = h5py.File(filename, mode)

        files.append(f)

        metadata = f.create_dataset("metadata", (len(vars(args).keys()),), dtype=dt)

        for i, k in enumerate(vars(args).keys()):
            metadata[i] = "{}: {}".format(k, vars(args).get(k))


        # NOTE: TO FACILITATE PARSING THERE IS A ZERO TACKED ONTO THE FRONT OF THIS LIST
        # [0, 0, 1, 0, 2, 0] would mean that the network had to expand 0 times before successfully learning the 1st task,
        # once before successfully learning the 2nd task, 0 MORE times before learning the 3rd task, twice MORE before
        # successfully learning the 4th task (total of 3 expansions now) and 0 MORE times before successfully learning
        # the 5th task
        expansions = f.create_dataset("expansions", (args.tasks + 1,), dtype='i')
        expansions[...] = np.zeros(len(expansions))
        expansions_list.append(expansions)

        # NOTE: TO FACILITATE PARSING THERE IS A ZERO TACKED ONTO THE FRONT OF THIS LIST
        # avg accuracy on all tasks as new tasks are added
        avg_acc = f.create_dataset("avg_acc", (args.tasks + 1,), dtype='f')
        avg_acc[...] = np.zeros(len(avg_acc))
        avg_acc_list.append(avg_acc)

        # NOTE: TO FACILITATE PARSING THERE IS A ZERO TACKED ONTO THE FRONT OF THIS LIST
        # final post-training accuracy on each individual task
        task_acc = f.create_dataset("task_acc", (args.tasks + 1,), dtype='f')
        task_acc[...] = np.zeros(len(task_acc))
        task_acc_list.append(task_acc)

        

    # todo fix the models list style so only one model at a time, and make these lists into single h5 datasets
    return files, expansions_list, avg_acc_list, task_acc_list, f
