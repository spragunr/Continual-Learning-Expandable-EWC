import argparse
import torch
import torch.optim as optim
import torch.utils.data as D
import numpy as np
from utils import apply_permutation
from torchvision import datasets, transforms
from model import Model
from matplotlib import pyplot as plt

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--train-dataset-size', type=int, default=50000, metavar='TDS',
                        help='how many images to put in the training dataset')
    parser.add_argument('--validation-dataset-size', type=int, default=10000, metavar='VDS',
                        help='how many images to put in the validation dataset')
    args = parser.parse_args()

    # determines if CUDA should be used - only if available AND not disabled via arguments
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # arguments specific to CUDA computation
    # num_workers: how many subprocesses to use for data loading - if set to 0, data will be loaded in the main process
    # pin_memory: if True, the DataLoader will copy tensors into CUDA pinned memory before returning them
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    # set the device on which to perform computations - later calls to .to(device) will move tensors to GPU or CPU
    # based on the value determined here
    device = torch.device("cuda" if use_cuda else "cpu")

    # set a manual seed for random number generation
    torch.manual_seed(args.seed)

    # A PyTorch DataLoader combines a dataset and a sampler, and returns single- or multi-process iterators over
    # the dataset.

    # Split the PyTorch MNIST training dataset into training and validation datasets, and transform the data.
    #
    # D.dataset.random_split(dataset, lengths):
    #   Randomly split a dataset into non-overlapping new datasets of given lengths
    #
    # datasets.MNIST():
    #   ARGUMENTS (in order):
    #   root (string) – Root directory of dataset where processed/training.pt and processed/test.pt exist.
    #   train (bool, optional) – If True, creates dataset from training.pt, otherwise from test.pt.
    #   download (bool, optional) – If true, downloads the dataset from the internet and puts it in root directory.
    #                                       If dataset is already downloaded, it is not downloaded again.
    #   transform (callable, optional) – A function/transform that takes in an PIL image and returns a transformed
    #                                       version. E.g, transforms.RandomCrop
    train_data, validation_data = D.dataset.random_split(
        datasets.MNIST('../data', train=True,
                       # transforms.Compose() composes several transforms together.
                       #
                       # The transforms composed here are as follows:
                       #
                       # transforms.ToTensor():
                       #     Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a
                       #     torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
                       #
                       # transforms.Normalize(mean, std):
                       #     Normalize a tensor image with mean and standard deviation. Given mean: (M1,...,Mn) and
                       #     std: (S1,..,Sn) for n channels, this transform will normalize each channel of the
                       #     input torch.*Tensor i.e. input[channel] = (input[channel] - mean[channel]) / std[channel]
                       #
                       #     NOTE: the values used here for mean and std are those computed on the MNIST dataset
                       #           SOURCE: https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), [args.train_dataset_size, args.validation_dataset_size])

    # DataLoader for the training data.
    # ARGUMENTS (in order):
    # dataset (Dataset) – dataset from which to load the data (train_data prepared in above statement, in this case).
    # batch_size (int, optional) – how many samples per batch to load (default: 1).
    # shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).
    # kwargs - see above definition
    train_loader = D.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)

    # Dataloader for the validation dataset-
    # ARGUMENTS (in order):
    # 1) validation_data as the dataset
    # 2) we use this in compute_fisher by sampling a SINGLE image from the validation set per iteration,
    #       hence batch_size=1
    # 3) shuffle=True ensures we are drawing random samples by shuffling the data each time we contstruct a new iterator
    #       over the data, and is implemented in the source code as a RandomSampler() - see comments in compute_fisher
    #       for more details and a link to the source code
    # 4) kwargs defined above
    validation_loader = D.DataLoader(validation_data, batch_size=1, shuffle=True, **kwargs)

    #TODO comment
    test_data = datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))

    # Instantiate a DataLoader for the testing data in the same manner as above for training data, with one exception:
    #       train=False, because we want to draw the data here from <root>/test.pt (as opposed to <root>/training.pt)
    test_loader = D.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Move all parameters and buffers in the module Net to device (CPU or GPU- set above).
    # Both integral and floating point values are moved.
    model = Model().to(device)

    # Set the optimization algorithm for the model- in this case, Stochastic Gradient Descent with
    # momentum.
    #
    # ARGUMENTS (in order):
    #     params (iterable) – iterable of parameters to optimize or dicts defining parameter groups
    #     lr (float) – learning rate
    #     momentum (float, optional) – momentum factor (default: 0)
    #
    # NOTE on params:
    #   model.parameters() returns an iterator over a list of the trainable model parameters in the same order in
    #   which they appear in the network when traversed input -> output
    #   (e.g.
    #       [weights b/w input and first hidden layer,
    #        bias b/w input and hidden layer 1,
    #        ... ,
    #        weights between last hidden layer and output,
    #        bias b/w hidden layer and output]
    #   )
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # TODO seed NUMPY random number generator (different from PyTorch)

    # TODO UPDATE ALL COMMENTS TO REFLECT THE FACT THAT there are image/label combos(??) (tuple)...


    # TODO comment
    train_loaders = []
    validation_loaders = []
    test_loaders = []

    # keep learning tasks ad infinitum
    while(True):
        print()
        #TODO comment each step in this loop
        # for each desired epoch, train and test the model
        for epoch in range(1, args.epochs + 1):
            model.train_step(args, device, train_loader, optimizer, epoch, False)
            model.test_step(device, test_loader)

        # using validation set in Fisher Information Matrix computation as specified by:
        #   https://github.com/ariseff/overcoming-catastrophic/blob/master/experiment.ipynb
            model.compute_fisher(validation_loader)

        for epoch in range(1, args.epochs + 1):
            model.train_step(args, device, train_loader, optimizer, epoch, True)
            model.test_step(device, test_loader)


if __name__ == '__main__':
    main()