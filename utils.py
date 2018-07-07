import torch
import os
import torch.nn as nn
import torch.utils.data as D
from torch.autograd import Variable
from torchvision import datasets, transforms
from ExpandableModel import ExpandableModel
from EWCModel import EWCModel
from NoRegModel import NoRegModel
from tensorboardX import SummaryWriter
import numpy as np


# generate the DataLoaders corresponding to a permuted mnist task
def generate_new_mnist_task(args, kwargs, first_task):

    # permutation to be applied to all images in the dataset (if this is not the first dataset being generated)
    pixel_permutation = torch.randperm(args.input_size)

    # transforms.Compose() composes several transforms together.
    #
    # IF this is NOT the FIRST task, we should permute the original MNIST dataset to form a new task.
    #
    #  The transforms composed here are as follows:
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
    #
    # transforms.Lambda() applies the enclosed lambda function to each image (x) in the DataLoader
    # todo comment on sequential mnist and pixel permuation
    # permutation from: https://discuss.pytorch.org/t/sequential-mnist/2108 (first response)
    transformations = transforms.Compose(
        [
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,)), # TODO determine why network performs better w/o normalization
            transforms.Lambda(lambda x: x.view(-1, 1))
        ]) if first_task else transforms.Compose(
        [
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,)), # TODO determine why network performs better w/o normalization
            transforms.Lambda(lambda x: x.view(-1, 1)[pixel_permutation])
        ])

    # Split the PyTorch MNIST training dataset into training and validation datasets, and transform the data.
    #
    # D.dataset.random_split(dataset, lengths):
    #   Randomly split a dataset into non-overlapping new datasets of given lengths
    #
    # datasets.MNIST():
    #   ARGUMENTS (in order):
    #   root (string) – Root directory of dataset where processed/training.pt and processed/test.pt exist.
    #   train (bool, optional) – If True, creates dataset from training.pt, otherwise from test.pt.
    #   transform (callable, optional) – A function/transform that takes in an PIL image and returns a transformed
    #                                       version. E.g, transforms.RandomCrop
    #   download (bool, optional) – If true, downloads the dataset from the internet and puts it in root directory.
    #                                       If dataset is already downloaded, it is not downloaded again.
    train_data, validation_data = \
        D.dataset.random_split(datasets.MNIST('../data', train=True, transform=transformations, download=True),
            [args.train_dataset_size, args.validation_dataset_size])

    # Testing dataset.
    # train=False, because we want to draw the data here from <root>/test.pt (as opposed to <root>/training.pt)
    test_data = datasets.MNIST('../data', train=False, transform=transformations, download=True)

    # A PyTorch DataLoader combines a dataset and a sampler, and returns single- or multi-process iterators over
    # the dataset.

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
    # 2) batch_size is the entire validation set in one batch
    # 3) shuffle=True ensures we are drawing random samples by shuffling the data each time we contstruct a new iterator
    #       over the data, and is implemented in the source code as a RandomSampler()
    # 4) kwargs defined above
    validation_loader = D.DataLoader(validation_data, batch_size=args.validation_dataset_size, shuffle=True, **kwargs)

    # Instantiate a DataLoader for the testing data in the same manner as above for training data, with two exceptions:
    #   Here, we use test_data rather than train_data, and we use test_batch_size
    test_loader = D.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    return train_loader, validation_loader, test_loader


# Generate and return a tuple representing the padding size to be used as an argument to torch.nn.functional.pad().
# Tuple format and more in-depth explanation of the effects of pad() are in documentation of the pad() method here:
# https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
def pad_tuple(smaller, larger):

    pads_required = []

    # loop over the dimensions of the tensor we are padding so that this method can be used with both 2D weights and
    # 1D biases
    for dim in range(len(list(smaller.size()))):

        # pad by the difference between the existing and desired sizes in the given dimension
        pads_required.append(list(larger.size())[dim] - list(smaller.size())[dim])

        # After following reversal, will result in NO zero padding to the left of a 1D tensor (only extend to the right) and NO zero padding on
        # the left or top of a 2D tensor (only extend to the right and down). For instance, if a 2D tensor is
        # quadrupling in size, the original values in the tensor will be in the upper-left quadrant, and the other
        # three quadrants will be padded with zeros.
        pads_required.append(0)

    # this will correct the order of the values in the resulting list to produce the desired output
    pads_required.reverse()

    return tuple(pads_required)


def output_tensorboard_graph(args, models, task_count):

    dummy_input = Variable(torch.rand(args.batch_size, args.input_size))

    for model in models:
        with SummaryWriter(comment='MODEL task count: {}, type: {}'.format(task_count, model.__class__.__name__)) as w:
            w.add_graph(model, (dummy_input,))

def expand(models, args):

    # output expansion notification to terminal
    text = "EXPANDING MODEL AND RETRAINING LAST TASK"
    banner = ""
    for char in range(len(text) + 2): banner += u"\u2588"
    banner += "\n" + u"\u2588" + text + u"\u2588" + "\n"
    for char in range(len(text) + 2): banner += u"\u2588"
    print("\033[93m\033[1m{}\033[0m".format(banner)) # print banner with bold warning text formatting

    expanded_models = []

    for model_num, model in enumerate(models):
        expanded_models.append(model.__class__.from_existing_model(model, model.hidden_size * args.scale_factor).to(model.device))

    return expanded_models


# generate the DataLoaders corresponding to a permuted mnist task
def generate_cifar_tasks(args, kwargs):

    train_loaders = []
    validation_loaders = []
    test_loaders = []

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])  # meanstd transformation

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Split the PyTorch MNIST training dataset into training and validation datasets, and transform the data.
    #
    # D.dataset.random_split(dataset, lengths):
    #   Randomly split a dataset into non-overlapping new datasets of given lengths
    #
    # datasets.MNIST():
    #   ARGUMENTS (in order):
    #   root (string) – Root directory of dataset where processed/training.pt and processed/test.pt exist.
    #   train (bool, optional) – If True, creates dataset from training.pt, otherwise from test.pt.
    #   transform (callable, optional) – A function/transform that takes in an PIL image and returns a transformed
    #                                       version. E.g, transforms.RandomCrop
    #   download (bool, optional) – If true, downloads the dataset from the internet and puts it in root directory.
    #                                       If dataset is already downloaded, it is not downloaded again.
    train_data = datasets.CIFAR100('../data', transform=transform_train, train=True, download=True)

    # Testing dataset.
    # train=False, because we want to draw the data here from <root>/test.pt (as opposed to <root>/training.pt)
    test_data = datasets.CIFAR100('../data', transform=transform_test, train=False, download=True)

    # A PyTorch DataLoader combines a dataset and a sampler, and returns single- or multi-process iterators over
    # the dataset.

    # DataLoader for the training data.
    # ARGUMENTS (in order):
    # dataset (Dataset) – dataset from which to load the data (train_data prepared in above statement, in this case).
    # batch_size (int, optional) – how many samples per batch to load (default: 1).
    # shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).
    # kwargs - see above definition
    train_loader = D.DataLoader(train_data, batch_size=1, shuffle=False, **kwargs)

    # Instantiate a DataLoader for the testing data in the same manner as above for training data, with two exceptions:
    #   Here, we use test_data rather than train_data, and we use test_batch_size
    test_loader = D.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    data_org_by_class = []

    for i in range(100):
        data_org_by_class.append([])

    for (data, target) in train_loader:
        data_org_by_class[target.item()].append((data, target))

    for sample in data_org_by_class[0]:
        print(sample[1])


    return train_loaders, validation_loaders, test_loaders
