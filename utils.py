import torch
import torchvision
import numpy as np
import torch.utils.data as D
from torchvision import datasets, transforms
from copy import deepcopy
from model import Model

# tested, works!
def apply_permutation(image, permutation):

    orig_shape = image.size()

    image = image.numpy()

    image.resize(784)

    perm_image = (deepcopy(image))

    for pixel_index, pixel in enumerate(perm_image):
        perm_image[pixel_index] = image[permutation[pixel_index]]

    image = torch.Tensor(perm_image)

    image.resize_(orig_shape)

    return image

def generate_new_mnist_task(train_dataset_size, validation_dataset_size, batch_size, test_batch_size, kwargs, first_task):

    # Note that, as in experiment from github/ariseff, these are permutations of the ORIGINAL dataset
    #
    # Generate numpy array containing 0 - 783 in random order - a permutation "mask" to be applied to each image
    # in the MNIST dataset
    permutation = np.random.permutation(784)

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
    transformations = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]) if first_task else transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: apply_permutation(x, permutation))
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
            [train_dataset_size, validation_dataset_size])

    # Testing dataset.
    # train=False, because we want to draw the data here from <root>/test.pt (as opposed to <root>/training.pt)
    test_data = \
        datasets.MNIST('../data', train=False, transform=transformations, download=True)

    # A PyTorch DataLoader combines a dataset and a sampler, and returns single- or multi-process iterators over
    # the dataset.

    # DataLoader for the training data.
    # ARGUMENTS (in order):
    # dataset (Dataset) – dataset from which to load the data (train_data prepared in above statement, in this case).
    # batch_size (int, optional) – how many samples per batch to load (default: 1).
    # shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).
    # kwargs - see above definition
    train_loader = D.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)

    # Dataloader for the validation dataset-
    # ARGUMENTS (in order):
    # 1) validation_data as the dataset
    # 2) batch_size is same as that provided for the training dataset
    # 3) shuffle=True ensures we are drawing random samples by shuffling the data each time we contstruct a new iterator
    #       over the data, and is implemented in the source code as a RandomSampler() - see comments in compute_fisher
    #       for more details and a link to the source code
    # 4) kwargs defined above
    validation_loader = D.DataLoader(validation_data, batch_size=batch_size, shuffle=True, **kwargs)

    # Instantiate a DataLoader for the testing data in the same manner as above for training data, with two exceptions:
    #   Here, we use test_data rather than train_data, and we use test_batch_size
    test_loader = D.DataLoader(test_data, batch_size=test_batch_size, shuffle=True, **kwargs)

    return train_loader, validation_loader, test_loader


def copy_weights(old_model, expanded_model):

    old_sizes = []
    old_weights = []

    # save data from old model
    for param_index, parameter in enumerate(old_model.parameters()):
        old_sizes.append(np.array(list(parameter.size())))
        old_weights.append(parameter.data.clone())

    # transfer that data to the expanded model
    for param_index, parameter in enumerate(expanded_model.parameters()):

        # weights - 2 dims
        if list(old_sizes[param_index].shape)[0] == 2:

            for row in range(len(old_weights[param_index])):

                for column in range(len(old_weights[param_index][row])):

                    # todo does this need to be in-place?
                    parameter.data[row][column] = old_weights[param_index][row][column]

        else:

            # biases - one dim
            for value_index in range(len(old_weights[param_index])):

                # todo does this need to be in-place?
                parameter.data[value_index] = old_weights[param_index][value_index]


def expand_model(model):

    expanded_model = Model(
        model.hidden_size * 2,
        model.hidden_dropout_prob,
        model.input_dropout_prob,
        model.input_size,
        model.output_size,
        model.ewc,
        model.lam
    )

    copy_weights(model, expanded_model)

    return expanded_model
