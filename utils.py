import torch
import torchvision
import numpy as np
import torch.utils.data as D
from torchvision import datasets, transforms
from copy import deepcopy

# tested, works!
def apply_permutation(image, permutation):

    orig_shape = image.size()

    image = image.numpy()

    # TODO replace magic number
    # numpy version - in-place, make sure OK that not a tuple
    image.resize(784)

    perm_image = (deepcopy(image))

    for pixel_index, pixel in enumerate(perm_image):
        perm_image[pixel_index] = image[permutation[pixel_index]]

    image = torch.Tensor(perm_image)

    image.resize_(orig_shape)

    return image

def generate_new_mnist_task(train_dataset_size, validation_dataset_size, batch_size, test_batch_size, kwargs, first_task):

    # TODO note the "spread" rather than travel method- as used in ariseff, also get rid of magic number
    # generate numpy array containing 0 - 783 in random order - a permutation "mask" to be applied to each image
    # in the MNIST dataset
    permutation = np.random.permutation(784)

    transformations = \
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ] if first_task else \
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: apply_permutation(x, permutation))
        ]

    # TODO comment, also check if we need to alter the root for each new dataset
    train_data, validation_data = \
        D.dataset.random_split(datasets.MNIST('../data', train=True, transform=transforms.Compose(transformations)),
            [train_dataset_size, validation_dataset_size])

    test_data = \
        datasets.MNIST('../data', train=False, transform=transforms.Compose(transformations))

    train_loader = D.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)

    validation_loader = D.DataLoader(validation_data, batch_size=1, shuffle=True, **kwargs)

    test_loader = D.DataLoader(test_data, batch_size=test_batch_size, shuffle=True, **kwargs)

    return train_loader, validation_loader, test_loader

