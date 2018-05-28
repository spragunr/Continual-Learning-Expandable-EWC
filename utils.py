import torch
import torchvision
import numpy as np
import torch.utils.data as data
from torchvision import datasets, transforms
from copy import deepcopy
import matplotlib.pyplot as plt


def create_new_mnist_task(mnist_train_original, mnist_test_original, mnist_validation_original):

    #TODO seed NUMPY random number generator (different from PyTorch)

    # TODO UPDATE ALL COMMENTS TO REFLECT THE FACT THAT these are image/label combos(??)

    # each MNIST image/ label combo represented as 4D tensor- 1 x 1 x 28 x 28
    # this statement gets the height and width of a sample image in pixels (should both be 28)
    # and multiplies them to get the total number of pixels in the image - should be 784
    pixels_per_image = len(mnist_train_original[0][0][0]) * len(mnist_train_original[0][0][0][0])

    # generate numpy array containing 0 - 783 in random order - a permutation "mask" to be applied to each image
    # in the MNIST dataset
    mask = np.random.permutation(pixels_per_image)

    permuted_train_list = []

    #TODO comment the steps of this loop
    # target label is stored in _, we don't want to alter the targets

    print(mnist_train_original[0])
    for image, label in mnist_train_original:

        orig_shape = image.size()

        image = image.numpy()

        # numpy version - in-place
        image.resize((pixels_per_image))

        perm_image = (deepcopy(image))

        for pixel_index, pixel in enumerate(perm_image):
            perm_image[pixel_index] = image[mask[pixel_index]]

        image = torch.Tensor(perm_image)

        image.resize_(orig_shape)

        permuted_train_list.append((image, label))

    permuted_train_data = data.TensorDataset(permuted_train_list[0])

    sample, _ = permuted_train_data[0]

    print(_)

    plt.imshow(sample.numpy()[0])

    plt.show()


# trying this out to see if I can use PyTorch lambda transpose to simplify mnist task generation
def apply_permutation(image, permutation):
    plt.imshow(image.numpy()[0])
    plt.show()

    orig_shape = image.size()

    image = image.numpy()

    # numpy version - in-place
    image.resize((784))

    perm_image = (deepcopy(image))

    for pixel_index, pixel in enumerate(perm_image):
        perm_image[pixel_index] = image[permutation[pixel_index]]

    image = torch.Tensor(perm_image)

    image.resize_(orig_shape)

    return image
