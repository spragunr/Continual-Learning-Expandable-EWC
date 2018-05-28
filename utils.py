import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms
from copy import deepcopy
import matplotlib.pyplot as plt


def create_new_mnist_task(mnist_train_original, mnist_test_original, mnist_validation_original):

    #TODO seed NUMPY random number generator (different from PyTorch)

    # each MNIST image represented as 4D tensor- 1 x 1 x 28 x 28
    # this statement gets the height and width of a sample image in pixels (should both be 28)
    # and multiplies them to get the total number of pixels in the image - should be 784
    pixels_per_image = len(mnist_train_original[0][0][0]) * len(mnist_train_original[0][0][0][0])

    # generate numpy array containing 0 - 783 in random order - a permutation "mask" to be applied to each image
    # in the MNIST dataset
    mask = np.random.permutation(pixels_per_image)

    permuted_train = deepcopy(mnist_train_original)


    #TODO comment the steps of this loop
    # target label is stored in _, we don't want to alter the targets
    for image, _ in permuted_train:

        perm_image = (deepcopy(image))

        #in-place
        perm_image.resize((1, pixels_per_image))

        for pixel_index, pixel in enumerate(perm_image):
            perm_image[pixel_index] = image[mask[pixel_index]]

        image = perm_image

    fig = plt.figure()

    for i in range(3):
        sample = permuted_train
        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.axis('off')

        if i == 3:
            plt.show()
            break

