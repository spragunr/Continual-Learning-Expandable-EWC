import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms


def create_new_mnist_task(mnist_train_original, mnist_test_original, mnist_validation_original):

    # each MNIST image represented as 4D tensor- 1 x 1 x 28 x 28
    # this statement gets the height and width of a sample image in pixels (should both be 28)
    # and multiplies them to get the total number of pixels in the image - should be 784
    pixels_per_image = len(mnist_train_original[0][0][0]) * len(mnist_train_original[0][0][0][0])

    # generate numpy array containing 0 - 783 in sequential order
    pixels = np.array(range(pixels_per_image))

    # generate a permutation "mask" to be applied to each image by shuffling the order of the numbers in the pixels array
    np.random.shuffle(pixels)

