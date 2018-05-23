import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as D
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.y = F.log_softmax(self.fc2.weight, dim=1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):

    # Set the module "model" in "training mode"
    # This is necessary because some network layers behave differently when training vs testing.
    # Dropout, for example, is used to zero/mask certain weights during TRAINING to prevent overfitting.
    # However, during TESTING (e.g. model.eval()) we do not want this to happen.
    model.train()

    # Enumerate will keep an automatic loop counter and store it in batch_idx.
    # The (data, target) pair returned by DataLoader train_loader each iteration consists
    # of an MNIST image data sample and an associated label classifying it as a digit 0-9.
    #
    # The image data for the batch represented as a 4D torch tensor (see train_loader definition in main())
    # with dimensions (batch size, 1, 28, 28)- containing a normalized floating point value for the color of
    # each pixel in each image in the batch (MNIST images are 28 x 28 pixels).
    #
    # The target is represented as a torch tensor containing the digit classification labels for
    # the training data as follows:
    #       [ 3,  4,  2,  9,  7] represents ground truth labels for a 3, a 4, a 2, a 9, and a 7.
    # NOTE:
    # This should be distinguished from the other common representation of such data in which the following:
    #       [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
    # represents labels for a 5 and a 2, because 1's are at index 5 and 2 in rows 0 and 1, respectively.
    # THIS IS NOT THE WAY THE DATA IS REPRESENTED IN THIS EXPERIMENT.
    for batch_idx, (data, target) in enumerate(train_loader):

        # set the device (CPU or GPU) to be used with data and target to device variable (defined in main())
        data, target = data.to(device), target.to(device)

        # Gradients are automatically accumulated- therefore, they need to be zeroed out before the next backward
        # pass through the network so that they are replaced by newly computed gradients at later training iterations,
        # rather than SUMMED with those future gradients. The reasoning behind this approach and the need to zero
        # gradients manually with each training minibatch is presented here in more detail:
        # https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/9
        #
        # From PyTorch examples:
        #   Before the backward pass, use the optimizer object to zero all of the
        #   gradients for the variables it will update (which are the learnable
        #   weights of the model). This is because by default, gradients are
        #   accumulated in buffers( i.e, not overwritten) whenever .backward()
        #   is called.
        optimizer.zero_grad()

        # forward pass: compute predicted output by passing data to the network
        # NOTE: we have overridden forward() in class Net above, so this will call model.forward()
        output = model(data)

        # Define the training loss function for the model to be negative log likelihood loss based on predicted values
        # and ground truth labels.
        # The addition of a log_softmax layer as the last layer of our network
        # produces log probabilities from softmax and allows us to use this loss function instead of cross entropy,
        # because torch.nn.CrossEntropyLoss combines torch.nn.LogSoftmax() and torch.nn.NLLLoss() in one single class.
        loss = F.nll_loss(output, target)

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Simplified abstraction provided by PyTorch which uses a single statement to update all model parameters
        # according to gradients and optimization function.
        # In the case of SGD (without momentum), essentially executes the following:
        #
        #       with torch.no_grad():
        #           for param in model.parameters():
        #               param -= learning_rate * param.grad
        optimizer.step()

        # Each time the batch index is a multiple of the specified progress display interval (args.log_interval),
        # print a message of the following format:
        #
        # Train Epoch: <epoch number> [data sample number/total samples (% progress)]    Loss: <current loss value>
        #
        # With example values, output looks as follows:
        #
        # Train Epoch: 1 [3200/60000 (5%)]	Loss: 2.259442
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):

    # Set the module "model" in "evaluation mode"
    # This is necessary because some network layers behave differently when training vs testing.
    # Dropout, for example, is used to zero/mask certain weights during TRAINING (e.g. model.train())
    # to prevent overfitting. However, during TESTING/EVALUATION we do not want this to happen.
    model.eval()

    # total testing loss over ALL test batches (sum)
    test_loss = 0

    correct = 0

    # Wrap in torch.no_grad() because weights have requires_grad=True (meaning pyTorch autograd knows to
    # automatically track history of computed gradients for those weights) but we don't need to track testing
    # in autograd - we are no longer training so gradients should no longer be altered/computed (only "used")
    # and therefore we don't need to track this.
    with torch.no_grad():
        # Each step of the iterator test_loader will return the following values:
        #
        # data: a 4D tensor of dimensions (test batch size, 1, 28, 28), representing the MNIST data
        # for each of the 28 x 28 = 784 pixels of each of the images in a given test batch
        #
        # target: a 1D tensor of dimension <test batch size> containing ground truth labels for each of the
        # images in the corresponding test batch in order (contained within the data variable)
        for data, target in test_loader:
            # set the device (CPU or GPU) to be used with data and target to device variable (defined in main())
            data, target = data.to(device), target.to(device)

            # Forward pass: compute predicted output by passing data to the model. Module objects
            # override the __call__ operator so you can call them like functions. When
            # doing so you pass a Tensor of input data to the Module and it produces
            # a Tensor of output data. We have overriden forward() above, so our forward() method will be called here.
            output = model(data)

            # Define the testing loss to be negative likelihood loss based on predicted values (output)
            # and ground truth labels (target), calculate the testing batch loss, and sum it with the total testing
            # loss over ALL test batches (contained within test_loss).
            #
            # NOTE: size_average = False:
            # By default, the losses are averaged over observations for each minibatch.
            # If size_average is False, the losses are summed for each minibatch. Default: True
            #
            # Here we use size_average = False because we want to SUM all testing batch losses and average those
            # at the end of testing (by dividing by total number of testing SAMPLES (not batches) to obtain an
            # average loss over all testing batches). Otherwise, if size_average == True, we would be getting average
            # loss for each testing batch and then would average those at the end to obtain average testing loss,
            # which could theoretically result in some comparative loss of accuracy in the calculation of the
            # final value.
            #
            # NOTE:
            # <some loss function>.item() gets the a scalar value held in the loss
            test_loss += F.nll_loss(output, target, size_average=False).item()

            # Get the index of the max log-probability for each of the samples in the testing batch.
            #
            # output is a 2D tensor of dimensions (test batch size, 10) containing network-predicted probabilities
            # that the testing input is an image of each class (digits 0-9, signified by the index of each probability
            # in the output tensor for a given test image). That is to say that in the second dimension of output
            # the classification probabilities might look like the following for a given image:
            #       [0.1, 0.1, 0.05, 0.05, 0.2, 0.4, 0.1, 0.0, 0.0, 0.0]
            # Because the sixth entry (index 5) contains the maximum value relative to all other indices, the network's
            # prediction is that this image belongs to the sixth class- and is therefore the digit 5.
            #
            # NOTE: torch.max() Returns the maximum value of EACH ROW of the input tensor in the given dimension dim.
            # The second return value is the index location of each maximum value found (argmax). This is why we use
            # the second return value as the value of the variable pred, because we want the index of the maximum
            # probability (not its value)- hence the [1] indexing at the end of the statement.
            #
            # ARGUMENTS:
            #
            # Using dimension 1 as the first argument allows us to get the index of the highest valued
            # column in each row of output, which practically translates to getting the maximum predicted class
            # probability for each sample.
            #
            # If keepdim is True, the output tensors are of the same size as input except in the dimension dim
            # (first argument- in this case 1) where they are of size 1 (because we calculated ONE maximum value per
            # row). Otherwise, dim is squeezed (see torch.squeeze()), resulting in the output tensors having 1
            # fewer dimension than input.
            pred = output.max(1, keepdim=True)[1]

            # Check if predictions are correct, and if so add one to the total number of correct predictions across the
            # entire testing set for each correct prediction.
            #
            # A prediction is correct if the index of the highest value in the
            # prediction output is the same as the index of the highest value in the label for that sample.
            #
            # For example (MNIST):
            #   prediction: [0.1, 0.1, 0.05, 0.05, 0.2, 0.4, 0.1, 0.0, 0.0, 0.0]
            #   label:      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            #
            #   This would be a correct prediction- the sixth entry (index 5) in each array holds the highest
            #   value
            #
            # tensor_X.view_as(other) returns a resulting version of tensor_X with the same size as other.size()
            #
            # torch.eq() -> element wise equality:
            # tensor_X.eq(tensor_Y) returns a tensor of the same size as tensor_X with 0's at every index for which
            # the entry at that index in tensor_X does not match the entry at that index in tensor_Y and 1's at every
            # index for which tensor_X and tensor_Y contain matching values
            #
            # .sum() sums every row of the tensor into a tensor holding a single value
            #
            # .item() gets the scalar value held in the sum tensor
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Divide the accumulated test loss across all testing batches by the total number of testing samples (in this
    # case, 10,000) to get the average loss for the entire test set.
    test_loss /= len(test_loader.dataset)

    # The overall accuracy of the model's predictions as a percent value is the count of its accurate predictions
    # divided by the number of predictions it made, all multiplied by 100
    accuracy = 100. * correct / len(test_loader.dataset)

    # For the complete test set, display the average loss and accuracy
    # e.g. Test set: Average loss: 0.2073, Accuracy: 9415/10000 (94%)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))


def compute_fisher(model, validation_loader, num_samples=200):

    # Fisher Information is defined as the variance in the score.

    # list of all Fisher Information Matrices for all parameters in the model
    list_of_FIMs = []

    # Initialize a series of empty (all 0's) Fisher Information Matrices for the most recently trained task (SGD alone,
    # no Elastic Weight Consolidation)- one for each trainable parameter of the model. Each FIM has the same dimensions
    # as the parameter (weight or bias) to which it corresponds. Append each of these matrices to the list list_of_FIMs.
    #
    # torch.size() returns tensor dimensions as another tensor, so need to cast return value to a list to use
    # as the dimensionality argument to np.zeros(), which will created a zero-filled matrix of the dimensions
    # specified in that list
    for parameter in model.parameters():
        list_of_FIMs.append(np.zeros(list(parameter.size())))

    # Sample a random class from the model's output after testing run on most recent task (trained using
    # SGD alone, with no Elastic Weight Consolidation)
    #
    # torch.multinomial(input, num_samples):
    # Returns a tensor where each row contains num_samples indices sampled from the multinomial probability
    # distribution located in the corresponding row of tensor input. Indices are ordered from left to right according
    # to when each was sampled (first samples are placed in first column).
    # If input is a vector, out is a vector of size num_samples.
    # If input is a matrix with m rows, out is an matrix of shape (m × num_samples).
    #
    # In this case, output is a matrix, so to take just one value we want to index into a single entry-
    # hence the [0][0] indexing to get the first entry. Also, because a tensor is returned, we use
    # .item() to get the tensor as a scalar after the indexing has occured. The int cast is to make sure
    # the scalar we got is not a float, but an integer which we can use as an INDEX later.
    #
    # NOTE:
    # Our model's output after training/testing has already had softmax activations applied and log computed
    # (see last line of Net.forward() - torch.nn.functional.log_softmax()), which are required for use with
    # the negative log likelihood loss function. Therefore, when sampling a random class from our output, we
    # do NOT need to apply softmax and log functions as in code in compute_fisher() at:
    # https://github.com/ariseff/overcoming-catastrophic/blob/master/model.py

    print(model.y.detach().numpy())

    # TODO comment all of this

    probs = torch.t(torch.exp(model.y))

    class_index = torch.multinomial(probs, 1)[0][0]

    print(class_index.numpy())

    class_index = np.random.randint(0, 10)


    for iteration in range(num_samples):
        # Get a single random image from the validation dataset (4D tensor- dimensions: torch.Size([1, 1, 28, 28])),
        # and store the image in data variable. We don't need the target (ground truth label), so we just put that
        # in variable _.
        #
        # The images selected are random because shuffle=True is used in the validation_loader
        # definition, which is implemented in the source code as a torch.utils.data.sampler.RandomSampler().
        # Based on the __init__ function in DataLoader and the __init__ function in DataLoaderIter, each time
        # the statement below executes, we should get a NEW iterator and RandomSampler() that draws the random samples
        # over which that iterator will iterate.
        # See source at: https://pytorch.org/docs/master/_modules/torch/utils/data/dataloader.html#DataLoader
        #
        # NOTE:
        # The reason that this statement is not used outside of the loop is that RandomSampler() samples without
        # replacement, so if num_samples were larger than the validation set size, we would run out of samples.
        data, _ = next(iter(validation_loader))

        ders = torch.autograd.grad(model(data)[0,class_index], model.parameters())

        for parameter in range(len(list_of_FIMs)):
            list_of_FIMs[parameter] += np.square(ders[parameter])

    # divide totals by number of samples
    for parameter in range(len(list_of_FIMs)):
        list_of_FIMs[parameter] /= num_samples






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

    # set a manual seed for random number generation
    torch.manual_seed(args.seed)

    # set the device on which to perform computations
    device = torch.device("cuda" if use_cuda else "cpu")

    # arguments specific to CUDA computation
    # num_workers: how many subprocesses to use for data loading - if set to 0, data will be loaded in the main process
    # pin_memory: if True, the DataLoader will copy tensors into CUDA pinned memory before returning them
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

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
    train_data, validation_data = D.dataset.random_split(datasets.MNIST('../data', train=True,
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

    # TODO comment this
    validation_loader = D.DataLoader(validation_data, batch_size=1, shuffle=True, **kwargs)

    # Instantiate a DataLoader for the testing data in the same manner as above for training data, with one exception:
    #       train=False, because we want to draw the data here from <root>/test.pt (as opposed to <root>/training.pt)
    test_loader = D.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Move all parameters and buffers in the module Net to device (CPU or GPU- set above).
    # Both integral and floating point values are moved.
    model = Net().to(device)

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

    # for each desired epoch, train and test the model
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        compute_fisher(model, validation_loader)


if __name__ == '__main__':
    main()
