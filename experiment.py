"""
from model import Model
import torch
import torchvision
import torchvision.datasets as datasets


def test():
    # Check if predictions are correct- a prediction is correct if the index of the highest value in the prediction
    # output is the same as the index of the highest value in the label for that sample.
    # For example (MNIST):
    #   prediction: [0.1, 0.1, 0.05, 0.05, 0.2, 0.4, 0.1, 0.0, 0.0, 0.0]
    #   label:      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    #
    #   This would be a correct prediction- the sixth entry (index 5) in each array holds the highest
    #   value
    #
    # Using dimension 1 as an argument allows us to get the index of the highest valued column in each row,
    # which practically translates to getting the maximum predicted value for each sample and comparing that
    # value to the label for that sample. NOTE: torch.eq() returns 0 if args not equal, 1 if args equal
    correct_prediction = torch.eq(torch.argmax(y_pred, 1), torch.argmax(y_labels, 1))

    # The overall accuracy of the model's predictions is the sum of its accurate predictions (valued at 1)
    # divided by the number of predictions it made (incorrect predictions evaluate to 0)
    accuracy = torch.mean(correct_prediction.type(torch.DoubleTensor))


mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)

mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

BATCH_SIZE = 64

# Create random Tensors to hold inputs and outputs
model = Model(mnist_trainset.train_data, mnist_trainset.train_labels)

learning_rate = 1e-4
# parameters() returns an iterator over a list of the trainable model parameters in the same order in
# which they appear in the network when traversed input -> output
# (e.g.
#       [weights b/w input and first hidden layer,
#        bias b/w input and hidden layer 1,
#        ... ,
#        weights between last hidden layer and output,
#        bias b/w hidden layer and output]
# )
optimizer = torch.optim.Adam(model.model.parameters(), lr=learning_rate)

for i in range(500):
    # The output- predictions of classification likelihoods- is produced by running
    # the model on given input data.
    train_predictions = model.model(mnist_trainset.train_data)

    loss = model.loss(train_predictions, mnist_trainset.train_labels)

    print(i, loss.item())

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

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

    # DataLoader for the training data.
    # ARGUMENTS (in order):
    # dataset (Dataset) – dataset from which to load the data.
    # batch_size (int, optional) – how many samples per batch to load (default: 1).
    # shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).
    train_loader = torch.utils.data.DataLoader(
        # Use the MNIST dataset.
        # ARGUMENTS (in order):
        # root (string) – Root directory of dataset where processed/training.pt and processed/test.pt exist.
        # train (bool, optional) – If True, creates dataset from training.pt, otherwise from test.pt.
        # download (bool, optional) – If true, downloads the dataset from the internet and puts it in root directory.
        #                                       If dataset is already downloaded, it is not downloaded again.
        # transform (callable, optional) – A function/transform that takes in an PIL image and returns a transformed
        #                                       version. E.g, transforms.RandomCrop
        datasets.MNIST('../data', train=True, download=True,
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
            ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # Instantiate a DataLoader for the testing data in the same manner as above for training data, with one exception:
    # train=False, because we want to draw the data here from test.pt (as opposed to training.pt)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Move all parameters and buffers in the module Net to device (CPU or GPU- set above).
    # Both integral and floating point values are moved.
    model = Net().to(device)

    # Set the optimization algorithm for the model- in this case, standard Stochastic Gradient Descent with
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

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)


if __name__ == '__main__':
    main()