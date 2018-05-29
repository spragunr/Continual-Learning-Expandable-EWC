import argparse
import torch
import torch.optim as optim
import torch.utils.data as D
import numpy as np
import utils
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
    # TODO change this back to 10
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
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
    parser.add_argument('--hidden-size', type=int, default=400)
    parser.add_argument('--hidden-layer-num', type=int, default=2)
    parser.add_argument('--hidden-dropout-prob', type=float, default=.5)
    parser.add_argument('--input-dropout-prob', type=float, default=.2)

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

    # TODO ensure this carries over multiple modules- but I'm fairly confident it does...
    # set a manual seed for random number generation
    torch.manual_seed(args.seed)

    # TODO update this comment
    # Move all parameters and buffers in the module Net to device (CPU or GPU- set above).
    # Both integral and floating point values are moved.
    model = Model(args.hidden_size,
                  args.hidden_layer_num,
                  args.hidden_dropout_prob,
                  args.input_dropout_prob,
                  input_size=784,  # 28 x 28 = 784 pixels per image
                  output_size=10,  # 10 classes - digits 0-9
                  ).to(device)

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
    test_loaders = []

    task_count = 1

    # TODO comment each step in this loop
    # keep learning tasks ad infinitum
    while(True):
        train_loader, validation_loader, test_loader = utils.generate_new_mnist_task(
            args.train_dataset_size,
            args.validation_dataset_size,
            args.batch_size,
            args.test_batch_size,
            kwargs,
            first_task=(task_count == 1))

        test_loaders.append(test_loader)

        # for each desired epoch, train and test the model with no ewc
        for epoch in range(1, args.epochs + 1):
            model.train_step(args, device, train_loader, optimizer, epoch, task_count, ewc=False)
            model.test_step(device, test_loaders, ewc=False)

        if task_count > 1:
            # now with ewc
            for epoch in range(1, args.epochs + 1):
                model.train_step(args, device, train_loader, optimizer, epoch, task_count, ewc=True)
                model.test_step(device, test_loaders, ewc=True)

        # using validation set in Fisher Information Matrix computation as specified by:
        #   https://github.com/ariseff/overcoming-catastrophic/blob/master/experiment.ipynb
        model.compute_fisher(validation_loader)

        model.save_optimal_weights()

        task_count += 1

if __name__ == '__main__':
    main()