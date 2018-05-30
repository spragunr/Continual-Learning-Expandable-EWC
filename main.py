import argparse
import torch

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
    parser.add_argument('--lr', type=float, default=1e-03, metavar='LR',
                        help='learning rate (default: 1e-03)')
    parser.add_argument('--l2-reg-penalty', type=float, default=0.0, metavar='L2',
                        help='l2 regularization penalty (weight decay) (default: 0.0)')
    parser.add_argument('--lam', type=float, default=5e+3, metavar='LR',
                        help='ewc lambda value (default: 5e+3)')
    # TODO maybe remove this to avoid confusion
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
    sgd_dropout_model = Model(args.hidden_size,
                  args.hidden_layer_num,
                  args.hidden_dropout_prob,
                  args.input_dropout_prob,
                  input_size=784, # TODO comment
                  output_size=10,  # 10 classes - digits 0-9
                  ewc=False
                  ).to(device)

    # TODO update this comment
    # Move all parameters and buffers in the module Net to device (CPU or GPU- set above).
    # Both integral and floating point values are moved.
    ewc_model = Model(args.hidden_size,
                  args.hidden_layer_num,
                  args.hidden_dropout_prob,
                  args.input_dropout_prob,
                  input_size=784,  # TODO comment
                  output_size=10,  # 10 classes - digits 0-9
                  ewc=True,
                  lam=args.lam
                  ).to(device)

    models = [sgd_dropout_model, ewc_model]

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

        for model in models:
            # for each desired epoch, train and test the model
            for epoch in range(1, args.epochs + 1):
                model.train_model(args, device, train_loader, epoch, task_count)
                model.test_model(device, test_loaders)

                if model.ewc:
                    # using validation set in Fisher Information Matrix computation as specified by:
                    #   https://github.com/ariseff/overcoming-catastrophic/blob/master/experiment.ipynb
                    # This needs to happen after training
                    model.compute_fisher(device, validation_loader)

                    # MUST BE DONE AFTER COMPUTE_FISHER - we are actually saving the theta star values for this task,
                    # which will be used in the fisher matrix computations for the next task.
                    model.save_theta_stars()

        task_count += 1

if __name__ == '__main__':
    main()