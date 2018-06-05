import argparse
import torch
import utils
import numpy as np
from model import Model

def main():
    # Command Line args
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')

    # 1 is just for speed when testing - the original EWC paper hyperparameters are here:
    # https://arxiv.org/pdf/1612.00796.pdf#section.4
    # This experiment uses 100 epochs:
    # https://github.com/stokesj/EWC
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')

    # This learning rate is the same as the one used by:
    # https://github.com/ariseff/overcoming-catastrophic/blob/afea2d3c9f926d4168cc51d56f1e9a92989d7af0/model.py#L114
    #
    # The original EWC paper hyperparameters are here:
    # https://arxiv.org/pdf/1612.00796.pdf#section.4
    parser.add_argument('--lr', type=float, default= 0.1, metavar='LR',
                        help='learning rate (default: 0.1)')

    # We don't want an L2 regularization penalty because https://arxiv.org/pdf/1612.00796.pdf#subsection.2.1
    # (see figure 2A) shows that this would prevent the network from learning another task.
    #
    # NOTE: Interestingly, this experiment DOES use an L2 regularization penalty (weight decay) and I honestly do not
    # know why: https://github.com/kuc2477/pytorch-ewc/blob/4a75734ef091e91a83ce82cab8b272be61af3ab6/main.py#L21
    parser.add_argument('--l2-reg-penalty', type=float, default=0.0, metavar='L2',
                        help='l2 regularization penalty (weight decay) (default: 0.0)')

    # This is the lambda (fisher multiplier) value used by:
    # https://github.com/kuc2477/pytorch-ewc/blob/4a75734ef091e91a83ce82cab8b272be61af3ab6/main.py#L19
    #
    # Empirically, other lambda values appeared to be too small to give EWC an edge over SGD w/ Dropout-
    # I tried the following:
    # 400 (from https://arxiv.org/pdf/1612.00796.pdf#subsection.4.2)
    # 15 (from https://github.com/ariseff/overcoming-catastrophic/blob/master/experiment.ipynb) - see In [17]
    # inverse of learning rate (1.0 / lr) (from https://github.com/stokesj/EWC)- see readme
    parser.add_argument('--lam', type=float, default=15, metavar='LR',
                        help='ewc lambda value (fisher multiplier) (default: 5e+3)')

    # only necessary if optimizer SGD with momentum is desired, hence default is 0.0
    parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                        help='SGD momentum (default: 0.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed-torch', type=int, default=1, metavar='ST',
                        help='random seed for PyTorch (default: 1)')
    parser.add_argument('--seed-numpy', type=int, default=1, metavar='SN',
                        help='random seed for numpy (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    # since validation set, which is drawn from training set, is size 1024, the rest of the data from the training set
    # are used as the actual data on which the network is trained: 60000 - 1024 = 58976
    parser.add_argument('--train-dataset-size', type=int, default=58976, metavar='TDS',
                        help='how many images to put in the training dataset')

    # size of the validation set
    #
    # I got the value 1024 from:
    #    https://github.com/kuc2477/pytorch-ewc/blob/4a75734ef091e91a83ce82cab8b272be61af3ab6/main.py#L24
    parser.add_argument('--validation-dataset-size', type=int, default=1024, metavar='VDS',
                        help='how many images to put in the validation dataset')

    # the number of samples used in computation of
    # Fisher Information
    parser.add_argument('--fisher-num-samples', type=int, default=200)

    # weights in each hidden layer
    parser.add_argument('--hidden-size', type=int, default=50)

    # number of hidden layers
    parser.add_argument('--hidden-layer-num', type=int, default=1)

    # Dropout probability for hidden layers - see:
    # https://arxiv.org/pdf/1612.00796.pdf#section.4
    parser.add_argument('--hidden-dropout-prob', type=float, default=.5)

    # Dropout probability for input layer - see:
    # https://arxiv.org/pdf/1612.00796.pdf#section.4
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

    # set a manual seed for PyTorch random number generation
    torch.manual_seed(args.seed_torch)

    # set a manual seed for numpy random number generation
    np.random.seed(args.seed_numpy)

    # Instantiate a model that will be trained using only SGD with dropout (no EWC).
    #
    # .to(device):
    #   Move all parameters and buffers in the module Net to device (CPU or GPU- set above).
    #   Both integral and floating point values are moved.
    sgd_dropout_model = Model(
                  args.hidden_size,
                  args.hidden_dropout_prob,
                  args.input_dropout_prob,
                  input_size=784, # 28 x 28 pixels = 784 pixels per MNIST image
                  output_size=10,  # 10 classes - digits 0-9
                  ewc=False # don't use EWC
                  ).to(device)

    # Instantiate a model that will be trained using EWC.
    #
    # .to(device):
    #   Move all parameters and buffers in the module Net to device (CPU or GPU- set above).
    #   Both integral and floating point values are moved.
    ewc_model = Model(
                  args.hidden_size,
                  args.hidden_dropout_prob,
                  args.input_dropout_prob,
                  input_size=784,  # 28 x 28 pixels = 784 pixels per MNIST image
                  output_size=10,  # 10 classes - digits 0-9
                  ewc=True, # use EWC
                  lam=args.lam # the lambda (fisher multiplier) value to be used in the EWC loss formula
                  ).to(device)

    # a list of the models we instantiated above
    models = [sgd_dropout_model, ewc_model]

    # TODO UPDATE ALL COMMENTS TO REFLECT THE FACT THAT there are image/label combos(??) (tuple)...

    # A list of the different DataLoader objects that hold various permutations of the mnist testing dataset-
    # we keep these around in a persistent list here so that we can use them to test each of the models in the
    # list "models" after they are trained on the latest task's training dataset.
    # For more details, see: generate_new_mnist_task() in utils.py
    test_loaders = []

    # the number of the task on which we are CURRENTLY training in the loop below (as opposed to a list of the number
    # of tasks on which we have already trained) - e.g. when training on task 3 this value will be 3
    task_count = 1

    # todo comment
    model_size_dictionaries = []

    for model in models:
        model_size_dictionaries.append({})

    # keep learning tasks ad infinitum
    while(True):

        # get the DataLoaders for the training, validation, and testing data
        train_loader, validation_loader, test_loader = utils.generate_new_mnist_task(
            args.train_dataset_size,
            args.validation_dataset_size,
            args.batch_size,
            args.test_batch_size,
            kwargs,
            first_task=(task_count == 1) # if first_task is True, we won't permute the MNIST dataset.
        )

        # add the new test_loader for this task to the list of testing dataset DataLoaders for later re-use
        # to evaluate how well the models retain accuracy on old tasks after learning new ones
        #
        # NOTE: this list also includes the current test_loader, which we are appending here, because we also
        # need to test each network on the current task after training
        test_loaders.append(test_loader)

        # for both SGD w/ Dropout and EWC models...
        for model_num, model in enumerate(models):

            # for each desired epoch, train the model on the latest task, and then test the model on ALL tasks
            # trained thus far (including current task)
            for epoch in range(1, args.epochs + 1):
                utils.train(model, args, device, train_loader, epoch, task_count)
                model_size_dictionaries[model_num].update({task_count:model.hidden_size})
                test_models = utils.generate_model_dictionary(model, model_size_dictionaries[model_num])
                utils.test(test_models, device, test_loaders)

                # If the model currently being used in the loop is using EWC, we need to compute the fisher information
                # and save the theta* ("theta star") values after training
                #
                # NOTE: when I reference theta*, I am referring to the values represented by that variable in
                # equation (3) at:
                #   https://arxiv.org/pdf/1612.00796.pdf#section.2
                if model.ewc:
                    # using validation set in Fisher Information Matrix computation as specified by:
                    #   https://github.com/ariseff/overcoming-catastrophic/blob/master/experiment.ipynb
                    utils.compute_fisher_prob_dist(model, device, validation_loader, args.fisher_num_samples)

                    # we are saving the theta star values for THIS task, which will be used in the fisher matrix
                    # computations for the NEXT task.
                    utils.save_theta_stars(model)

        # increment the number of the current task before re-entering while loop
        task_count += 1

if __name__ == '__main__':
    main()