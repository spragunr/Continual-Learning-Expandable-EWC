import argparse
import torch
import utils
import numpy as np
from model import Model
from copy import deepcopy
import plot
from torch.autograd import Variable
from tensorboardX import SummaryWriter



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
    parser.add_argument('--l2-reg-penalty', type=float, default=0.0, metavar='L2',
                        help='l2 regularization penalty (weight decay) (default: 0.0)')

    # This is the lambda (fisher multiplier) value used by:
    # https://github.com/ariseff/overcoming-catastrophic/blob/master/experiment.ipynb - see In [17]
    #
    # some other examples:
    # 400 (from https://arxiv.org/pdf/1612.00796.pdf#subsection.4.2)
    #  inverse of learning rate (1.0 / lr) (from https://github.com/stokesj/EWC)- see readme
    parser.add_argument('--lam', type=float, default=15, metavar='LAM',
                        help='ewc lambda value (fisher multiplier) (default: 15)')

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
    parser.add_argument('--train-dataset-size', type=int, default=47200, metavar='TDS',
                        help='how many images to put in the training dataset')

    # size of the validation set
    #
    # I got the value 1024 from:
    #    https://github.com/kuc2477/pytorch-ewc/blob/4a75734ef091e91a83ce82cab8b272be61af3ab6/main.py#L24
    parser.add_argument('--validation-dataset-size', type=int, default=12800, metavar='VDS',
                        help='how many images to put in the validation dataset')

    # the number of samples used in computation of
    # Fisher Information
    parser.add_argument('--fisher-num-samples', type=int, default=200)

    # size of hidden layer(s)
    parser.add_argument('--hidden-size', type=int, default=50)

    # number of hidden layers
    # TODO implement this - currently does not actually modify network structure...
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

    # A list of the different DataLoader objects that hold various permutations of the mnist testing dataset-
    # we keep these around in a persistent list here so that we can use them to test each of the models in the
    # list "models" after they are trained on the latest task's training dataset.
    # For more details, see: generate_new_mnist_task() in utils.py
    test_loaders = []

    # the number of the task on which we are CURRENTLY training in the loop below (as opposed to a count of the number
    # of tasks on which we have already trained) - e.g. when training on task 3 this value will be 3
    task_count = 1

    # dictionary, format {task number: size of network parameters (weights) when the network was trained on the task}
    model_size_dictionaries = []

    # initialize model size dictionaries
    for model in models:
        model_size_dictionaries.append({})

    dummy_input = Variable(torch.rand(args.batch_size, 784))

    for model in models:
        with SummaryWriter(comment='model ewc: {}'.format(model.ewc)) as w:
            w.add_graph(model, (dummy_input,))

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
        for model_num in range(len(models)):

            # for each desired epoch, train the model on the latest task
            for epoch in range(1, args.epochs + 1):
                models[model_num].train_model(args, device, train_loader, epoch, task_count)

            # update the model size dictionary
            model_size_dictionaries[model_num].update({task_count: models[model_num].hidden_size})

            # generate a dictionary mapping tasks to models of the sizes that the network was when those tasks were
            # trained, containing subsets of the weights currently in the model (to mask new, post-expansion weights
            # when testing on tasks for which the weights did not exist during training)
            test_models = utils.generate_model_dictionary(models[model_num], model_size_dictionaries[model_num])

            # test the model on ALL tasks trained thus far (including current task)
            utils.test(test_models, device, test_loaders)


            # If the model currently being used in the loop is using EWC, we need to compute the fisher information
            if models[model_num].ewc:

                # save the theta* ("theta star") values after training - for plotting and comparative loss calculations
                # using the method in model.alternative_ewc_loss()
                #
                # NOTE: when I reference theta*, I am referring to the values represented by that variable in
                # equation (3) at:
                #   https://arxiv.org/pdf/1612.00796.pdf#section.2
                current_weights = []

                for parameter in models[model_num].parameters():
                    current_weights.append(deepcopy(parameter.data.clone()))


                models[model_num].task_post_training_weights.update({task_count: deepcopy(current_weights)})

                # using validation set in Fisher Information Matrix computation as specified by:
                # https://github.com/ariseff/overcoming-catastrophic/blob/master/experiment.ipynb
                models[model_num].compute_fisher_prob_dist(device, validation_loader, args.fisher_num_samples)

                # update the ewc loss sums in the model to incorporate weights and fisher info from the task on which
                # we just trained the network
                models[model_num].update_ewc_sums()

                # store the current fisher diagonals for use with plotting and comparative loss calculations
                # using the method in model.alternative_ewc_loss()
                models[model_num].task_fisher_diags.update({task_count: deepcopy(models[model_num].list_of_fisher_diags)})



        # expand each of the models (SGD + DROPOUT and EWC) after task 2 training and before task 3 training...
        if task_count == 2:
            print("expanding...")
            for model_num in range(len(models)):
                models[model_num].expand()
                with SummaryWriter(comment='model ewc: {}'.format(models[model_num].ewc)) as w:
                        w.add_graph(models[model_num], (dummy_input,))



        # increment the number of the current task before re-entering while loop
        task_count += 1

if __name__ == '__main__':
    main()
