import numpy as np
import torch
import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from functools import reduce
from copy import deepcopy



class Model(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, input_dropout_prob, input_size, output_size,
                 ewc, lam=0):

        super().__init__()

        self.ewc = ewc # determines whether or not the model will use EWC

        self.lam = lam # the value of lambda (fisher multiplier) to be used in EWC loss computation, if EWC enabled

        # copy specified model hyperparameters into instance variables
        self.input_size = input_size
        self.input_dropout_prob = input_dropout_prob
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.output_size = output_size

        self.fully_connected_input = nn.Linear(self.input_size, self.hidden_size)

        self.fully_connected_hidden = nn.Linear(self.hidden_size, self.hidden_size)

        self.fully_connected_output = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = self.fully_connected_input(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.input_dropout_prob, training=self.training)
        x = self.fully_connected_hidden(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.hidden_dropout_prob, training=self.training)
        x = self.fully_connected_output(x)

        self.y = x

        return self.y

    # compute fisher by randomly sampling from probability distribution of outputs rather than the activations
    # themselves
    def compute_fisher_prob_dist(self, device, validation_loader, num_samples):
        self.list_of_FIMs = []

        for parameter in self.parameters():
            self.list_of_FIMs.append(torch.zeros(tuple(parameter.size())))

        softmax = nn.Softmax()

        log_softmax = nn.LogSoftmax()

        probs = softmax(self.y)

        class_index = (torch.multinomial(probs, 1)[0][0]).item()

        for sample_number, (data, _) in enumerate(validation_loader):

            # For some reason, the data needs to be wrapped in another tensor to work with our network,
            # otherwise it is not of the appropriate dimensions... I believe this statement effectively adds
            # a dimension.
            #
            # For an explanation of the meaning of this statement, see:
            #   https://stackoverflow.com/a/42482819/9454504
            #
            # This code was used here in another experiment:
            # https://github.com/kuc2477/pytorch-ewc/blob/4a75734ef091e91a83ce82cab8b272be61af3ab6/model.py#L61
            data = data.view(validation_loader.batch_size, -1)

            # wrap data and target in variables- again, from the following experiment:
            #   https://github.com/kuc2477/pytorch-ewc/blob/4a75734ef091e91a83ce82cab8b272be61af3ab6/model.py#L62
            #
            # .to(device):
            # set the device (CPU or GPU) to be used with data and target to device variable (defined in main())
            data = Variable(data).to(device)

            loglikelihood_grads = torch.autograd.grad(log_softmax(self(data))[0, class_index], self.parameters())

            for parameter in range(len(self.list_of_FIMs)):
                self.list_of_FIMs[parameter] += torch.pow(loglikelihood_grads[parameter], 2.0)

            if sample_number == num_samples - 1:
                break

        for parameter in range(len(self.list_of_FIMs)):
            self.list_of_FIMs[parameter] /= num_samples


    # NOTE: using parameter.data, so for autograd it is critical that we re-initilize the optimizer after calling this
    # method during the training process!!! (we may already be doing this as long as we don't call it within the
    # train() method...
    def update_ewc_sums(self):

        current_weights = []  # list of the current weights in the network

        for parameter in self.parameters():
            current_weights.append(deepcopy(parameter.data.clone()))

        if not hasattr(self, 'sum_Fx'):
            self.initialize_fisher_sums()

        # in-place addition of the Fisher diagonal for each parameter to the existing sum_Fx
        for fisher_diagonal_index in range(len(self.sum_Fx)):

            self.sum_Fx[fisher_diagonal_index].add_(self.list_of_FIMs[fisher_diagonal_index])

        # add the element-wise multiplication of the fisher diagonal for each parameter and that parameter's current
        # weight values to the existing sum_Fx_Wx
        for fisher_diagonal_index in range(len(self.sum_Fx_Wx)):

            self.sum_Fx_Wx[fisher_diagonal_index] = torch.addcmul(
                self.sum_Fx_Wx[fisher_diagonal_index],
                self.list_of_FIMs[fisher_diagonal_index],
                current_weights[fisher_diagonal_index])

        # add the element-wise multiplication of the fisher diagonal for each parameter and the square of each of that
        # parameter's current weight values to the existing sum_Fx_Wx_sq
        for fisher_diagonal_index in range(len(self.sum_Fx_Wx_sq)):

            self.sum_Fx_Wx_sq[fisher_diagonal_index] = torch.addcmul(
                self.sum_Fx_Wx_sq[fisher_diagonal_index],
                self.list_of_FIMs[fisher_diagonal_index],
                torch.pow(current_weights[fisher_diagonal_index], 2.0))


    # helper method for initializing 0-filled tensors to hold sums used in calculation of ewc loss
    def initialize_fisher_sums(self):

        empty_sums = []

        for parameter in self.parameters():
            empty_sums.append(torch.zeros(tuple(parameter.size())))

        # the sum of each task's Fisher Information (list of Fisher diagonals for each parameter in the network,
        # and Fisher diagonals calculated for later tasks are summed with the fisher diagonal in the list at the
        # appropriate network parameter index)
        self.sum_Fx = deepcopy(empty_sums)

        # the sum of each task's Fisher Information multiplied by its respective post-training weights in the network
        self.sum_Fx_Wx = deepcopy(empty_sums)

        # the sum of each task's Fisher Information multiplied by the square of its respective post-training weights
        # in the network
        self.sum_Fx_Wx_sq = deepcopy(empty_sums)

    def expand_ewc_sums(self):

        ewc_sums = [self.sum_Fx, self.sum_Fx_Wx, self.sum_Fx_Wx_sq]

        for ewc_sum in ewc_sums:
            for parameter_index, parameter in enumerate(self.parameters()):
                sum_size = torch.Tensor(list(ewc_sum[parameter_index].size()))
                parameter_size = torch.Tensor(list(parameter.size()))

                # pad the sum tensor at the current parameter index of the given sum list with zeros so that it matches the size in
                # all dimensions of the corresponding parameter
                if not torch.equal(sum_size, parameter_size):
                    pad_tuple = utils.pad_tuple(ewc_sum[parameter_index],parameter)
                    ewc_sum[parameter_index] = F.pad(ewc_sum[parameter_index], pad_tuple, mode='constant', value=0)