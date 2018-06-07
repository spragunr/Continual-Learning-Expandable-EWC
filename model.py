import numpy as np
import torch
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

        if self.ewc:
            self.lam = lam          # the value of lambda (fisher multiplier) to be used in EWC loss computation

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

            loglikelihood_grads = torch.autograd.grad(log_softmax(model(data))[0, class_index], model.parameters())

            for parameter in range(len(self.list_of_FIMs)):
                self.list_of_FIMs[parameter] += torch.pow(loglikelihood_grads[parameter], 2.0)

            if sample_number == num_samples - 1:
                break

        for parameter in range(len(self.list_of_FIMs)):
            self.list_of_FIMs[parameter] /= num_samples


    # NOTE: using parameter.data, so for autograd it is critical that we re-initilize the optimizer after calling this
    # method during the training process!!!
    def update_ewc_sums(self):

        if not hasattr(self, 'sum_Fx'):
            self.initialize_fisher_sums()

        for fisher_diagonal_index in range(len(self.sum_Fx)):
            torch.Tensor.add_(self.sum_Fx[fisher_diagonal_index], self.list_of_FIMs[fisher_diagonal_index])





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