import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from functools import reduce


class Model(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, input_dropout_prob, input_size, output_size,
                 ewc, lam=0):

        super().__init__()

        self.ewc = ewc # determines whether or not the model will use EWC
        self.lam = lam # the value of lambda (fisher multiplier) to be used in EWC loss computation, if EWC used

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
    def compute_fisher_prob_dist(self, device, validation_loader):

        self.list_of_FIMs = []

        for parameter in self.parameters():
            self.list_of_FIMs.append(torch.zeros(tuple(parameter.size())))

        softmax = nn.Softmax()

        log_softmax = nn.LogSoftmax()

        probs = softmax(self.y)

        class_index = (torch.multinomial(probs, 1)[0][0]).item()

        # for every data sample in the validation set (default 1024)...
        for data, target in validation_loader:
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
            data, target = Variable(data).to(device), Variable(target).to(device)

            loglikelihood_grads = torch.autograd.grad(log_softmax(self(data))[0, class_index], self.parameters())

            for parameter in range(len(self.list_of_FIMs)):
                self.list_of_FIMs[parameter] += torch.pow(loglikelihood_grads[parameter], 2.0)


        for parameter in range(len(self.list_of_FIMs)):
            self.list_of_FIMs[parameter] /= 1024

    def save_theta_stars(self):

        # list of tensors used for saving optimal weights after most recent task training
        self.theta_stars = []

        # get the current values of each model parameter as tensors and add them to the list
        # self.theta_stars
        for parameter in self.parameters():
            self.theta_stars.append(parameter.data.clone())


    def calculate_ewc_loss_prev_tasks(self):

        losses = []

        for parameter_index, parameter in enumerate(self.parameters()):

            theta_star = Variable(self.theta_stars[parameter_index])
            fisher = Variable(self.list_of_FIMs[parameter_index])

            losses.append((fisher * (parameter - theta_star) ** 2).sum())

        return (self.lam / 2.0) * sum(losses)



