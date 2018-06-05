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






