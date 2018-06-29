import torch
import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from copy import deepcopy



class ExpandableModel(nn.Module):

    def __init__(self, hidden_size, input_size, output_size):

        super().__init__()

        # dictionary, format:
        #  {task number: size of network parameters (weights) when the network was trained on the task}
        self.size_dictionary = {}

        # copy specified model hyperparameters into instance variables
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.initialize_module_list()

        self.apply(utils.init_weights)

    def forward(self, x):

        # pass the data through all layers of the network
        for module in self.modulelist:
            x = module(x)

        self.y = x

        return self.y

    def initialize_module_list(self):

        self.modulelist = nn.ModuleList()

        self.modulelist.append(nn.Linear(self.input_size, self.hidden_size))
        self.modulelist.append(nn.ReLU())
        self.modulelist.append(nn.Linear(self.hidden_size, self.output_size))

    def expand(self):

        old_weights = []

        for parameter in self.parameters():
            old_weights.append(parameter.data.clone())
            parameter.requires_grad = False
            parameter.detach()

        self.hidden_size *= 2
        self.initialize_module_list()

        for module in self.modulelist:
            if type(module) == nn.Linear:
                module.reset_parameters()

        self.apply(utils.init_weights)

        # copy weights from smaller, old model into proper locations in the new, expanded model
        utils.copy_weights_expanding(old_weights, self)

        if self.ewc:

            self.expand_ewc_sums()

    def update_size_dict(self, task_count):

        self.size_dictionary.update({task_count: self.hidden_size})

    def copy_weights_expanding(self):