import torch.nn as nn
from copy import deepcopy

class ExpandableModel(nn.Module):

    def __init__(self, hidden_size, input_size, output_size, device):

        super().__init__()

        # dictionary, format:
        #  {task number: size of network parameters (weights) when the network was trained on the task}
        self.size_dictionary = {}

        # dictionary, format:
        # {task number : list of learnable parameter weight values after model trained on task}
        self.task_post_training_weights = {}

        # copy specified model hyperparameters into instance variables
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.device = device


    def forward(self, x):

        raise NotImplementedError("forward() is not implemented in ExpandableModel\n")

    @classmethod
    def from_existing_model(cls, m, new_hidden_size):

        raise NotImplementedError("from_existing_model() is not implemented in ExpandableModel\n")

    # kwargs contains validation_loader for EWC training (needed for Fisher estimation post-training)
    def train_model(self, args, train_loader, task_number, **kwargs):

        raise NotImplementedError("train_model() is not implemented in ExpandableModel\n")

    def initialize_module_list(self):

        if self.is_cifar:

            self.build_conv()

        else:

            self.build_mlp()

    def test(self, test_loaders, threshold, args):

        raise NotImplementedError("test() is not implemented in ExpandableModel\n")

    def update_size_dict(self, task_count):

        self.size_dictionary.update({task_count: self.hidden_size})

    def copy_weights_expanding(self, small_model):

        old_weights = []

        for parameter in small_model.parameters():
            old_weights.append(parameter.data.clone())
            parameter.requires_grad = False
            parameter.detach()

        for param_index, parameter in enumerate(self.parameters()):
            parameter.data[tuple(slice(0, n) for n in old_weights[param_index].shape)] = old_weights[param_index][...]

    def reset(self, task_count):
        old_weights = self.task_post_training_weights.get(task_count)

        for param_index, parameter in enumerate(self.parameters()):
            parameter.data[tuple(slice(0, n) for n in old_weights[param_index].shape)] = old_weights[param_index][...]

    def save_theta_stars(self, task_count):
        # save the theta* ("theta star") values after training - for plotting and comparative loss calculations
        # using the method in model.alternative_ewc_loss()
        #
        # NOTE: when I reference theta*, I am referring to the values represented by that variable in
        # equation (3) at:
        #   https://arxiv.org/pdf/1612.00796.pdf#section.2
        current_weights = []

        for parameter in self.parameters():
            current_weights.append(deepcopy(parameter.data.clone()))

        self.task_post_training_weights.update({task_count: deepcopy(current_weights)})

    # given a dictionary with task numbers as keys and model sizes (size of hidden layer(s) in the model when the model was
    # trained on a given task) as values, generate and return a dictionary correlating task numbers with model.Model
    # objects of the appropriate sizes, containing subsets of the weights currently in model
    def generate_model_dictionary(self):

        model_sizes = []

        # fetch all unique model sizes from the model size dictionary and store them in a list (model_sizes)
        for key in self.size_dictionary.keys():
            if not self.size_dictionary.get(key) in model_sizes:
                model_sizes.append(self.size_dictionary.get(key))

        models = []

        # make a model of each size specified in model_sizes, add them to models list
        for hidden_size in model_sizes:

            # make a model of the type corresponding to the model's direct superclass (CNN or MLP) for testing-
            # this way we don't need to pass lambda to the constructor, as it's not needed for testing
            test_model = self.__class__.__bases__[0](
                hidden_size,
                self.input_size,
                self.output_size,
                self.device
            ).to(self.device)

            # needed for restoration of output layer weights during testing
            test_model.task_post_training_weights = self.task_post_training_weights
            models.append(test_model)



        # copy weights from a larger to a smaller model - used when generating smaller models with subsets of current
        # model weights for testing the network on previous tasks...
        def copy_weights_shrinking(big_model, small_model):

            big_weights = []  # weights in parameters from the larger model

            # save data from big model
            for parameter in big_model.parameters():
                big_weights.append(parameter.data.clone())

            # transfer that data to the smaller model -
            # copy each weight from larger network that should still be in the smaller model to matching index
            # in the smaller network
            for param_index, parameter in enumerate(small_model.parameters()):
                parameter.data[...] = big_weights[param_index][tuple(slice(0, n) for n in list(parameter.size()))]

        # copy subsets of weights from the largest model to all other models
        for to_model in models:
            copy_weights_shrinking(self, to_model)

        model_dictionary = {}

        # build the model dictionary
        for model in models:
            for task_number in [k for k, v in self.size_dictionary.items() if v == model.hidden_size]:
                model_dictionary.update({task_number: model})

        return model_dictionary

