import torch.nn as nn

"""
DON'T DO THIS...
def generate_parameter_dictionaries(model):

    dictionaries = []

    #learning_rates = [1.0, 0.1, 0.01, 0.001, 0]
    learning_rates = [0.1]

    for lr in learning_rates:
        for parameter in model.parameters():

            # biases
            if len(list(parameter.size())) == 1:
                for weight in range(list(parameter.size())[0]):
                    dictionaries.append({'params': nn.Parameter(parameter[weight]), 'lr': lr})

            # weights
            else:
                for row in range(list(parameter.size())[0]):
                    for col in range(list(parameter.size())[1]):
                        dictionaries.append({'params': nn.Parameter(parameter[row][col]), 'lr': lr})

    return dictionaries
"""
