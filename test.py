import numpy as np
import torch


# -*- coding: utf-8 -*-
import torch

# TODO comment
expansion_count = 0

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(size_average=False)

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-4

# TODO explain this
params = filter(lambda p: p.requires_grad, model.parameters())

optimizer = torch.optim.Adam(params, lr=learning_rate)

for t in range(500):

    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

    if t == 250:
        """
        count = 0
        for p in model.parameters():
            count += 1

        new_parameters= []

        for param_index, parameter in enumerate(model.parameters()):
            print(parameter.size())
            old_size = np.array(list(parameter.size()))

            # TODO These are flipped, need to clarify...
            if param_index == 0:
                new_size = old_size * np.array([2,1])
            elif param_index == count - 2:
                new_size = old_size * np.array([1,2])
            elif param_index == count - 1:
                new_size = old_size
            else:
                new_size = old_size * 2

            print(new_size)

            # TODO may need to flip this...
            new_param = torch.zeros(tuple(new_size))

            old_values = parameter.data.clone()

            print(old_values)

            # the following statements use += because of in-place operations and their importance in PyTorch:
            # see: https://discuss.pytorch.org/t/what-is-the-recommended-way-to-re-assign-update-values-in-a-variable-or-tensor/6125

            #weights - 2 dims
            if len(old_size.shape) == 2:
                for row in range(len(old_values)):
                    for column in range(len(old_values[row])):
                        new_param.data[row][column] += old_values[row][column]

            else:
                #biases - one dim
                for value_index in range(len(old_values)):
                    new_param.data[value_index] += old_values[value_index]

            # don't keep updating this old parameter with autograd
            parameter.requires_grad = False

            new_param = torch.nn.Parameter(new_param, requires_grad=True)

            new_parameters.append(("{}_expansion_{}".format(param_index, expansion_count), new_param))

        for new_param_name, new_param_data in new_parameters:
            model.register_parameter(new_param_name, new_param_data)

        params = filter(lambda par: par.requires_grad, model.parameters())

        for param in params:
            print(param.size())

        optimizer = torch.optim.Adam(params, lr=learning_rate)
        """
        old_sizes = []
        old_values = []

        for param_index, parameter in enumerate(model.parameters()):
            old_sizes.append(np.array(list(parameter.size())))
            old_values.append(parameter.data.clone())

        model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H*2),
            torch.nn.ReLU(),
            torch.nn.Linear(H*2, D_out),
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # put into the expanded model the values that were previously in these parameters in the smaller model
        for param_index, parameter in enumerate(model.parameters()):
            # weights - 2 dims
            if len(old_sizes[param_index].shape) == 2:
                for row in range(len(old_values[param_index])):
                    for column in range(len(old_values[param_index][row])):
                        parameter.data[row][column] += old_values[param_index][row][column]

            else:
                # biases - one dim
                for value_index in range(len(old_values[param_index])):
                    parameter.data[value_index] += old_values[param_index][value_index]
