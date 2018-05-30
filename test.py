import numpy as np
import torch


# -*- coding: utf-8 -*-
import torch

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
params = filter(lambda p : p.requires_grad, model.parameters())


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
        count = 0
        for p in model.parameters():
            count += 1

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

            for row in range(len(old_values)):
                for column in range(len(old_values[row])):
                    new_param.data[row][column] = old_values[row][column]

            # don't keep updating this old parameter with autograd
            parameter.requires_grad = False

            # TODO may need to do this OUTSIDE of loop
            # TODO check if names need to vary - THEY DO
            model.register_parameter("{}".format(param_index), new_param)





@torch.enable_grad()
def param_setter():
    return 0



for parameter in model.parameters():
        parameter.data = torch.ones(list(parameter.data.size()))
        torch.autograd.set_grad_enabled(True)
for parameter in model.parameters():
    print(parameter.data)


