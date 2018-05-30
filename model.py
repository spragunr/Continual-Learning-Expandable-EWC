import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from functools import reduce
from matplotlib import pyplot as plt


class Model(nn.Module):
    def __init__(self, hidden_size, hidden_layer_num, hidden_dropout_prob, input_dropout_prob, input_size, output_size, ewc, lam=0):
        super().__init__()

        self.ewc = ewc # determines whether or not the model will use ewc
        self.lam = lam

        self.input_size = input_size
        self.input_dropout_prob = input_dropout_prob
        self.hidden_size = hidden_size
        self.hidden_layer_num = hidden_layer_num
        self.hidden_dropout_prob = hidden_dropout_prob
        self.output_size = output_size

        # Layers
        self.layers = nn.ModuleList([
            # input
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.input_dropout_prob),
            # hidden
            *((nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
               nn.Dropout(self.hidden_dropout_prob)) * self.hidden_layer_num),
            # output
            nn.Linear(self.hidden_size, self.output_size)
        ])

        #TODO comment
        self.prev_tasks_ewc_loss = 0

    def forward(self, x):
        # TODO comment this and check if the last x is necessary... is it the initializer? I think yes.
        # TODO maybe change name of x in lambda to improve clarity
        # this essentially allows the flexibility of defining a forward() function that should work
        # regardless of network structure
        # reduce will apply the lambda function to all of the layers in the iterable self.layers,
        # with an initial value of x for the variable holding the result of the operation and self.layers
        # being used as a sequence of functions l() applied to x
        return reduce(lambda x, l: l(x), self.layers, x)

    def train_model(self, args, device, train_loader, epoch, task_number):
        # Set the module in "training mode"
        # This is necessary because some network layers behave differently when training vs testing.
        # Dropout, for example, is used to zero/mask certain weights during TRAINING to prevent overfitting.
        # However, during TESTING (e.g. model.eval()) we do not want this to happen.
        self.train()

        # Set the optimization algorithm for the model- in this case, Stochastic Gradient Descent with
        # momentum.
        #
        # ARGUMENTS (in order):
        #     params (iterable) – iterable of parameters to optimize or dicts defining parameter groups
        #     lr (float) – learning rate
        #     momentum (float, optional) – momentum factor (default: 0)
        #
        # NOTE on params:
        #   model.parameters() returns an iterator over a list of the trainable model parameters in the same order in
        #   which they appear in the network when traversed input -> output
        #   (e.g.
        #       [weights b/w input and first hidden layer,
        #        bias b/w input and hidden layer 1,
        #        ... ,
        #        weights between last hidden layer and output,
        #        bias b/w hidden layer and output]
        #   )
        optimizer = optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.l2_reg_penalty)

        # Enumerate will keep an automatic loop counter and store it in batch_idx.
        # The (data, target) pair returned by DataLoader train_loader each iteration consists
        # of an MNIST image data sample and an associated label classifying it as a digit 0-9.
        #
        # The image data for the batch represented as a 4D torch tensor (see train_loader definition in main())
        # with dimensions (batch size, 1, 28, 28)- containing a normalized floating point value for the color of
        # each pixel in each image in the batch (MNIST images are 28 x 28 pixels).
        #
        # The target is represented as a torch tensor containing the digit classification labels for
        # the training data as follows:
        #       [ 3,  4,  2,  9,  7] represents ground truth labels for a 3, a 4, a 2, a 9, and a 7.
        # NOTE:
        # This should be distinguished from the other common representation of such data in which the following:
        #       [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        #        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
        # represents labels for a 5 and a 2, because 1's are at index 5 and 2 in rows 0 and 1, respectively.
        # THIS IS NOT THE WAY THE DATA IS REPRESENTED IN THIS EXPERIMENT.
        for batch_idx, (data, target) in enumerate(train_loader):
            # TODO comment
            data_size = len(data)
            data = data.view(data_size, -1)

            # TODO update comment
            # set the device (CPU or GPU) to be used with data and target to device variable (defined in main())
            data, target = Variable(data).to(device), Variable(target).to(device)

            # Gradients are automatically accumulated- therefore, they need to be zeroed out before the next backward
            # pass through the network so that they are replaced by newly computed gradients at later training iterations,
            # rather than SUMMED with those future gradients. The reasoning behind this approach and the need to zero
            # gradients manually with each training minibatch is presented here in more detail:
            # https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/9
            #
            # From PyTorch examples:
            #   Before the backward pass, use the optimizer object to zero all of the
            #   gradients for the variables it will update (which are the learnable
            #   weights of the model). This is because by default, gradients are
            #   accumulated in buffers( i.e, not overwritten) whenever .backward()
            #   is called.
            optimizer.zero_grad()

            # forward pass: compute predicted output by passing data to the network
            # NOTE: we have overridden forward() in class Net above, so this will call model.forward()
            output = self(data)

            # TODO comment
            # Define the training loss function for the model to be negative log likelihood loss based on predicted values
            # and ground truth labels. This loss function only takes into account loss on the most recent task (no
            # regularization- SGD only).
            # The addition of a log_softmax layer as the last layer of our network
            # produces log probabilities from softmax and allows us to use this loss function instead of cross entropy,
            # because torch.nn.CrossEntropyLoss combines torch.nn.LogSoftmax() and torch.nn.NLLLoss() in one single class.
            criterion = nn.CrossEntropyLoss()

            loss = criterion(output, target)

            # TODO comment
            if self.ewc and task_number > 1:
                old_tasks_loss = self.calculate_ewc_loss_prev_tasks()
                loss += old_tasks_loss

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()

            # Simplified abstraction provided by PyTorch which uses a single statement to update all model parameters
            # according to gradients (with respect to the last loss function on which .backward() was called and
            # optimization function's update rule.
            # In the case of SGD (without momentum), essentially executes the following:
            #
            #       with torch.no_grad():
            #           for param in model.parameters():
            #               param -= learning_rate * param.grad
            optimizer.step()

            # Each time the batch index is a multiple of the specified progress display interval (args.log_interval),
            # print a message of the following format:
            #
            # Train Epoch: <epoch number> [data sample number/total samples (% progress)]    Loss: <current loss value>
            #
            # With example values, output looks as follows:
            #
            # Train Epoch: 1 [3200/60000 (5%)]	Loss: 2.259442
            if batch_idx % args.log_interval == 0:
                print('{} Task: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    'EWC' if self.ewc else 'SGD + DROPOUT', task_number,
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    loss.item()))

    def test_model(self, device, test_loaders):
        # Set the module in "evaluation mode"
        # This is necessary because some network layers behave differently when training vs testing.
        # Dropout, for example, is used to zero/mask certain weights during TRAINING (e.g. model.train())
        # to prevent overfitting. However, during TESTING/EVALUATION we do not want this to happen.
        self.eval()

        # TODO comment this
        for task_number, test_loader in enumerate(test_loaders):
            # total testing loss over ALL test batches (sum)
            test_loss = 0

            # total number of correct predictions over the entire testing set
            correct = 0

            # Wrap in torch.no_grad() because weights have requires_grad=True (meaning pyTorch autograd knows to
            # automatically track history of computed gradients for those weights) but we don't need to track testing
            # in autograd - we are no longer training so gradients should no longer be altered/computed (only "used")
            # and therefore we don't need to track this.
            with torch.no_grad():
                # Each step of the iterator test_loader will return the following values:
                #
                # data: a 4D tensor of dimensions (test batch size, 1, 28, 28), representing the MNIST data
                # for each of the 28 x 28 = 784 pixels of each of the images in a given test batch
                #
                # target: a 1D tensor of dimension <test batch size> containing ground truth labels for each of the
                # images in the corresponding test batch in order (contained within the data variable)
                for data, target in test_loader:
                    # TODO comment
                    data = data.view(test_loader.batch_size, -1)

                    # TODO update comment
                    # set the device (CPU or GPU) to be used with data and target to device variable (defined in main())
                    data, target = Variable(data).to(device), Variable(target).to(device)

                    # Forward pass: compute predicted output by passing data to the model. Module objects
                    # override the __call__ operator so you can call them like functions. When
                    # doing so you pass a Tensor of input data to the Module and it produces
                    # a Tensor of output data. We have overriden forward() above, so our forward() method will be called here.
                    output = self(data)

                    # TODO update comment
                    # Define the testing loss to be negative likelihood loss based on predicted values (output)
                    # and ground truth labels (target), calculate the testing batch loss, and sum it with the total testing
                    # loss over ALL test batches (contained within test_loss).
                    #
                    # NOTE: size_average = False:
                    # By default, the losses are averaged over observations for each minibatch.
                    # If size_average is False, the losses are summed for each minibatch. Default: True
                    #
                    # Here we use size_average = False because we want to SUM all testing batch losses and average those
                    # at the end of testing (by dividing by total number of testing SAMPLES (not batches) to obtain an
                    # average loss over all testing batches). Otherwise, if size_average == True, we would be getting average
                    # loss for each testing batch and then would average those at the end to obtain average testing loss,
                    # which could theoretically result in some comparative loss of accuracy in the calculation of the
                    # final value.
                    #
                    # NOTE:
                    # <some loss function>.item() gets the a scalar value held in the loss
                    criterion = nn.CrossEntropyLoss(size_average=False)
                    test_loss += criterion(output, target).item()

                    # Get the index of the max log-probability for each of the samples in the testing batch.
                    #
                    # output is a 2D tensor of dimensions (test batch size, 10) containing network-predicted probabilities
                    # that the testing input is an image of each class (digits 0-9, signified by the index of each probability
                    # in the output tensor for a given test image). That is to say that in the second dimension of output
                    # the classification probabilities might look like the following for a given image:
                    #       [0.1, 0.1, 0.05, 0.05, 0.2, 0.4, 0.1, 0.0, 0.0, 0.0]
                    # Because the sixth entry (index 5) contains the maximum value relative to all other indices, the network's
                    # prediction is that this image belongs to the sixth class- and is therefore the digit 5.
                    #
                    # NOTE: torch.max() Returns the maximum value of EACH ROW of the input tensor in the given dimension dim.
                    # The second return value is the index location of each maximum value found (argmax). This is why we use
                    # the second return value as the value of the variable pred, because we want the index of the maximum
                    # probability (not its value)- hence the [1] indexing at the end of the statement.
                    #
                    # ARGUMENTS:
                    #
                    # Using dimension 1 as the first argument allows us to get the index of the highest valued
                    # column in each row of output, which practically translates to getting the maximum predicted class
                    # probability for each sample.
                    #
                    # If keepdim is True, the output tensors are of the same size as input except in the dimension dim
                    # (first argument- in this case 1) where they are of size 1 (because we calculated ONE maximum value per
                    # row). Otherwise, dim is squeezed (see torch.squeeze()), resulting in the output tensors having 1
                    # fewer dimension than input.
                    pred = output.max(1, keepdim=True)[1]

                    # TODO alter this comment to account for the fact that targets are actual scalar index values in a tensor
                    # Check if predictions are correct, and if so add one to the total number of correct predictions across the
                    # entire testing set for each correct prediction.
                    #
                    # A prediction is correct if the index of the highest value in the
                    # prediction output is the same as the index of the highest value in the label for that sample.
                    #
                    # For example (MNIST):
                    #   prediction: [0.1, 0.1, 0.05, 0.05, 0.2, 0.4, 0.1, 0.0, 0.0, 0.0]
                    #   label:      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
                    #
                    #   This would be a correct prediction- the sixth entry (index 5) in each array holds the highest
                    #   value
                    #
                    # tensor_X.view_as(other) returns a resulting version of tensor_X with the same size as other.size()
                    #
                    # torch.eq() -> element wise equality:
                    # tensor_X.eq(tensor_Y) returns a tensor of the same size as tensor_X with 0's at every index for which
                    # the entry at that index in tensor_X does not match the entry at that index in tensor_Y and 1's at every
                    # index for which tensor_X and tensor_Y contain matching values
                    #
                    # .sum() sums every row of the tensor into a tensor holding a single value
                    #
                    # .item() gets the scalar value held in the sum tensor
                    correct += pred.eq(target.view_as(pred)).sum().item()

            # Divide the accumulated test loss across all testing batches by the total number of testing samples (in this
            # case, 10,000) to get the average loss for the entire test set.
            test_loss /= len(test_loader.dataset)

            # The overall accuracy of the model's predictions as a percent value is the count of its accurate predictions
            # divided by the number of predictions it made, all multiplied by 100
            accuracy = 100. * correct / len(test_loader.dataset)

            # TODO update comment
            # For the complete test set, display the average loss and accuracy
            # e.g. Test set: Average loss: 0.2073, Accuracy: 9415/10000 (94%)
            print('\n{} Test set {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                'EWC' if self.ewc else 'SGD + DROPOUT', task_number + 1, test_loss, correct, len(test_loader.dataset), accuracy))

    def compute_fisher(self, device, validation_loader, sample_size=1024):
        # TODO comment

        loglikelihoods = []

        for data, target in validation_loader:
            data = data.view(validation_loader.batch_size, -1)
            data, target = Variable(data).to(device), Variable(target).to(device)

            loglikelihoods.append(
                F.log_softmax(self(data))[range(validation_loader.batch_size), target.data]
            )

            if len(loglikelihoods) >= sample_size // validation_loader.batch_size:
                break

        # concatenate loglikelihood tensors in list loglikelihoods along 0th (default) dimension,
        # then calculate the mean of each row of the resulting tensor along the 0th dimension
        loglikelihood = torch.cat(loglikelihoods).mean(0)

        loglikelihood_grads = torch.autograd.grad(loglikelihood, self.parameters())

        self.fisher = []

        for grad in loglikelihood_grads:
            self.fisher.append(torch.pow(grad, 2.0))


    def save_theta_stars(self):

        # list of tensors used for saving optimal weights after most recent task training
        self.theta_stars = []

        # get the current values of each learnable model parameter as tensors of weights and add them to the list
        # optimal_weights
        for parameter in self.parameters():
            self.theta_stars.append(parameter.data.clone())


    def calculate_ewc_loss_prev_tasks(self):
        losses = []

        # TODO if this doesn't work, try using the dictionary with named parameters approach
        # (but ensure params all actually have different names)s
        for parameter_index, parameter in enumerate(self.parameters()):
            theta_star = Variable(self.theta_stars[parameter_index])
            fisher = Variable(self.fisher[parameter_index])

            losses.append((fisher * (parameter - theta_star) ** 2).sum())

        return (self.lam / 2.0) * sum(losses)



