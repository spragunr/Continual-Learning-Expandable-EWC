import torch
import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from copy import deepcopy



class Model(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, input_dropout_prob, input_size, output_size,
                 ewc, lam=0, task_fisher_diags ={}, task_post_training_weights={}):

        super().__init__()

        self.ewc = ewc # determines whether or not the model will use EWC

        self.lam = lam # the value of lambda (fisher multiplier) to be used in EWC loss computation, if EWC enabled

        self.task_fisher_diags = task_fisher_diags
        self.task_post_training_weights = task_post_training_weights

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

        softmax = nn.Softmax(dim=-1)

        log_softmax = nn.LogSoftmax(dim=-1)

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

    def ewc_loss_prev_tasks(self):
        loss_prev_tasks = 0

        for parameter_index, parameter in enumerate(self.parameters()):
            # NOTE: * operator is element-wise multiplication
            loss_prev_tasks += torch.sum(torch.pow(parameter, 2.0) * self.sum_Fx[parameter_index])
            loss_prev_tasks -= 2 * torch.sum(parameter * self.sum_Fx_Wx[parameter_index])
            loss_prev_tasks += torch.sum(self.sum_Fx_Wx_sq[parameter_index])

        return loss_prev_tasks * (self.lam / 2.0)

    def train_model(self, args, device, train_loader, epoch, task_number):
        # Set the module in "training mode"
        # This is necessary because some network layers behave differently when training vs testing.
        # Dropout, for example, is used to zero/mask certain weights during TRAINING to prevent overfitting.
        # However, during TESTING (e.g. model.eval()) we do not want this to happen.
        self.train()

        # Set the optimization algorithm for the model- in this case, Stochastic Gradient Descent with/without
        # momentum (depends on the value of args.momentum- default is 0.0, so no momentum by default).
        #
        # ARGUMENTS (in order):
        #     params (iterable) – iterable of parameters to optimize or dicts defining parameter groups
        #     lr (float) – learning rate
        #     momentum (float, optional) – momentum factor (default: 0)
        #
        # NOTE on params:
        #   model.parameters() returns an iterator over a list of the model parameters in the same order in
        #   which they appear in the network when traversed input -> output
        #   (e.g.
        #       [weights b/w input and first hidden layer,
        #        bias b/w input and hidden layer 1,
        #        ... ,
        #        weights between last hidden layer and output,
        #        bias b/w hidden layer and output]
        #   )
        optimizer = optim.SGD(self.parameters(), lr=args.lr, momentum=args.momentum)

        # Enumerate will keep an automatic loop counter and store it in batch_idx.
        # The (data, target) pair returned by DataLoader train_loader each iteration consists
        # of an MNIST image data sample and an associated label classifying it as a digit 0-9.
        #
        # The image data for the batch is represented as a 4D torch tensor (see train_loader definition in main())
        # with dimensions (batch size, 1, 28, 28)- containing a normalized floating point value for the color of
        # each pixel in each image in the batch (MNIST images are 28 x 28 pixels).
        #
        # The target is represented as a torch tensor containing the digit classification labels for
        # the training data as follows:
        #       [ 3,  4,  2,  9,  7] represents ground truth labels for a 3, a 4, a 2, a 9, and a 7.
        # NOTE:
        # The indices are converted to one-hot label representations inside of the loss function:
        #       [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        #        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
        # represents labels for a 5 and a 2, because 1's are at index 5 and 2 in rows 0 and 1, respectively.
        #
        # SOURCE:
        # https://discuss.pytorch.org/t/why-does-the-minimal-pytorch-tutorial-not-have-mnist-images-be-onehot-for-logistic-regression/12562/6
        for batch_idx, (data, target) in enumerate(train_loader):

            # For some reason, the data needs to be wrapped in another tensor to work with our network,
            # otherwise it is not of the appropriate dimensions... I believe these two statements effectively add
            # a dimension.
            #
            # For an explanation of the meaning of these statements, see:
            #   https://stackoverflow.com/a/42482819/9454504
            #
            # This code was used here in another experiment:
            # https://github.com/kuc2477/pytorch-ewc/blob/4a75734ef091e91a83ce82cab8b272be61af3ab6/train.py#L35
            data_size = len(data)
            data = data.view(data_size, -1)

            # wrap data and target in variables- again, from the following experiment:
            #   https://github.com/kuc2477/pytorch-ewc/blob/4a75734ef091e91a83ce82cab8b272be61af3ab6/train.py#L50
            #
            # .to(device):
            #   set the device (CPU or GPU) to be used with data and target to device variable (defined in main())
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

            # Define the training loss function for the model to be cross entropy loss based on predicted values
            # and ground truth labels. This loss function only takes into account loss on the most recent task.
            #
            # NOTE: torch.nn.CrossEntropyLoss combines torch.nn.LogSoftmax() and torch.nn.NLLLoss() in one single class.
            criterion = nn.CrossEntropyLoss()

            # apply the loss function to the predictions/labels for this batch to compute loss
            loss = criterion(output, target)

            # if the model is using EWC, the summed loss term from the EWC equation must be calculated and
            # added to the loss that will be minimized by the optimizer.
            #
            # See equation (3) at:
            #   https://arxiv.org/pdf/1612.00796.pdf#section.2
            if self.ewc and task_number > 1:
                #loss += self.ewc_loss_prev_tasks()
                loss += self.alternative_ewc_loss(task_number)
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
            # print a message indicating progress AND which network (model) is reporting values.
            if batch_idx % args.log_interval == 0:
                print('{} Task: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    'EWC' if self.ewc else 'SGD + DROPOUT', task_number,
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    loss.item()))


    # try defining loss based on all extant Fisher diagonals and previous task weights (in lists in main.py)
    def alternative_ewc_loss(self, task_count):
        loss_prev_tasks = 0

        for task in range(1, task_count):

            task_weights = self.task_post_training_weights.get(task)
            task_fisher = self.task_fisher_diags.get(task)

            for param_index, parameter in enumerate(self.parameters()):
                task_weights_size = torch.Tensor(list(task_weights[param_index].size()))
                task_fisher_size = torch.Tensor(list(task_fisher[param_index].size()))

                parameter_size = torch.Tensor(list(parameter.size()))

                if not torch.equal(task_weights_size, parameter_size):
                    pad_tuple = pad_tuple(task_weights[param_index], parameter)
                    task_weights[param_index] = nn.functional.pad(task_weights[param_index], pad_tuple, mode='constant', value=0)

                if not torch.equal(task_fisher_size, parameter_size):
                    pad_tuple = pad_tuple(task_fisher[param_index], parameter)
                    task_fisher[param_index] = nn.functional.pad(task_fisher[param_index], pad_tuple, mode='constant', value=0)

                loss_prev_tasks += (((parameter - task_weights[param_index]) ** 2) * task_fisher[param_index]).sum()

        return loss_prev_tasks * (self.lam / 2.0)


