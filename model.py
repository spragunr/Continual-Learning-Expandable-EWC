import torch
import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from copy import deepcopy



class Model(nn.Module):

    # TODO remove the dropout probability parameters
    def __init__(self, hidden_size, hidden_dropout_prob, input_dropout_prob, input_size, output_size,
                 ewc, lam=0, task_fisher_diags ={}, task_post_training_weights={}):

        super().__init__()

        self.ewc = ewc # determines whether or not the model will use EWC

        self.lam = lam # the value of lambda (fisher multiplier) to be used in EWC loss computation, if EWC enabled

        # If this is the original model in the experiment, the next two instance variables are, by default,
        # instantiated as empty dictionaries. Otherwise, they will be copied from the parameters to the class
        # constructor. This is used to copy data from a pre-existing model to a new, expanded version of the model.

        # dictionary, format:
        # {task number : list of Fisher diagonals calculated after model trained on task}
        self.task_fisher_diags = task_fisher_diags

        # dictionary, format:
        # {task number : list of learnable parameter weight values after model trained on task}
        self.task_post_training_weights = task_post_training_weights

        # copy specified model hyperparameters into instance variables
        self.input_size = input_size
        self.input_dropout_prob = input_dropout_prob
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.output_size = output_size

        self.main = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size)
        )

        self.main.apply(utils.init_weights)

    def forward(self, x):

        # pass the data through all layers of the network
        self.y = self.main(x) # TODO consider this in relation to compute_fisher_prob_dist()

        return self.y

    # Compute fisher by randomly sampling from probability distribution of outputs rather than the activations
    # themselves. Replication of the sampling method used by:
    # https://github.com/ariseff/overcoming-catastrophic/blob/afea2d3c9f926d4168cc51d56f1e9a92989d7af0/model.py#L44
    def compute_fisher_prob_dist(self, device, validation_loader, num_samples):

        # List to hold the computed fisher diagonals for the task on which the network was just trained.
        # Fisher Information Matrix diagonals are stored as a list of tensors of the same dimensions and in the same
        # order as the parameters of the model given by model.parameters()
        self.list_of_fisher_diags = []

        # populate self.list_of_fisher_diags with tensors of zeros of the appropriate sizes
        for parameter in self.parameters():
            self.list_of_fisher_diags.append(torch.zeros(tuple(parameter.size())))

        # dim=-1 uses the last dimension. This is the tensorflow default, so meant to mimic the behavior in:
        # https://github.com/ariseff/overcoming-catastrophic/blob/afea2d3c9f926d4168cc51d56f1e9a92989d7af0/model.py#L53
        softmax = nn.Softmax(dim=-1)

        # log and softmax together- computing them sequentially as separate computations can be mathematically unstable
        # in edge-case scenarios
        log_softmax = nn.LogSoftmax(dim=-1)

        # get softmax activations from output layer (probabilities)
        probs = softmax(self.y)

        # sample a random class index from the softmax activations (.item() gets value in tensor as a scalar)
        class_index = (torch.multinomial(probs, 1)[0][0]).item()

        # sample_number is running count of samples (used to ensure sampling continues until num_samples reached)
        # data is an image
        # _ is the label for the image (not needed)
        for sample_number, (data, _) in enumerate(validation_loader):

            # The data needs to be wrapped in another tensor to work with our network,
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

            # gradients of parameters with respect to log likelihoods (log_softmax applied to output layer),
            # data for the sample from the validation set is sent through the network to mimic the behavior
            # of the feed_dict argument at:
            # https://github.com/ariseff/overcoming-catastrophic/blob/afea2d3c9f926d4168cc51d56f1e9a92989d7af0/model.py#L65
            loglikelihood_grads = torch.autograd.grad(log_softmax(self(data))[0, class_index], self.parameters())

            # square the gradients computed above and add each of them to the index in list_of_fisher_diags that
            # corresponds to the parameter for which the gradient was calculated
            for parameter in range(len(self.list_of_fisher_diags)):
                self.list_of_fisher_diags[parameter] += torch.pow(loglikelihood_grads[parameter], 2.0)

            # stop iterating through loop if num_samples reached
            if sample_number == num_samples - 1:
                break

        # divide totals by number of samples, getting average squared gradient values across num_samples as the
        # Fisher diagonal values
        for parameter in range(len(self.list_of_fisher_diags)):
            self.list_of_fisher_diags[parameter] /= num_samples



    # This method is used to update the summed error terms:
    # The sums are actually lists of sums, with one entry per model parameter in the shape of that model parameter
    #   sigma (Fisher_{task})
    #   sigma (Fisher_{task} * Weights_{task})
    #   sigma (Fisher_{task} * (Weights_{task}) ** 2)
    #
    #   NOTE: using parameter.data, so for pytorch autograd it is critical that we re-initilize the optimizer after calling this
    #   method during the training process! Otherwise gradient tracking may not work and training may be disrupted.
    #   We redefine the optimizer WITHIN the train method, so this is taken care of.
    def update_ewc_sums(self):

        current_weights = []  # list of the current weights in the network (one entry per parameter)

        # get deep copies of the values currently in the model parameters and append each of them to current_weights
        for parameter in self.parameters():
            current_weights.append(deepcopy(parameter.data.clone()))

        # if no summed terms yet, initialize them...
        if not hasattr(self, 'sum_Fx'):
            self.initialize_fisher_sums()

        # in-place addition of the Fisher diagonal for each parameter to the existing sum_Fx at corresponding
        # parameter index
        for fisher_diagonal_index in range(len(self.sum_Fx)):

            self.sum_Fx[fisher_diagonal_index].add_(self.list_of_fisher_diags[fisher_diagonal_index])

        # add the fisher diagonal for each parameter multiplied (element-wise) by that parameter's current weight values
        # to the existing sum_Fx_Wx entry at the corresponding parameter index
        for fisher_diagonal_index in range(len(self.sum_Fx_Wx)):

            self.sum_Fx_Wx[fisher_diagonal_index] = torch.addcmul(
                self.sum_Fx_Wx[fisher_diagonal_index],
                self.list_of_fisher_diags[fisher_diagonal_index],
                current_weights[fisher_diagonal_index])

        # add the fisher diagonal for each parameter multiplied (element-wise) by the square of that parameter's
        # current weight values to the existing sum_Fx_Wx_sq entry at the corresponding parameter index
        for fisher_diagonal_index in range(len(self.sum_Fx_Wx_sq)):

            self.sum_Fx_Wx_sq[fisher_diagonal_index] = torch.addcmul(
                self.sum_Fx_Wx_sq[fisher_diagonal_index],
                self.list_of_fisher_diags[fisher_diagonal_index],
                torch.pow(current_weights[fisher_diagonal_index], 2.0))


    # helper method for initializing 0-filled tensors to hold sums used in calculation of ewc loss
    def initialize_fisher_sums(self):

        empty_sums = []

        for parameter in self.parameters():
            empty_sums.append(torch.zeros(tuple(parameter.size())))

        # the sum of each task's Fisher Information (list of Fisher diagonals for each parameter in the network,
        # and Fisher diagonals calculated for later tasks are summed with the fisher diagonal in the list at the
        # appropriate parameter index)
        self.sum_Fx = deepcopy(empty_sums)

        # the sum of each task's Fisher Information multiplied by its respective post-training weights in the network
        # (list of entries- one per parameter- of same size as model parameters)
        self.sum_Fx_Wx = deepcopy(empty_sums)

        # the sum of each task's Fisher Information multiplied by the square of its respective post-training weights
        # in the network (list of entries- one per parameter- of same size as model parameters)
        self.sum_Fx_Wx_sq = deepcopy(empty_sums)

    # expand the sums used to compute ewc loss to fit an expanded model
    def expand_ewc_sums(self):

        ewc_sums = [self.sum_Fx, self.sum_Fx_Wx, self.sum_Fx_Wx_sq]

        for ewc_sum in ewc_sums:
            for parameter_index, parameter in enumerate(self.parameters()):
                # current size of entry at parameter_index in given list of sums
                sum_size = torch.Tensor(list(ewc_sum[parameter_index].size()))

                # current size of parameter in the model corresponding to the sum entry above
                parameter_size = torch.Tensor(list(parameter.size()))

                # pad the sum tensor at the current parameter index of the given sum list with zeros so that it matches the size in
                # all dimensions of the corresponding parameter
                if not torch.equal(sum_size, parameter_size):
                    pad_tuple = utils.pad_tuple(ewc_sum[parameter_index],parameter)
                    ewc_sum[parameter_index] = F.pad(ewc_sum[parameter_index], pad_tuple, mode='constant', value=0)

    # calculate the EWC loss on previous tasks only (not incorporating current task cross entropy)
    def ewc_loss_prev_tasks(self):

        loss_prev_tasks = 0

        # this computes the ewc loss on previous tasks via the algebraically manipulated fisher sums method:
        # (Weights_{current}) ** 2 * sigma (Fisher_{task}) - 2 * Weights_{current} * sigma (Fisher_{task} * Weights_{task}) +
        #          sigma (Fisher_{task} * (Weights_{task}) ** 2)
        #
        # for each parameter, we add to the loss the above loss term calculated for each weight in the parameter (summed)
        for parameter_index, parameter in enumerate(self.parameters()):
            # NOTE: * operator is element-wise multiplication
            loss_prev_tasks += torch.sum(torch.pow(parameter, 2.0) * self.sum_Fx[parameter_index])
            loss_prev_tasks -= 2 * torch.sum(parameter * self.sum_Fx_Wx[parameter_index])
            loss_prev_tasks += torch.sum(self.sum_Fx_Wx_sq[parameter_index])

        # mutliply error by fisher multiplier (lambda) divided by 2
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
        optimizer = optim.SGD(self.parameters(), lr=args.lr, momentum=args.momentum) # can use filter and requires_grad=False to freeze part of the network...

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

            # The data needs to be wrapped in another tensor to work with our network,
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

            # if the model is using EWC, the summed loss term from the EWC equation (loss on previuous tasks) must be calculated and
            # added to the loss that will be minimized by the optimizer.
            #
            # See equation (3) at:
            #   https://arxiv.org/pdf/1612.00796.pdf#section.2
            if self.ewc and task_number > 1:
                # This statement computed loss on previous tasks using the summed fisher terms as in ewc_loss_prev_tasks()
                loss += self.ewc_loss_prev_tasks()

                # Using the commented-out version statement below instead of the one above will calculate ewc loss
                # on previous tasks by multiplying the square of the difference between the current network
                # parameter weights and those after training each previously encountered task, multiplied by the
                # Fisher diagonal computed for the respective previous task in each difference, all summed together.

                #loss += self.alternative_ewc_loss(task_number)

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


    # Defines loss based on all extant Fisher diagonals and previous task weights
    def alternative_ewc_loss(self, task_count):

        loss_prev_tasks = 0

        # calculate ewc loss on previous tasks by multiplying the square of the difference between the current network
        # parameter weights and those after training each previously encountered task, multiplied by the
        # Fisher diagonal computed for the respective previous task in each difference, all summed together.
        for task in range(1, task_count):

            task_weights = self.task_post_training_weights.get(task) # weights after training network on task
            task_fisher = self.task_fisher_diags.get(task) # fisher diagonals computed for task

            for param_index, parameter in enumerate(self.parameters()):

                # size of weights at parameter_index stored after network was trained on the previous task in question
                task_weights_size = torch.Tensor(list(task_weights[param_index].size()))

                # size of the computed fisher diagonal for the parameter in question, for the given task (in outer for loop)
                task_fisher_size = torch.Tensor(list(task_fisher[param_index].size()))

                # current size of parameter in network corresponding to the weights and fisher info above
                parameter_size = torch.Tensor(list(parameter.size()))

                # If size of tensor of weights after training previous task does not match current parameter size at corresponding
                # index (if, for example, we have expanded the network since training on that previous task),
                # pad the tensor of weights from parameter after training on given task with zeros so that it matches the
                # size in all dimensions of the corresponding parameter in the network
                if not torch.equal(task_weights_size, parameter_size):
                    pad_tuple = utils.pad_tuple(task_weights[param_index], parameter)
                    task_weights[param_index] = F.pad(task_weights[param_index], pad_tuple, mode='constant', value=0)

                # If size of fisher diagonal computed for previous task does not match current parameter size at corresponding
                # index (if, for example, we have expanded the network since training on that previous task),
                # pad the fisher diagonal for the parameter computed after training on the given task with zeros so that it matches the
                # size in all dimensions of the corresponding parameter in the network
                if not torch.equal(task_fisher_size, parameter_size):
                    pad_tuple = utils.pad_tuple(task_fisher[param_index], parameter)
                    task_fisher[param_index] = F.pad(task_fisher[param_index], pad_tuple, mode='constant', value=0)

                # add to the loss the part of the original summed ewc loss term corresponding to the specific task and parameter
                # in question (specified by the two for loops in this function)
                # (see: https://arxiv.org/pdf/1612.00796.pdf#section.2  equation 3)
                loss_prev_tasks += (((parameter - task_weights[param_index]) ** 2) * task_fisher[param_index]).sum()

        # multiply summed loss term by fisher multiplier divided by 2
        return loss_prev_tasks * (self.lam / 2.0)

    """
    def expand(self):
        expanded_sizes = []

        for parameter in self.parameters():
            expanded_sizes.append(list(parameter.size()))

        for i in range(len(expanded_sizes)):
            if i != 0 and i != len(expanded_sizes) - 1:
                expanded_sizes[i][-1] *= 2
            elif i == 0:
                expanded_sizes[i][0] *= 2

        for param_index, (name, parameter) in enumerate(self.named_parameters()):
            print(name)
            print(tuple(expanded_sizes[param_index]))
            print(parameter.size())
            parameter.detach()
            parameter.requires_grad = False
            data = parameter.data.clone()
            data.resize_(tuple(expanded_sizes[param_index]))
            self.register_parameter('{}'.format(param_index), torch.nn.Parameter(data=data, requires_grad=True))

        for parameter in self.parameters():
            print(parameter.size(), parameter.requires_grad)

    """