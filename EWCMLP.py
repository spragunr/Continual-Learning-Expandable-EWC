import torch
import utils
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from copy import deepcopy
from MLP import MLP


class EWCMLP(MLP):
    def __init__(self, hidden_size, input_size, output_size, device, lam):

        super().__init__(hidden_size, input_size, output_size, device)

        self.lam = lam  # the value of lambda (fisher multiplier) to be used in EWC loss computation

        # dictionary, format:
        # {task number : list of Fisher diagonals calculated after model trained on task}
        self.task_fisher_diags = {}

        self.initialize_fisher_sums()

    @classmethod
    def from_existing_model(cls, m, new_hidden_size):

        model = cls(new_hidden_size, m.input_size, m.output_size, m.device, m.lam)

        model.task_fisher_diags = deepcopy(m.task_fisher_diags)

        model.task_post_training_weights = deepcopy(m.task_post_training_weights)

        model.size_dictionary = deepcopy(m.size_dictionary)

        model.sum_Fx = deepcopy(m.sum_Fx)
        model.sum_Fx_Wx = deepcopy(m.sum_Fx_Wx)
        model.sum_Fx_Wx_sq = deepcopy(m.sum_Fx_Wx_sq)

        model.copy_weights_expanding(m)

        model.expand_ewc_sums()

        return model

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
                zeros = torch.zeros(tuple(parameter.size())).to(self.device)

                empty_sums.append(zeros)

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

        for ewc_sum in range(len(ewc_sums)):
            for parameter_index, parameter in enumerate(self.parameters()):
                # current size of entry at parameter_index in given list of sums
                sum_size = torch.Tensor(list(ewc_sums[ewc_sum][parameter_index].size()))

                # current size of parameter in the model corresponding to the sum entry above
                parameter_size = torch.Tensor(list(parameter.size()))

                # pad the sum tensor at the current parameter index of the given sum list with zeros so that it matches the size in
                # all dimensions of the corresponding parameter
                if not torch.equal(sum_size, parameter_size):
                    pad_tuple = utils.pad_tuple(ewc_sums[ewc_sum][parameter_index],parameter)
                    ewc_sums[ewc_sum][parameter_index] = F.pad(ewc_sums[ewc_sum][parameter_index], pad_tuple, mode='constant', value=0)


    # calculate the EWC loss on previous tasks only (not incorporating current task cross entropy)
    def ewc_loss_prev_tasks(self):

        loss_prev_tasks = 0

        # this computes the ewc loss on previous tasks via the algebraically manipulated fisher sums method:
        # (Weights_{current}) ** 2 * sigma (Fisher_{task}) - 2 * Weights_{current} * sigma (Fisher_{task} * Weights_{task}) +
        #          sigma (Fisher_{task} * (Weights_{task}) ** 2)
        #
        # for each parameter, we add to the loss the above loss term calculated for each weight in the parameter (summed)
        for parameter_index, (name, parameter) in enumerate(self.named_parameters()):

            if name != 'modulelist.{}.weight'.format(len(self.modulelist) - 1) and \
              name != 'modulelist.{}.bias'.format(len(self.modulelist) - 1):

                # NOTE: * operator is element-wise multiplication
                loss_prev_tasks += torch.sum(torch.pow(parameter, 2.0) * self.sum_Fx[parameter_index])
                loss_prev_tasks -= 2 * torch.sum(parameter * self.sum_Fx_Wx[parameter_index])
                loss_prev_tasks += torch.sum(self.sum_Fx_Wx_sq[parameter_index])

        # Mutliply error by fisher multiplier (lambda) divided by 2
        return loss_prev_tasks * (self.lam / 2.0)

    def train_model(self, args, train_loader, task_number, **kwargs):

        # Set the module in "training mode"
        # This is necessary because some network layers behave differently when training vs testing.
        # Dropout, for example, is used to zero/mask certain weights during TRAINING to prevent overfitting.
        # However, during TESTING (e.g. model.eval()) we do not want this to happen.
        self.train()

        self.reinitialize_output_weights() #todo compare to Vanilla MLP to confirm changes

        # Set the optimization algorithm for the model- in this case, Stochastic Gradient Descent with/without
        # momentum (depends on the value of args.momentum- default is 0.0, so no momentum by default).
        #
        # ARGUMENTS (in order):
        #     params (iterable) - iterable of parameters to optimize or dicts defining parameter groups
        #     lr (float) - learning rate
        #     momentum (float, optional) - momentum factor (default: 0)
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
        #optimizer = optim.Adadelta(self.parameters())

        for epoch in range(1, args.epochs + 1):
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
                data_size = len(data) # todo maybe adjust dataloader interpretations to deal with my custom dataloaders?

                data = data.view(data_size, -1)

                # wrap data and target in variables- again, from the following experiment:
                #   https://github.com/kuc2477/pytorch-ewc/blob/4a75734ef091e91a83ce82cab8b272be61af3ab6/train.py#L50
                #
                # .to(device):
                #   set the device (CPU or GPU) to be used with data and target to device variable (defined in main())
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)

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
                # apply the loss function to the predictions/labels for this batch to compute loss
                loss = F.cross_entropy(output, target)

                # if the model is using EWC, the summed loss term from the EWC equation (loss on previuous tasks) must be calculated and
                # added to the loss that will be minimized by the optimizer.
                #
                # See equation (3) at:
                #   https://arxiv.org/pdf/1612.00796.pdf#section.2
                if task_number > 1: # todo change to hasattr() call
                    # This statement computes loss on previous tasks using the summed fisher terms as in ewc_loss_prev_tasks()
                    loss += self.ewc_loss_prev_tasks()

                    # Using the commented-out version statement below instead of the one above will calculate ewc loss
                    # on previous tasks by multiplying the square of the difference between the current network
                    # parameter weights and those after training each previously encountered task, multiplied by the
                    # Fisher diagonal computed for the respective previous task in each difference, all summed together.

                    #loss += self.alternative_ewc_loss(task_number)

                # Backward pass: compute gradient of the loss with respect to model
                # parameters
                loss.backward()

                if task_number > 1: # todo change to hasattr() call
                    self.tune_variable_learning_rates()

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
                                                                                    'EWC',
                                                                                    task_number,
                                                                                    epoch,
                                                                                    batch_idx * len(data),
                                                                                    args.train_dataset_size,
                                                                                    100. * batch_idx / len(train_loader),
                                                                                    loss.item()
                                                                                    ))


        # update the model size dictionary
        self.update_size_dict(task_number)

        self.save_theta_stars(task_number)

        # using validation set in Fisher Information Matrix computation as specified by:
        # https://github.com/ariseff/overcoming-catastrophic/blob/master/experiment.ipynb
        self.estimate_fisher(kwargs.get("validation_loader"), args)

        # update the ewc loss sums in the model to incorporate weights and fisher info from the task on which
        # we just trained the network
        self.update_ewc_sums()

        # store the current fisher diagonals for use with plotting and comparative loss calculations
        # using the method in model.alternative_ewc_loss()
        self.save_fisher_diags(task_number)

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

    # used for whole batch
    def estimate_fisher(self, validation_loader, args):

        # List to hold the computed fisher diagonals for the task on which the network was just trained.
        # Fisher Information Matrix diagonals are stored as a list of tensors of the same dimensions and in the same
        # order as the parameters of the model given by model.parameters()
        self.list_of_fisher_diags = []

        # populate self.list_of_fisher_diags with tensors of zeros of the appropriate sizes
        for parameter in self.parameters():
            empty_diag = torch.zeros(tuple(parameter.size())).to(self.device)

            self.list_of_fisher_diags.append(empty_diag)

        softmax_activations = []

        # data is an batch of images
        # _ is a batch of labels for the images in the data batch (not needed)
        data, _ = next(iter(validation_loader))

        # The data needs to be wrapped in another tensor to work with our network,
        # otherwise it is not of the appropriate dimensions... I believe this statement effectively adds
        # a dimension.
        #
        # For an explanation of the meaning of this statement, see:
        #   https://stackoverflow.com/a/42482819/9454504
        #
        # This code was used here in another experiment:
        # https://github.com/kuc2477/pytorch-ewc/blob/4a75734ef091e91a83ce82cab8b272be61af3ab6/model.py#L61
        data = data.view(args.validation_dataset_size, -1)

        # wrap data and target in variables- again, from the following experiment:
        #   https://github.com/kuc2477/pytorch-ewc/blob/4a75734ef091e91a83ce82cab8b272be61af3ab6/model.py#L62
        #
        # .to(device):
        # set the device (CPU or GPU) to be used with data and target to device variable (defined in main())
        data = Variable(data).to(self.device)

        softmax_activations.append(
            F.softmax(self(data), dim=-1)
        )

        class_indices = torch.multinomial(softmax_activations[0], 1)

        random_log_likelihoods = []

        for row in range(len(class_indices)):
            random_log_likelihoods.append(torch.log(softmax_activations[0][row].index_select(0, class_indices[row][0])))


        for loglikelihood in random_log_likelihoods:

            # gradients of parameters with respect to log likelihoods (log_softmax applied to output layer),
            # data for the sample from the validation set is sent through the network to mimic the behavior
            # of the feed_dict argument at:
            # https://github.com/ariseff/overcoming-catastrophic/blob/afea2d3c9f926d4168cc51d56f1e9a92989d7af0/model.py#L65
            loglikelihood_grads = torch.autograd.grad(loglikelihood, self.parameters(), retain_graph=True)

            # square the gradients computed above and add each of them to the index in list_of_fisher_diags that
            # corresponds to the parameter for which the gradient was calculated
            for parameter in range(len(self.list_of_fisher_diags)):
                self.list_of_fisher_diags[parameter].add_(torch.pow(loglikelihood_grads[parameter], 2.0))

        # divide totals by number of samples, getting average squared gradient values across sample_count as the
        # Fisher diagonal values
        for parameter in range(len(self.list_of_fisher_diags)):
            self.list_of_fisher_diags[parameter] /= args.validation_dataset_size

    def save_fisher_diags(self, task_count):

        self.task_fisher_diags.update({task_count: deepcopy(self.list_of_fisher_diags)})

    # todo add modified version to EWCCNN
    def tune_variable_learning_rates(self):

        for parameter_index, (name, parameter) in enumerate(self.named_parameters()):

            if name != 'modulelist.{}.weight'.format(len(self.modulelist) - 1) and \
              name != 'modulelist.{}.bias'.format(len(self.modulelist) - 1):

                parameter.grad /= torch.clamp(self.sum_Fx[parameter_index] * self.lam, min = 1)
