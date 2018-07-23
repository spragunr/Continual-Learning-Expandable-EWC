import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from copy import deepcopy
from CNN import CNN

class VanillaCNN(CNN):
    def __init__(self, hidden_size, input_size, output_size, device):

        super().__init__(hidden_size, input_size, output_size, device)

    @classmethod
    def from_existing_model(cls, m, new_hidden_size):

        model = cls(new_hidden_size, m.input_size, m.output_size, m.device)

        model.size_dictionary = deepcopy(m.size_dictionary)

        model.task_post_training_weights = deepcopy(m.task_post_training_weights)

        model.copy_weights_expanding(m)

        return model

    def train_model(self, args, train_loader, task_number, **kwargs):

        # Set the module in "training mode"
        # This is necessary because some network layers behave differently when training vs testing.
        # Dropout, for example, is used to zero/mask certain weights during TRAINING to prevent overfitting.
        # However, during TESTING (e.g. model.eval()) we do not want this to happen.
        # todo uncomment this later ?
        #self.train()

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

        running_loss = 0.0

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
            for batch_idx, (data, target) in enumerate(train_loader, 0):

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

                # todo remove from CNN?
                # data = data.view(data_size, -1)

                # wrap data and target in variables- again, from the following experiment:
                #   https://github.com/kuc2477/pytorch-ewc/blob/4a75734ef091e91a83ce82cab8b272be61af3ab6/train.py#L50
                #
                # .to(device):
                #   set the device (CPU or GPU) to be used with data and target to device variable (defined in main())

                # todo may not need to wrap in variables - just to move to GPU if necessary
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

                running_loss += loss.item()

                # Each time the batch index is a multiple of the specified progress display interval (args.log_interval),
                # print a message indicating progress AND which network (model) is reporting values.
                if batch_idx % args.log_interval == args.log_interval - 1:
                    print('{} Task: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                                                                    'NoReg',
                                                                                    task_number,
                                                                                    epoch,
                                                                                    batch_idx * len(data),
                                                                                    args.train_dataset_size,
                                                                                    100. * batch_idx / len(train_loader),
                                                                                    running_loss / args.log_interval
                                                                                    ))
                    running_loss = 0.0


        # update the model size dictionary
        self.update_size_dict(task_number)

        self.save_theta_stars(task_number)
