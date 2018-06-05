import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as D
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from copy import deepcopy
from model import Model

# tested, works!
def apply_permutation(image, permutation):

    orig_shape = image.size()

    image = image.numpy()

    image.resize(784)

    perm_image = (deepcopy(image))

    for pixel_index, pixel in enumerate(perm_image):
        perm_image[pixel_index] = image[permutation[pixel_index]]

    image = torch.Tensor(perm_image)

    image.resize_(orig_shape)

    return image

def generate_new_mnist_task(train_dataset_size, validation_dataset_size, batch_size, test_batch_size, kwargs, first_task):

    # Note that, as in experiment from github/ariseff, these are permutations of the ORIGINAL dataset
    #
    # Generate numpy array containing 0 - 783 in random order - a permutation "mask" to be applied to each image
    # in the MNIST dataset
    permutation = np.random.permutation(784)

    # transforms.Compose() composes several transforms together.
    #
    # IF this is NOT the FIRST task, we should permute the original MNIST dataset to form a new task.
    #
    #  The transforms composed here are as follows:
    #
    # transforms.ToTensor():
    #     Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a
    #     torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    #
    # transforms.Normalize(mean, std):
    #     Normalize a tensor image with mean and standard deviation. Given mean: (M1,...,Mn) and
    #     std: (S1,..,Sn) for n channels, this transform will normalize each channel of the
    #     input torch.*Tensor i.e. input[channel] = (input[channel] - mean[channel]) / std[channel]
    #
    #     NOTE: the values used here for mean and std are those computed on the MNIST dataset
    #           SOURCE: https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
    #
    # transforms.Lambda() applies the enclosed lambda function to each image (x) in the DataLoader
    transformations = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]) if first_task else transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: apply_permutation(x, permutation))
        ])

    # Split the PyTorch MNIST training dataset into training and validation datasets, and transform the data.
    #
    # D.dataset.random_split(dataset, lengths):
    #   Randomly split a dataset into non-overlapping new datasets of given lengths
    #
    # datasets.MNIST():
    #   ARGUMENTS (in order):
    #   root (string) – Root directory of dataset where processed/training.pt and processed/test.pt exist.
    #   train (bool, optional) – If True, creates dataset from training.pt, otherwise from test.pt.
    #   transform (callable, optional) – A function/transform that takes in an PIL image and returns a transformed
    #                                       version. E.g, transforms.RandomCrop
    #   download (bool, optional) – If true, downloads the dataset from the internet and puts it in root directory.
    #                                       If dataset is already downloaded, it is not downloaded again.
    train_data, validation_data = \
        D.dataset.random_split(datasets.MNIST('../data', train=True, transform=transformations, download=True),
            [train_dataset_size, validation_dataset_size])

    # Testing dataset.
    # train=False, because we want to draw the data here from <root>/test.pt (as opposed to <root>/training.pt)
    test_data = \
        datasets.MNIST('../data', train=False, transform=transformations, download=True)

    # A PyTorch DataLoader combines a dataset and a sampler, and returns single- or multi-process iterators over
    # the dataset.

    # DataLoader for the training data.
    # ARGUMENTS (in order):
    # dataset (Dataset) – dataset from which to load the data (train_data prepared in above statement, in this case).
    # batch_size (int, optional) – how many samples per batch to load (default: 1).
    # shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).
    # kwargs - see above definition
    train_loader = D.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)

    # Dataloader for the validation dataset-
    # ARGUMENTS (in order):
    # 1) validation_data as the dataset
    # 2) batch_size is same as that provided for the training dataset
    # 3) shuffle=True ensures we are drawing random samples by shuffling the data each time we contstruct a new iterator
    #       over the data, and is implemented in the source code as a RandomSampler() - see comments in compute_fisher
    #       for more details and a link to the source code
    # 4) kwargs defined above
    validation_loader = D.DataLoader(validation_data, batch_size=batch_size, shuffle=True, **kwargs)

    # Instantiate a DataLoader for the testing data in the same manner as above for training data, with two exceptions:
    #   Here, we use test_data rather than train_data, and we use test_batch_size
    test_loader = D.DataLoader(test_data, batch_size=test_batch_size, shuffle=True, **kwargs)

    return train_loader, validation_loader, test_loader


def copy_weights_expanding(old_model, expanded_model):

    old_sizes = []
    old_weights = []

    # save data from old model
    for parameter in old_model.parameters():
        old_sizes.append(np.array(list(parameter.size())))
        old_weights.append(parameter.data.clone())

    # transfer that data to the expanded model
    for param_index, parameter in enumerate(expanded_model.parameters()):

        # weights - 2 dims
        if list(old_sizes[param_index].shape)[0] == 2:

            for row in range(len(old_weights[param_index])):

                for column in range(len(old_weights[param_index][row])):

                    # todo does this need to be in-place?
                    parameter.data[row][column] = old_weights[param_index][row][column]

        else:

            # biases - one dim
            for value_index in range(len(old_weights[param_index])):

                # todo does this need to be in-place?
                parameter.data[value_index] = old_weights[param_index][value_index]


def expand_model(model):

    expanded_model = Model(
        model.hidden_size * 2,
        model.hidden_dropout_prob,
        model.input_dropout_prob,
        model.input_size,
        model.output_size,
        model.ewc,
        model.lam
    )

    copy_weights_expanding(model, expanded_model)

    return expanded_model


def train(model, args, device, train_loader, epoch, task_number):
    # Set the module in "training mode"
    # This is necessary because some network layers behave differently when training vs testing.
    # Dropout, for example, is used to zero/mask certain weights during TRAINING to prevent overfitting.
    # However, during TESTING (e.g. model.eval()) we do not want this to happen.
    model.train()

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
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

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
        output = model(data)

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
        if model.ewc and task_number > 1:
            old_tasks_loss = calculate_ewc_loss_prev_tasks(model)
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
        # print a message indicating progress AND which network (model) is reporting values.
        if batch_idx % args.log_interval == 0:
            print('{} Task: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                'EWC' if model.ewc else 'SGD + DROPOUT', task_number,
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))



def test(models, device, test_loaders):
    # Test the model on ALL tasks, including that on which the model was most recently trained
    for task_number, test_loader in enumerate(test_loaders):

        model = models.get(task_number + 1)

        # Set the module in "evaluation mode"
        # This is necessary because some network layers behave differently when training vs testing.
        # Dropout, for example, is used to zero/mask certain weights during TRAINING (e.g. model.train())
        # to prevent overfitting. However, during TESTING/EVALUATION we do not want this to happen.
        model.eval()

        # total testing loss over all test batches for the given task_number's entire testset (sum)
        test_loss = 0

        # total number of correct predictions over the given task_number's entire testset (sum)
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
            # images in the corresponding test batch in order
            for data, target in test_loader:
                # For some reason, the data needs to be wrapped in another tensor to work with our network,
                # otherwise it is not of the appropriate dimensions... I believe this statement effectively
                # adds a dimension.
                #
                # For an explanation of the meaning of this statement, see:
                #   https://stackoverflow.com/a/42482819/9454504
                #
                # This code was used here in another experiment:
                #   https://github.com/kuc2477/pytorch-ewc/blob/4a75734ef091e91a83ce82cab8b272be61af3ab6/utils.py#L75
                data = data.view(test_loader.batch_size, -1)

                # wrap data and target in variables- again, from the following experiment:
                #   https://github.com/kuc2477/pytorch-ewc/blob/4a75734ef091e91a83ce82cab8b272be61af3ab6/utils.py#L76
                #
                # .to(device):
                # set the device (CPU or GPU) to be used with data and target to device variable (defined in main())
                data, target = Variable(data).to(device), Variable(target).to(device)

                # Forward pass: compute predicted output by passing data to the model. Module objects
                # override the __call__ operator so you can call them like functions. When
                # doing so you pass a Tensor of input data to the Module and it produces
                # a Tensor of output data. We have overriden forward() above, so our forward() method will be called here.
                output = model(data)

                # Define the testing loss to be cross entropy loss based on predicted values (output)
                # and ground truth labels (target), calculate the testing batch loss, and sum it with the total testing
                # loss over all batches in the given task_number's entire testset (contained within test_loss).
                #
                # NOTE: size_average = False:
                # By default, the losses are averaged over observations for each minibatch.
                # If size_average is False, the losses are summed for each minibatch. Default: True
                #
                # Here we use size_average = False because we want to SUM all testing batch losses and average those
                # at the end of testing on the current task (by dividing by total number of testing SAMPLES (not batches) to obtain an
                # average loss over all testing batches). Otherwise, if size_average == True, we would be getting average
                # loss for each testing batch and then would average those at the end of traing on the current task
                # to obtain average testing loss, which could theoretically result in some comparative loss of accuracy
                # in the calculation of the final testing loss value for this task.
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
                # In this case, the targets/labels are stored as scalar index values (e.g. torch.Tensor([1, 4, 5])
                # for labels for a one, a four, and a five (in that order)
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

        # Divide the accumulated test loss across all testing batches for the current task_number by the total number
        # of testing samples in the task_number's testset (in this case, 10,000) to get the average loss for the
        # entire test set for task_number.
        test_loss /= len(test_loader.dataset)

        # The overall accuracy of the model's predictions on the task indicated by task_number as a percent
        # value is the count of its accurate predictions divided by the number of predictions it made, all multiplied by 100
        accuracy = 100. * correct / len(test_loader.dataset)

        # For task_number's complete test set (all batches), display the average loss and accuracy
        print('\n{} Test set {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            'EWC' if model.ewc else 'SGD + DROPOUT', task_number + 1, test_loss, correct, len(test_loader.dataset),
            accuracy))


# compute fisher by randomly sampling from probability distribution of outputs rather than the activations
# themselves
def compute_fisher_prob_dist(model, device, validation_loader, num_samples):
    model.list_of_FIMs = []

    for parameter in model.parameters():
        model.list_of_FIMs.append(torch.zeros(tuple(parameter.size())))

    softmax = nn.Softmax()

    log_softmax = nn.LogSoftmax()

    probs = softmax(model.y)

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

        loglikelihood_grads = torch.autograd.grad(log_softmax(model(data))[0, class_index], model.parameters())

        for parameter in range(len(model.list_of_FIMs)):
            model.list_of_FIMs[parameter] += torch.pow(loglikelihood_grads[parameter], 2.0)

        if sample_number == num_samples - 1:
            break

    for parameter in range(len(model.list_of_FIMs)):
        model.list_of_FIMs[parameter] /= num_samples


# THE "OLD" WAY OF COMPUTING THIS...
# This method relies on the logic here:
#   https://github.com/kuc2477/pytorch-ewc/blob/4a75734ef091e91a83ce82cab8b272be61af3ab6/model.py#L56
def compute_fisher(model, device, validation_loader):
    # a list of log_likelihoods sampled from the model output when the input is
    # a sample from the validation dataset
    loglikelihoods = []

    # for every data sample in the validation set (default 1024)...
    for data, target in validation_loader:
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
        data, target = Variable(data).to(device), Variable(target).to(device)

        loglikelihoods.append(
            F.log_softmax(model(data))[range(validation_loader.batch_size), target.data]
        )

    # concatenate loglikelihood tensors in list loglikelihoods along 0th (default) dimension,
    # then calculate the mean of each row of the resulting tensor along the 0th dimension
    loglikelihood = torch.cat(loglikelihoods).mean(0)

    # here are the parameter gradients with respect to log likelihood
    loglikelihood_grads = torch.autograd.grad(loglikelihood, model.parameters())

    # list of Fisher Information Matrix diagonals
    model.fisher = []

    # see equation (2) at:
    #   https://arxiv.org/pdf/1605.04859.pdf#subsection.2.1
    for grad in loglikelihood_grads:
        model.fisher.append(torch.pow(grad, 2.0))


def save_theta_stars(model):

    # list of tensors used for saving optimal weights after most recent task training
    model.theta_stars = []

    # get the current values of each model parameter as tensors and add them to the list
    # self.theta_stars
    for parameter in model.parameters():
        model.theta_stars.append(parameter.data.clone())



def calculate_ewc_loss_prev_tasks(model):

    losses = []

    for parameter_index, parameter in enumerate(model.parameters()):

        theta_star = Variable(model.theta_stars[parameter_index])
        fisher = Variable(model.list_of_FIMs[parameter_index])

        losses.append((fisher * (parameter - theta_star) ** 2).sum())

    return (model.lam / 2.0) * sum(losses)


def copy_weights_shrinking(big_model, small_model):

    big_weights = []

    # save data from big model
    for parameter in big_model.parameters():
        big_weights.append(parameter.data.clone())

    small_sizes = []

    for parameter in small_model.parameters():
        small_sizes.append(np.array(list(parameter.size())))


    # transfer that data to the smaller model
    for param_index, parameter in enumerate(small_model.parameters()):

        # weights - 2 dims
        if list(small_sizes[param_index].shape)[0] == 2:

            for row in range(len(parameter.data)):

                for column in range(len(parameter.data[row])):

                    # todo does this need to be in-place?
                    parameter.data[row][column] = big_weights[param_index][row][column]

        else:

            # biases - one dim
            for value_index in range(len(parameter.data)):

                # todo does this need to be in-place?
                parameter.data[value_index] = big_weights[param_index][value_index]


# given a dictionary with task numbers as keys and model sizes (size of hidden layer(s) in the model when the model was
# trained on a given task) as values, generate and return a dictionary correlating task numbers with model.Model
# objects of the appropriate sizes
def generate_model_dictionary(model, model_size_dictionary):

    model_sizes = []

    # fetch all unique model sizes from the model size dictionary and store them in a list (model_sizes)
    for key in model_size_dictionary.keys():
        if not model_size_dictionary.get(key) in model_sizes:
            model_sizes.append(model_size_dictionary.get(key))

    models = []

    # make a model of each size specified in model_sizes, add them to models list
    for hidden_size in model_sizes:
        models.append(
            Model(
                hidden_size,
                model.hidden_dropout_prob,
                model.input_dropout_prob,
                model.input_size,
                model.output_size,
                model.ewc,
                model.lam
            )
        )

    for to_model in models:
        copy_weights_shrinking(model, to_model)

    model_dictionary = {}

    for model in models:
        for task_number in [k for k,v in model_size_dictionary.items() if v == model.hidden_size]:
            model_dictionary.update({task_number: model})


    return model_dictionary