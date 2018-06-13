import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as D
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

                    parameter.data[row][column].copy_(old_weights[param_index][row][column])

        else:

            # biases - one dim
            for value_index in range(len(old_weights[param_index])):

                # todo does this need to be in-place?
                parameter.data[value_index].copy_(old_weights[param_index][value_index])


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

    if model.ewc:
        # copy over old post-training weights and Fisher info
        #expanded_model.theta_stars = model.theta_stars
        expanded_model.list_of_FIMs = model.list_of_FIMs
        expanded_model.sum_Fx = model.sum_Fx
        expanded_model.sum_Fx_Wx = model.sum_Fx_Wx
        expanded_model.sum_Fx_Wx_sq = model.sum_Fx_Wx_sq

        expanded_model.expand_ewc_sums()

    copy_weights_expanding(model, expanded_model)

    return expanded_model


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


def pad_tuple(smaller, larger):

    pads_required = []

    for dim in range(len(list(smaller.size()))):
        pads_required.append(list(larger.size())[dim] - list(smaller.size())[dim])
        pads_required.append(0)

    pads_required.reverse()

    return tuple(pads_required)

