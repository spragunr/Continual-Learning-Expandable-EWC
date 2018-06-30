import torch
import torch.nn as nn
import torch.utils.data as D
from torch.autograd import Variable
from torchvision import datasets, transforms
from ExpandableModel import ExpandableModel
from EWCModel import EWCModel
from NoRegModel import NoRegModel
from tensorboardX import SummaryWriter


# generate the DataLoaders corresponding to a permuted mnist task
def generate_new_mnist_task(args, kwargs, first_task):

    # permutation to be applied to all images in the dataset (if this is not the first dataset being generated)
    pixel_permutation = torch.randperm(args.input_size)

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
    # todo comment on sequential mnist and pixel permuation
    # permutation from: https://discuss.pytorch.org/t/sequential-mnist/2108 (first response)
    transformations = transforms.Compose(
        [
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,)), # TODO determine why network performs better w/o normalization
            transforms.Lambda(lambda x: x.view(-1, 1))
        ]) if first_task else transforms.Compose(
        [
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,)), # TODO determine why network performs better w/o normalization
            transforms.Lambda(lambda x: x.view(-1, 1)[pixel_permutation])
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
            [args.train_dataset_size, args.validation_dataset_size])

    # Testing dataset.
    # train=False, because we want to draw the data here from <root>/test.pt (as opposed to <root>/training.pt)
    test_data = datasets.MNIST('../data', train=False, transform=transformations, download=True)

    # A PyTorch DataLoader combines a dataset and a sampler, and returns single- or multi-process iterators over
    # the dataset.

    # DataLoader for the training data.
    # ARGUMENTS (in order):
    # dataset (Dataset) – dataset from which to load the data (train_data prepared in above statement, in this case).
    # batch_size (int, optional) – how many samples per batch to load (default: 1).
    # shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).
    # kwargs - see above definition
    train_loader = D.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)

    # Dataloader for the validation dataset-
    # ARGUMENTS (in order):
    # 1) validation_data as the dataset
    # 2) batch_size is the entire validation set in one batch
    # 3) shuffle=True ensures we are drawing random samples by shuffling the data each time we contstruct a new iterator
    #       over the data, and is implemented in the source code as a RandomSampler()
    # 4) kwargs defined above
    validation_loader = D.DataLoader(validation_data, batch_size=args.validation_dataset_size, shuffle=True, **kwargs)

    # Instantiate a DataLoader for the testing data in the same manner as above for training data, with two exceptions:
    #   Here, we use test_data rather than train_data, and we use test_batch_size
    test_loader = D.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    return train_loader, validation_loader, test_loader


def test(models, device, test_loaders):
    # Test the model on ALL tasks, including that on which the model was most recently trained
    for task_number, test_loader in enumerate(test_loaders):

        # from a dictionary formatted as {task number: model to use when testing that task number}, generated by
        # utils.generate_model_dictionary(), fetch the model to be used when testing this task (so as to mask
        # weights which should not be taken into consideration)
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

                # The data needs to be wrapped in another tensor to work with our network,
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
                # loss for each testing batch and then would average those at the end of testing on the current task
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
        print('\nTest set {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            task_number + 1, test_loss, correct, len(test_loader.dataset),
            accuracy))


# copy weights from a larger to a smaller model - used when generating smaller models with subsets of current
# model weights for testing the network on previous tasks...
def copy_weights_shrinking(big_model, small_model):


    big_weights = [] # weights in parameters from the larger model

    # save data from big model
    for parameter in big_model.parameters():
        big_weights.append(parameter.data.clone())

    # transfer that data to the smaller model -
    # copy each weight from larger network that should still be in the smaller model to matching index
    # in the smaller network
    for param_index, parameter in enumerate(small_model.parameters()):
        parameter.data[...] = big_weights[param_index][tuple(slice(0, n) for n in list(parameter.size()))]


# given a dictionary with task numbers as keys and model sizes (size of hidden layer(s) in the model when the model was
# trained on a given task) as values, generate and return a dictionary correlating task numbers with model.Model
# objects of the appropriate sizes, containing subsets of the weights currently in model
def generate_model_dictionary(m, device):

    model_sizes = []

    # fetch all unique model sizes from the model size dictionary and store them in a list (model_sizes)
    for key in m.size_dictionary.keys():
        if not m.size_dictionary.get(key) in model_sizes:
            model_sizes.append(m.size_dictionary.get(key))

    models = []

    # make a model of each size specified in model_sizes, add them to models list
    for hidden_size in model_sizes:
        models.append(
            ExpandableModel(
                hidden_size,
                m.input_size,
                m.output_size,
            ).to(device)
        )

    # copy subsets of weights from the largest model to all other models
    for to_model in models:
        copy_weights_shrinking(m, to_model)

    model_dictionary = {}

    # build the model dictionary
    for model in models:
        for task_number in [k for k,v in m.size_dictionary.items() if v == model.hidden_size]:
            model_dictionary.update({task_number: model})

    return model_dictionary


# Generate and return a tuple representing the padding size to be used as an argument to torch.nn.functional.pad().
# Tuple format and more in-depth explanation of the effects of pad() are in documentation of the pad() method here:
# https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
def pad_tuple(smaller, larger):

    pads_required = []

    # loop over the dimensions of the tensor we are padding so that this method can be used with both 2D weights and
    # 1D biases
    for dim in range(len(list(smaller.size()))):

        # pad by the difference between the existing and desired sizes in the given dimension
        pads_required.append(list(larger.size())[dim] - list(smaller.size())[dim])

        # After following reversal, will result in NO zero padding to the left of a 1D tensor (only extend to the right) and NO zero padding on
        # the left or top of a 2D tensor (only extend to the right and down). For instance, if a 2D tensor is
        # quadrupling in size, the original values in the tensor will be in the upper-left quadrant, and the other
        # three quadrants will be padded with zeros.
        pads_required.append(0)

    # this will correct the order of the values in the resulting list to produce the desired output
    pads_required.reverse()

    return tuple(pads_required)


def output_tensorboard_graph(args, models, task_count):

    dummy_input = Variable(torch.rand(args.batch_size, args.input_size))

    for model in models:
        with SummaryWriter(comment='MODEL task count: {}, type: {}'.format(task_count, model.__class__.__name__)) as w:
            w.add_graph(model, (dummy_input,))

def expand(models, args):

    expanded_models = []

    for model_num, model in enumerate(models):
        expanded_models.append(model.__class__.from_existing_model(model, model.hidden_size * args.scale_factor))

    return expanded_models