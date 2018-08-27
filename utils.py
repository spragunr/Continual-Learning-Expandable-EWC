import torch
import torch.utils.data as D
from torch.autograd import Variable
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import subprocess
import pickle

def generate_percent_permutation(percent, length):
    
    perm_size = int(length * (percent / 100.0))
    
    indices = np.random.choice(length, size=perm_size, replace=False)
    
    return indices

def apply_permutation(image, indices):

    permute_sample = image[indices]
    np.random.shuffle(permute_sample)
    
    image[indices] = permute_sample
    
    return image

# generate the DataLoaders corresponding to a permuted mnist task
def generate_new_mnist_task(args, kwargs, first_task):
    
    if args.perm == 100: # TODO reset to 100
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
    
    else:
        # permute only a specified percentage of the pixels in the image

        permutation = generate_percent_permutation(args.perm, 784)

        
        transformations = transforms.Compose(
            [
                transforms.ToTensor(),
                #transforms.Normalize((0.1307,), (0.3081,)), # TODO determine why network performs better w/o normalization
                transforms.Lambda(lambda x: x.view(-1, 1))
            ]) if first_task else transforms.Compose(
            [
                transforms.ToTensor(),
                #transforms.Normalize((0.1307,), (0.3081,)), # TODO determine why network performs better w/o normalization
                transforms.Lambda(lambda x: x.view(-1, 1)),
                transforms.Lambda(lambda x: apply_permutation(x, permutation))
            ])

    # Split the PyTorch MNIST training dataset into training and validation datasets, and transform the data.
    #
    # D.dataset.random_split(dataset, lengths):
    #   Randomly split a dataset into non-overlapping new datasets of given lengths
    #
    # datasets.MNIST():
    #   ARGUMENTS (in order):
    #   root (string) - Root directory of dataset where processed/training.pt and processed/test.pt exist.
    #   train (bool, optional) - If True, creates dataset from training.pt, otherwise from test.pt.
    #   transform (callable, optional) - A function/transform that takes in an PIL image and returns a transformed
    #                                       version. E.g, transforms.RandomCrop
    #   download (bool, optional) - If true, downloads the dataset from the internet and puts it in root directory.
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
    # dataset (Dataset) - dataset from which to load the data (train_data prepared in above statement, in this case).
    # batch_size (int, optional) - how many samples per batch to load (default: 1).
    # shuffle (bool, optional) - set to True to have the data reshuffled at every epoch (default: False).
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

    # output expansion notification to terminal
    print("|-----[EXPANDING MODEL AND RETRAINING LAST TASK]-----|\n")

    expanded_models = []

    for model_num, model in enumerate(models):
        if model.__class__.__bases__[0].__name__ == 'MLP':
            expanded_models.append(
                model.__class__.from_existing_model(model, model.hidden_size * args.scale_factor).to(model.device))
        elif model.__class__.__bases__[0].__name__ == 'CNN':
            expanded_models.append(
                model.__class__.from_existing_model(model, model.hidden_size + args.scale_factor).to(model.device))
        else:
            print("ERROR- invalid network type detected")

    for model in expanded_models:
        for parameter in model.parameters():
            print(parameter.size())

    return expanded_models

# generate the DataLoaders corresponding to a permuted CIFAR 10 task
def generate_new_cifar_task(args, kwargs, first_task):

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
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]) if first_task else transforms.Compose(
            [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(3, 1024)[:, pixel_permutation]),
            transforms.Lambda(lambda x: x.view(3, 32, 32)),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]) if first_task else transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1, 1)[pixel_permutation]),
            transforms.Lambda(lambda x: x.view(32, 32)),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    # Split the PyTorch MNIST training dataset into training and validation datasets, and transform the data.
    #
    # D.dataset.random_split(dataset, lengths):
    #   Randomly split a dataset into non-overlapping new datasets of given lengths
    #
    # datasets.MNIST():
    #   ARGUMENTS (in order):
    #   root (string) - Root directory of dataset where processed/training.pt and processed/test.pt exist.
    #   train (bool, optional) - If True, creates dataset from training.pt, otherwise from test.pt.
    #   transform (callable, optional) - A function/transform that takes in an PIL image and returns a transformed
    #                                       version. E.g, transforms.RandomCrop
    #   download (bool, optional) - If true, downloads the dataset from the internet and puts it in root directory.
    #                                       If dataset is already downloaded, it is not downloaded again.
    train_data, validation_data = \
        D.dataset.random_split(datasets.CIFAR10('../data', train=True, transform=transform_train, download=True),
            [args.train_dataset_size, args.validation_dataset_size])

    # Testing dataset.
    # train=False, because we want to draw the data here from <root>/test.pt (as opposed to <root>/training.pt)
    test_data = datasets.CIFAR10('../data', train=False, transform=transform_test, download=True)

    # A PyTorch DataLoader combines a dataset and a sampler, and returns single- or multi-process iterators over
    # the dataset.

    # DataLoader for the training data.
    # ARGUMENTS (in order):
    # dataset (Dataset) - dataset from which to load the data (train_data prepared in above statement, in this case).
    # batch_size (int, optional) - how many samples per batch to load (default: 1).
    # shuffle (bool, optional) - set to True to have the data reshuffled at every epoch (default: False).
    # kwargs - see above definition
    train_loader = D.DataLoader(train_data, batch_size=10, shuffle=True, **kwargs)

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


# generate the DataLoaders corresponding to incremental CIFAR 100 tasks
def generate_cifar_tasks(args, kwargs):

    # for indicating progress...
    symbols = ['|', '/', '-', '\\']
    symbol_index = 0

    train_loaders = []
    validation_loaders = []
    test_loaders = []

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    

    # Split the PyTorch MNIST training dataset into training and validation datasets, and transform the data.
    #
    # D.dataset.random_split(dataset, lengths):
    #   Randomly split a dataset into non-overlapping new datasets of given lengths
    #
    # datasets.MNIST():
    #   ARGUMENTS (in order):
    #   root (string) - Root directory of dataset where processed/training.pt and processed/test.pt exist.
    #   train (bool, optional) - If True, creates dataset from training.pt, otherwise from test.pt.
    #   transform (callable, optional) - A function/transform that takes in an PIL image and returns a transformed
    #                                       version. E.g, transforms.RandomCrop
    #   download (bool, optional) - If true, downloads the dataset from the internet and puts it in root directory.
    #                                       If dataset is already downloaded, it is not downloaded again.
    train_data = datasets.CIFAR100('../data', transform=transform_train, train=True, download=True)

    # Testing dataset.
    # train=False, because we want to draw the data here from <root>/test.pt (as opposed to <root>/training.pt)
    test_data = datasets.CIFAR100('../data', transform=transform_test, train=False, download=True)

    # A PyTorch DataLoader combines a dataset and a sampler, and returns single- or multi-process iterators over
    # the dataset.

    # DataLoader for the training data.
    # ARGUMENTS (in order):
    # dataset (Dataset) - dataset from which to load the data (train_data prepared in above statement, in this case).
    # batch_size (int, optional) - how many samples per batch to load (default: 1).
    # shuffle (bool, optional) - set to True to have the data reshuffled at every epoch (default: False).
    # kwargs - see above definition
    train_loader = D.DataLoader(train_data, batch_size=1, shuffle=False, **kwargs)

    # Instantiate a DataLoader for the testing data in the same manner as above for training data, with two exceptions:
    #   Here, we use test_data rather than train_data, and we use test_batch_size
    test_loader = D.DataLoader(test_data, batch_size=1, shuffle=False, **kwargs)

    train_data_org_by_class = []
    test_data_org_by_class = []

    for i in range(100):
        train_data_org_by_class.append([])
        test_data_org_by_class.append([])

    for (data, target) in train_loader:
        print("CONSTRUCTING INCREMENTAL CIFAR 100 DATASET {}".format(symbols[(symbol_index // 100) % len(symbols)]), end='\r')
        symbol_index += 1

        train_data_org_by_class[target.item()].append((data, target))

    for (data, target) in test_loader:
        print("CONSTRUCTING INCREMENTAL CIFAR 100 DATASET {}".format(symbols[(symbol_index // 100) % len(symbols)]), end='\r')
        symbol_index += 1

        test_data_org_by_class[target.item()].append((data, target))

    task_class_indices = []

    class_count = 0

    for task in range(args.tasks):
        task_class_indices.append(range(class_count, class_count + 100 // args.tasks))
        class_count += 100 // args.tasks

    tasks_train = []
    tasks_test = []

    for task in task_class_indices:
        tasks_train.append([])
        tasks_test.append([])

        # task is a range object (e.g. range(0,5) for 1st task if CIFAR 100 split into 20 tasks)
        for class_data_index in task:
            print("CONSTRUCTING INCREMENTAL CIFAR 100 DATASET {}".format(symbols[(symbol_index // 100) % len(symbols)]), end='\r')
            symbol_index += 1

            for train_sample in train_data_org_by_class[class_data_index]:
                tasks_train[len(tasks_train) - 1].append(train_sample)

            for test_sample in test_data_org_by_class[class_data_index]:
                tasks_test[len(tasks_test) - 1].append(test_sample)


    for task in tasks_train:
        random.shuffle(task)

    for task in tasks_test:
        random.shuffle(task)

    for task in tasks_train:
        print("CONSTRUCTING INCREMENTAL CIFAR 100 DATASET {}".format(symbols[(symbol_index // 100) % len(symbols)]), end='\r')
        symbol_index += 1

        train_loader = task[:args.train_dataset_size]
        validation_loader = task[args.train_dataset_size:]

        batched_train_loader = []
        batched_validation_loader = []

        batch_start = 0

        for batch in range(len(train_loader) // args.batch_size):
            data_target_tuples = [train_loader[i] for i in range(batch_start, batch_start + args.batch_size)]

            data = torch.empty(0, dtype=torch.float)  # zero-dimensional tensor (empty)
            target = torch.empty(0, dtype=torch.long)

            for tuple in data_target_tuples:
                data = torch.cat((data, tuple[0]))
                target = torch.cat((target, tuple[1]))

            batched_train_loader.append((data, target))
            batch_start += args.batch_size

        # make the whole validation set one batch for fisher matrix computations (EWC)
        data_target_tuples = [validation_loader[i] for i in range(len(validation_loader))]

        data = torch.empty(0, dtype=torch.float) # zero-dimensional tensor (empty)
        target = torch.empty(0, dtype=torch.long)

        for tuple in data_target_tuples:
            data = torch.cat((data, tuple[0]))
            target = torch.cat((target, tuple[1]))

        batched_validation_loader.append((data, target))

        train_loaders.append(batched_train_loader)
        validation_loaders.append(batched_validation_loader)

    for task in tasks_test:
        print("CONSTRUCTING INCREMENTAL CIFAR 100 DATASET {}".format(symbols[(symbol_index // 100) % len(symbols)]), end='\r')
        symbol_index += 1

        batched_test_loader = []

        batch_start = 0

        for batch in range(len(task) // args.test_batch_size):
            data_target_tuples = [task[i] for i in range(batch_start, batch_start + args.test_batch_size)]

            data = torch.empty(0, dtype=torch.float)  # zero-dimensional tensor (empty)
            target = torch.empty(0, dtype=torch.long)

            for tuple in data_target_tuples:
                data = torch.cat((data, tuple[0]))
                target = torch.cat((target, tuple[1]))

            batched_test_loader.append((data, target))
            batch_start += args.test_batch_size

        test_loaders.append(batched_test_loader)

    print("\nDATASET CONSTRUCTION COMPLETE")
    return train_loaders, validation_loaders, test_loaders


# display a cifar image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def define_cifar100_labels():

    return [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]

def generate_1_cifar10_task(args):

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=1)

    testset = datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=1)

    return trainloader, testloader


def load_iCIFAR(args):
    d_tr, d_te = torch.load('data/processed/cifar100.pt')
    n_inputs = d_tr[0][1].size(1)
    n_outputs = 0
    for i in range(len(d_tr)):
        n_outputs = max(n_outputs, d_tr[i][2].max())
        n_outputs = max(n_outputs, d_te[i][2].max())
    return d_tr, d_te, n_inputs, n_outputs + 1, len(d_tr)


def build_iCIFAR(args):
    ########## DOWNLOAD AND FORMAT DATA ##########

    prefix = './data/raw/'

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    cifar_name = "cifar-100-python.tar.gz"

    cifar_path = prefix + cifar_name

    # URL from: https://www.cs.toronto.edu/~kriz/cifar.html
    if not os.path.exists(cifar_path):
        subprocess.call("wget -O {} https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz".format(
            cifar_path), shell=True)

    subprocess.call("tar xzfv {} -C {}".format(cifar_path, prefix), shell=True)

    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    cifar100_train = unpickle(prefix + 'cifar-100-python/train')
    cifar100_test = unpickle(prefix + 'cifar-100-python/test')

    x_tr = torch.from_numpy(cifar100_train[b'data'])
    y_tr = torch.LongTensor(cifar100_train[b'fine_labels'])
    x_te = torch.from_numpy(cifar100_test[b'data'])
    y_te = torch.LongTensor(cifar100_test[b'fine_labels'])

    torch.save((x_tr, y_tr, x_te, y_te), prefix + 'cifar100.pt')

    ######### SPLIT DATA INTO INCREMENTAL TASKS ##########

    tasks_tr = []
    tasks_te = []

    x_tr, y_tr, x_te, y_te = torch.load(prefix + 'cifar100.pt')
    x_tr = x_tr.float().view(x_tr.size(0), -1) / 255.0
    x_te = x_te.float().view(x_te.size(0), -1) / 255.0

    cpt = int(100 / args.tasks)

    for t in range(args.tasks):
        c1 = t * cpt
        c2 = (t + 1) * cpt
        i_tr = ((y_tr >= c1) & (y_tr < c2)).nonzero().view(-1)
        i_te = ((y_te >= c1) & (y_te < c2)).nonzero().view(-1)
        tasks_tr.append([(c1, c2), x_tr[i_tr].clone(), y_tr[i_tr].clone()])
        tasks_te.append([(c1, c2), x_te[i_te].clone(), y_te[i_te].clone()])

    processed_prefix = './data/processed/'

    if not os.path.exists(processed_prefix):
        os.makedirs(processed_prefix)

    torch.save([tasks_tr, tasks_te], processed_prefix + 'cifar100.pt')
