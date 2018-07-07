import torch
import utils
import setup
import torch.nn as nn
from EWCModel import EWCModel
from NoRegModel import NoRegModel
import matplotlib.pyplot as plt
import numpy as np
import torchvision

def main():

    args = setup.parse_arguments()

    kwargs, device = setup.set_gpu_options(args)

    setup.seed_rngs(args)

    # print 8 digits of precision when displaying floating point output from tensors
    torch.set_printoptions(precision=8)

    models = setup.build_models(args, device)

    # A list of the different DataLoader objects that hold various permutations of the mnist testing dataset-
    # used for testing models on all previously encountered tasks
    test_loaders = []

    # The number of the task on which we are CURRENTLY training in the loop below-
    # e.g. when training on task 3 this value will be 3
    task_count = 1

    #utils.output_tensorboard_graph(args, device, models, task_count) # TODO change this to model.device in utils/ model constructor

    # todo fix kwargs (**kwargs)
    # todo remove
    train_loader, validation_loader, test_loader = utils.generate_new_cifar_task(args, kwargs, True)

    data, target = next(iter(train_loader))



    retrain_task = False

    while(True):

        for model in models:
            for parameter in model.parameters():
                print(parameter.size())


        if not retrain_task:

            # get the DataLoaders for the training, validation, and testing data
            train_loader, validation_loader, test_loader = utils.generate_new_mnist_task(args, kwargs,
                first_task=(task_count == 1)
            )

            # add the new test_loader for this task to the list of testing dataset DataLoaders for later re-use
            # to evaluate how well the models retain accuracy on old tasks after learning new ones
            #
            # NOTE: this list also includes the current test_loader, which we are appending here, because we also
            # need to test each network on the current task after training
            test_loaders.append(test_loader)

        retrain_task = False

        for model in models:
            train_args = {'validation_loader': validation_loader} if type(model) == EWCModel else {}

            # for each desired epoch, train the model on the latest task
            model.train_model(args, train_loader, task_count, **train_args)

            threshold = 0 if type(model) == NoRegModel else args.accuracy_threshold

            # test the model on ALL tasks trained thus far (including current task)
            if model.test(test_loaders, threshold) == -1:
                retrain_task = True
                break

        if retrain_task:

            for model in models:
                model.reset(task_count - 1)

            models = utils.expand(models, args)
            #utils.output_tensorboard_graph(args, models, task_count + 1)

        else:
            # increment the number of the current task before re-entering while loop
            task_count += 1


if __name__ == '__main__':
    main()
