import torch
import utils
import setup
from EWCModel import EWCModel

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

    while(True):
        # expand all models before training the next task
        if task_count == 3:

            print("EXPANDING...")

            models = utils.expand(models, args)

            #utils.output_tensorboard_graph(args, models, task_count + 1)

        for model in models:
            for parameter in model.parameters():
                print(parameter.size())

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

        for model in models:

            train_args = {'validation_loader': validation_loader} if type(model) == EWCModel else {}

            # for each desired epoch, train the model on the latest task
            model.train_model(args, train_loader, task_count, **train_args)

            # generate a dictionary mapping tasks to models of the sizes that the network was when those tasks were
            # trained, containing subsets of the weights currently in the model (to mask new, post-expansion weights
            # when testing on tasks for which the weights did not exist during training)
            test_models = utils.generate_model_dictionary(model)

            # test the model on ALL tasks trained thus far (including current task)
            utils.test(test_models, test_loaders)

        # increment the number of the current task before re-entering while loop
        task_count += 1


if __name__ == '__main__':
    main()
