import torch
import utils
import setup
import numpy as np
from model import Model
from copy import deepcopy
from torch.autograd import Variable



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

    utils.output_tensorboard_graph(args, models, task_count)

    # dictionary, format {task number: size of network parameters (weights) when the network was trained on the task}
    model_size_dictionaries = []

    # initialize model size dictionaries
    for model in models:
        model_size_dictionaries.append({})


    # keep learning tasks ad infinitum
    while(True):

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

        for model_num in range(len(models)):

            # for each desired epoch, train the model on the latest task
            for epoch in range(1, args.epochs + 1):
                models[model_num].train_model(args, device, train_loader, epoch, task_count)

            # update the model size dictionary
            model_size_dictionaries[model_num].update({task_count: models[model_num].hidden_size})

            # generate a dictionary mapping tasks to models of the sizes that the network was when those tasks were
            # trained, containing subsets of the weights currently in the model (to mask new, post-expansion weights
            # when testing on tasks for which the weights did not exist during training)
            test_models = utils.generate_model_dictionary(models[model_num], model_size_dictionaries[model_num])

            # test the model on ALL tasks trained thus far (including current task)
            utils.test(test_models, device, test_loaders)


            # If the model currently being used in the loop is using EWC, we need to compute the fisher information
            if models[model_num].ewc:

                # save the theta* ("theta star") values after training - for plotting and comparative loss calculations
                # using the method in model.alternative_ewc_loss()
                #
                # NOTE: when I reference theta*, I am referring to the values represented by that variable in
                # equation (3) at:
                #   https://arxiv.org/pdf/1612.00796.pdf#section.2
                current_weights = []

                for parameter in models[model_num].parameters():
                    current_weights.append(deepcopy(parameter.data.clone()))


                models[model_num].task_post_training_weights.update({task_count: deepcopy(current_weights)})

                # using validation set in Fisher Information Matrix computation as specified by:
                # https://github.com/ariseff/overcoming-catastrophic/blob/master/experiment.ipynb
                models[model_num].estimate_fisher(device, validation_loader)

                # update the ewc loss sums in the model to incorporate weights and fisher info from the task on which
                # we just trained the network
                models[model_num].update_ewc_sums()

                # store the current fisher diagonals for use with plotting and comparative loss calculations
                # using the method in model.alternative_ewc_loss()
                models[model_num].task_fisher_diags.update({task_count: deepcopy(models[model_num].list_of_fisher_diags)})



        # expand all models before training the next task
        if task_count == 4:
            print("EXPANDING...")
            for model_num in range(len(models)):
                models[model_num].expand()
                utils.output_tensorboard_graph(args, models, task_count)



        # increment the number of the current task before re-entering while loop
        task_count += 1


if __name__ == '__main__':
    main()
