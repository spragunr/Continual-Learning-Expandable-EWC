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

            models = utils.expand(models, args, device) # todo change device to model instance variable

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

            # for each desired epoch, train the model on the latest task
            for epoch in range(1, args.epochs + 1):
                model.train_model(args, device, train_loader, epoch, task_count)

            # update the model size dictionary
            model.update_size_dict(task_count)

            # generate a dictionary mapping tasks to models of the sizes that the network was when those tasks were
            # trained, containing subsets of the weights currently in the model (to mask new, post-expansion weights
            # when testing on tasks for which the weights did not exist during training)
            test_models = utils.generate_model_dictionary(model, device)

            # test the model on ALL tasks trained thus far (including current task)
            utils.test(test_models, device, test_loaders)


            # If the model currently being used in the loop is using EWC, we need to compute the fisher information
            if type(model) == EWCModel:

                model.save_theta_stars(task_count)

                # using validation set in Fisher Information Matrix computation as specified by:
                # https://github.com/ariseff/overcoming-catastrophic/blob/master/experiment.ipynb
                model.estimate_fisher(device, validation_loader)

                # update the ewc loss sums in the model to incorporate weights and fisher info from the task on which
                # we just trained the network
                model.update_ewc_sums()

                # store the current fisher diagonals for use with plotting and comparative loss calculations
                # using the method in model.alternative_ewc_loss()
                model.save_fisher_diags(task_count)

        # increment the number of the current task before re-entering while loop
        task_count += 1


if __name__ == '__main__':
    main()
