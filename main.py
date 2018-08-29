import torch
import utils
import setup
from EWCCNN import EWCCNN
from EWCMLP import EWCMLP
from VanillaMLP import VanillaMLP
from VanillaCNN import VanillaCNN
import numpy as np
import scipy as sp
import h5py
import random
from Continuum import Continuum
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

def show_mnist_img(batch, task):
    fig = plt.figure()
    
    for i, img in enumerate(batch, 1):
        plt.subplot(len(batch) // 10, 10, i)
        npimg = img.numpy()
        npimg = np.resize(npimg, (28,28))
        plt.imshow(npimg)
    
    plt.show()

def main():

    args = setup.parse_arguments()

    kwargs, device = setup.set_gpu_options(args)
    
    print(device)

    setup.seed_rngs(args)
    
    # test seeding...
    print(np.random.randint(low=1, high=10000))
    print(sp.stats.randint.rvs(1, 10000, size=1))
    print(torch.randint(1, 10000, (1,), device=torch.device('cpu')))
    print(torch.randint(1, 10000, (1,), device=torch.device('cuda'))) 
    print(random.randint(1, 10000))

    # print 8 digits of precision when displaying floating point output from tensors
    torch.set_printoptions(precision=8)

    models = setup.build_models(args, device)

    # The number of the task on which we are CURRENTLY training in the loop below-
    # e.g. when training on task 3 this value will be 3
    task_count = 1

    #utils.output_tensorboard_graph(args, device, models, task_count) # TODO change this to model.device in utils/ model constructor


    # todo fix kwargs (**kwargs)
    # todo fix num_workers in utils

    #cifar_classes = utils.define_cifar100_labels()


    # for task in range(args.tasks):
    #     # get some random training images
    #     images, labels = next(iter(test_loaders[task]))
    #
    #     # show images
    #     utils.imshow(torchvision.utils.make_grid(images))
    #
    #     # print labels
    #     print(' '.join('%5s' % cifar_classes[labels[j]] for j in range(len(labels))))


    # A list of the different DataLoader objects that hold various permutations of the mnist testing dataset-
    # used for testing models on all previously encountered tasks
    prev_test_loaders = []

    retrain_task = False

    files, expansions, avg_acc, task_acc, h5file = setup.setup_h5_file(args, models)

    # metrics for measuring network strain- saved to h5 dataset later

    ### ALL OF THESE ARE EVALUATED BASED ON THE RUNNING SUMS OF THE FISHER INFO ###
    failure = [0] # task at which network fails without expansion
    fisher_total = [0] # sum of all calculated FIM diagonals for each task
    post_training_loss = [0] # training loss after final training iteration on each task
    fisher_average = [0] # avg of FIM diags
    fisher_st_dev = [0] # standard deviation of FIM diags
    fisher_max = [0] # maximum Fisher Info of any parameter in the network
    fisher_information = [[0]] # actual FIM diagonals
    ewc_pen = [0, 0] # ewc penalty (loss on previous tasks only) is 0 for task 1 and 0 (filler)

    if args.dataset == "cifar":
        train_loaders, validation_loaders, test_loaders = utils.generate_cifar_tasks(args, kwargs)

    while task_count < (args.tasks + 1) :

        torch.cuda.empty_cache() # free any available gpu memory

        if not retrain_task:

            if args.dataset == "cifar":
                train_loader = train_loaders[task_count - 1]
                validation_loader = validation_loaders[task_count - 1]
                test_loader = test_loaders[task_count - 1]

                # todo remove- just for testing CNNs
                #train_loader, test_loader = utils.generate_1_cifar10_task(args)

            else:# todo add this to the arg parser
                # get the DataLoaders for the training, validation, and testing data
                train_loader, validation_loader, test_loader = utils.generate_new_mnist_task(args, kwargs,
                    first_task=(task_count == 1)
                )
    
            # add the new test_loader for this task to the list of testing dataset DataLoaders for later re-use
            # to evaluate how well the models retain accuracy on old tasks after learning new ones
            #
            # NOTE: this list also includes the current test_loader, which we are appending here, because we also
            # need to test each network on the current task after training
            prev_test_loaders.append(test_loader)
                
            # verify percent permuation of images
            # train_batch, labels = next(iter(train_loader))

            # show_mnist_img(train_batch, task_count)
            #  
            # test_batch, labels = next(iter(test_loader))

            # show_mnist_img(test_batch, task_count)

        retrain_task = False

        for model_num, model in enumerate(models):
            train_args = {'validation_loader': validation_loader,
                          'failure': failure,
                          'fisher_total': fisher_total,
                          'post_training_loss': post_training_loss,
                          'fisher_average': fisher_average,
                          'fisher_st_dev': fisher_st_dev,
                          'fisher_max': fisher_max,
                          'fisher_information': fisher_information,
                          'h5file': h5file,
                          'ewc_pen': ewc_pen
                        } \
            if isinstance(model, (EWCMLP, EWCCNN)) else {}

            # for each desired epoch, train the model on the latest task
            model.train_model(args, train_loader, task_count, **train_args)

            threshold = 0 if isinstance(model, (VanillaMLP, VanillaCNN)) else args.accuracy_threshold

            # test the model on ALL tasks trained thus far (including current task)
            test_results = model.test(prev_test_loaders, threshold, args)

            if test_results == -1:
                retrain_task = True
                break

            else:
                task_acc[model_num][:len(test_results)] = np.array(test_results)[...]

        if retrain_task:

            for model in models:
                model.reset(task_count - 1)

            models = utils.expand(models, args)

            for model_num in range(len(models)):
                expansions[model_num][task_count] += 1

            #utils.output_tensorboard_graph(args, models, task_count + 1)

        else:
            for model_num in range(len(models)):
                avg_acc[model_num][task_count] = sum(task_acc[model_num]) / task_count

            # increment the number of the current task before re-entering while loop
            task_count += 1

        for f in files:
            f.flush()


    for f in files:

        print("|-----[", f.filename, "]-----|", '\n')

        print("METADATA:\n")
        print([d for d in f['metadata']], '\n')

        print("FINAL TASK ACCURACIES:\n")
        print([d for d in f['task_acc']], '\n')

        print("EXPANSIONS:\n")
        print([d for d in f['expansions']], '\n')

        print("AVERAGE ACCURACIES:\n")
        print([d for d in f['avg_acc']], '\n')

        f.close()

if __name__ == '__main__':
    main()
