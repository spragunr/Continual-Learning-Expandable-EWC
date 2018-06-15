from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from matplotlib import rc

# NOTE on unused imports above: for some reason, removing these will result in errors when attempting to plot in 3D-
# unknown argument 'projection=3d'... so they are left in



# plotting of sum of how far each weight in each layer of the network has moved from each previous
# theta* value for that weight, multiplied by the fisher diagonal value for that weight computed for each
# of the tasks to which the theta* values correspond

def plot(weights, task_post_training_weights, task_count, task_fisher_diags):

    # main figure and title
    fig = plt.figure()
    fig.suptitle('Task {}'.format(task_count))

    position = 1 # used to place the subplots

    weight_count = 1 # used to label the parameters
    bias_count = 1 # used to label the parameters

    z_limits = [.00007, .000020, .0002] # z-axis upper limits for the weight parameters in order they appear in network (optional)
    y_limits = [.0000016, .0000035, .00001] # y-axis upper limits for the bias parameters in order they appear in network (optional)


    # NOTE: each of the subplots will be the shape/size of the corresponding network parameter, to help visualize
    # locations of each individual weight within the parameter
    for param_index, parameter in enumerate(weights):

        # weights - 2 dim
        if len(list(parameter.shape)) == 2:
            ax = fig.add_subplot(len(weights) / 2, 2, position, projection='3d')
            ax.set_title('weights layer {}'.format(weight_count))

            # optional setting of z limit
            #ax.set_zlim(0, z_limits[weight_count - 1])

            # z-axis label
            ax.set_zlabel(r'$\sum_{task=1}^{T-1} F_{task,\theta}(\theta - \theta_{task})^2$')

            weight_count += 1 # each parameter containing weights should have its own unique number label

            # set up x and y axes
            x = np.arange(list(parameter.shape)[1])
            y = np.arange(list(parameter.shape)[0])
            X, Y = np.meshgrid(x, y)

            # create zero-filled z axis data to the appropriate shape (parameter shape)
            z_data = np.zeros((list(parameter.shape)[0], list(parameter.shape)[1]))

            # calculate and plot the values specified in the method docstring for all of the weights in the current
            # parameter (this iteration in outer for loop)
            for row in range(len(z_data)):
                for col in range(len(z_data[row])):
                    for task in range(1, task_count):
                        task_weights = task_post_training_weights.get(task)
                        task_fisher = task_fisher_diags.get(task)
                        z_data[row][col] += ((parameter.data[row][col] - task_weights[param_index][row][col]) ** 2) \
                                            * task_fisher[param_index][row][col]

            Z = z_data

            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='inferno')

        # biases - 1 dim
        else:
            ax = fig.add_subplot(len(weights) / 2, 2, position)
            ax.set_title('bias layer {}'.format(bias_count))

            # optional setting of y limit
            #ax.set_ylim(0, y_limits[bias_count - 1])

            # y-axis label
            ax.set_ylabel(r'$\sum_{task=1}^{T-1} F_{task,\theta}(\theta - \theta_{task})^2$')

            bias_count += 1 # each parameter containing biases should have its own unique number label

            # set up x-axis
            x = np.arange(list(parameter.shape)[0])

            # create zero-filled y-axis data of appropriate size (dimension of bias parameter)
            y_data = np.zeros((list(parameter.shape)[0]))

            # calculate and plot the values specified in the method docstring for all of the weights in the current
            # parameter (this iteration in outer for loop)
            for index in range(len(y_data)):
                for task in range(1, task_count):
                    task_weights = task_post_training_weights.get(task)
                    task_fisher = task_fisher_diags.get(task)
                    y_data[index] += ((parameter.data[index] - task_weights[param_index][index]) ** 2) * task_fisher[param_index][index]

            y = y_data

            ax.plot(x, y)

        position += 1 # move to next subplot

    fig.set_size_inches(36, 17) # can be adjusted for screen resolution
    fig.savefig('../../task{}.png'.format(task_count)) # save the entire figure (containing all subplots)


