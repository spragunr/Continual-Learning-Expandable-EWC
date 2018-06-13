'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np



def plot(weights, task_post_training_weights, task_count, sum_Fx):
    fig = plt.figure()
    position = 1
    for param_index, parameter in enumerate(weights):

        if len(list(parameter.shape)) == 2:
            ax = fig.add_subplot(len(weights) / 2, 2, position, projection='3d')
            x = np.arange(list(parameter.shape)[1])
            y = np.arange(list(parameter.shape)[0])
            X, Y = np.meshgrid(x, y)
            z_data = np.zeros((list(parameter.shape)[0], list(parameter.shape)[1]))

            for row in range(len(z_data)):
                for col in range(len(z_data[row])):
                    for task in range(1, task_count):
                        task_data = task_post_training_weights.get(task)
                        z_data[row][col] += (parameter.data[row][col] - task_data[param_index][row][col]) ** 2
                z_data[row][col] *= sum_Fx[param_index][row][col]

            Z = z_data

            ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                            cmap='viridis', edgecolor='none')

        else:
            ax = fig.add_subplot(len(weights) / 2, 2, position)
            x = np.arange(list(parameter.shape)[0])

            y_data = np.zeros((list(parameter.shape)[0]))


            for index in range(len(y_data)):
                for task in range(1, task_count):
                    task_data = task_post_training_weights.get(task)
                    y_data[index] += (parameter.data[index] - task_data[param_index][index]) ** 2

                y_data[index] *= sum_Fx[param_index][index]

            y = y_data

            ax.plot(x, y)

        position += 1

    plt.show()


"""
X, Y = np.meshgrid(x, y)
Z = np.array([[.25,0,0,.25,0, 0, 0],[0, 0, .25, 0, 0, 0, 0],[0, 0, 1, 0, 0, 0, 0],[0, 0, .25, 0, 0, 0, 0],[0, 0, 0, .25, 0, 0, 0]])

ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

plt.show()
"""
