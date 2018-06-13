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

    for param_index, parameter in enumerate(weights):
        if len(list(parameter.shape)) == 2:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            y = np.arange(list(parameter.shape)[0])
            x = np.arange(list(parameter.shape)[1])
            print(x.shape)
            print(y.shape)
            X, Y = np.meshgrid(x, y)
            z_data = np.zeros((list(parameter.shape)[0], list(parameter.shape)[1]))

            for row in range(len(z_data)):
                for col in range(len(z_data[row])):
                    for task in range(task_count):
                        z_data[row][col] += (parameter.data[row][col] - task_post_training_weights.get(task)[parameter][row][col]) ** 2
                z_data[row][col] *= sum_Fx[param_index][row][col]

            Z = z_data

            ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                            cmap='viridis', edgecolor='none')

            plt.show()



"""
X, Y = np.meshgrid(x, y)
Z = np.array([[.25,0,0,.25,0, 0, 0],[0, 0, .25, 0, 0, 0, 0],[0, 0, 1, 0, 0, 0, 0],[0, 0, .25, 0, 0, 0, 0],[0, 0, 0, .25, 0, 0, 0]])

ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

plt.show()
"""
