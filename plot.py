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


fig = plt.figure()
ax = fig.gca(projection='3d')

x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 2, 3, 4])

X, Y = np.meshgrid(x, y)
Z = np.array([[.25,0,0,.25,0],[0, 0, .25, 0, 0],[0, 0, 1, 0, 0],[0, 0, .25, 0, 0],[0, 0, 0, .25, 0]])

ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

plt.show()