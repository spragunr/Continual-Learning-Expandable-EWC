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

# Make data.
X = np.arange(1, 11, 1)
Y = np.arange(1, 11, 1)
X, Y = np.meshgrid(X,Y)
list = []

for num in range(10):
    list.append(np.random.randn(10))

Z = np.array(list)
print(Z.shape)
# Plot the surface.
ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)


plt.show()