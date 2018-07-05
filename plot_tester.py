import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker


mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
z = np.array([90, 78, 90, 90, 78, 90, 90, 78, 90])
x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])


ax.plot(x, y, z, label='small')
z = np.array([89, 90, 80, 90, 89, 90, 80, 90])
x = np.array([1, 1, 1, 1, 1, 1, 1, 1])
y = np.array([10, 11, 12, 13, 14, 15, 16, 17])
ax.plot(x, y, z, label='medium')

z = np.array([95, 92, 90, 79, 90, 95, 92, 90, 79, 90])
x = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
y = np.array([18, 19, 20, 21, 22, 23, 24, 25, 26, 27])
ax.plot(x, y, z, label='large')


ax.legend()

ax.set_zlim(0, 100)

loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)

plt.show()