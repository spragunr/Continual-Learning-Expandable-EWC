import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_line_avg_acc(avg_accuracies, expansion_markers, threshold, label1, avg_accuracies2=None, label2=None):

    plt.figure()

    plt.plot(avg_accuracies, color='c', label=label1)
    if avg_accuracies2 is not None:
        plt.plot(avg_accuracies2, color='orange', label=label2)

    plt.ylabel('Average Accuracy on All Tasks')
    plt.xlabel('Total Task Count')
    plt.xlim(1, len(avg_accuracies))
    plt.ylim(0, 100)

    for marker in expansion_markers:
        plt.axvline(x=marker, color='r')

    plt.axhline(y=threshold, color='b', linestyle='dashed')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=3, fancybox=True, shadow=True)

    plt.savefig('avg_accs.eps', dpi=300, format='eps')

def plot_bar_each_task_acc(single_task_accuracies1, label1, single_task_accuracies2=None, label2=None):
    plt.figure()

    x_values = np.arange(1, len(single_task_accuracies1) + 1)

    w = 0.5

    plt.bar(x_values-0.25, width=w, height=single_task_accuracies1, align='center', color='c', edgecolor='k', label=label1)
    if single_task_accuracies2 is not None:
        plt.bar(x_values+0.25, width=w, height=single_task_accuracies2, align='center', color='orange', edgecolor='k', label=label2)

    plt.ylabel('Accuracy')
    plt.xlabel('Task')
    plt.xlim(0, len(single_task_accuracies1) + 1)
    plt.ylim(0, 100)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=3, fancybox=True, shadow=True)
    plt.savefig('final_per_task_acc.eps', dpi=300, format='eps')

def plot_line_compare_avg_accs(avg_accuracies, labels):

    for index, data in enumerate(avg_accuracies):
        plt.plot(data, label=labels[index])

    plt.ylabel('Average Accuracy on All Tasks')
    plt.xlabel('Total Task Count')
    plt.xlim(1, 100)
    plt.ylim(0, 100)
    plt.legend()

    plt.show()

def plot_wireframe_weight_surface(weights, minz, maxz):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X = np.arange(0, len(weights[0]))
    Y = np.arange(0, len(weights))
    X, Y = np.meshgrid(X, Y)
    Z = weights

    #ax.scatter(X, Y, Z, marker='.')

    upper_bound = np.amax(weights, axis=(0, 1))

    Z = np.full(weights.shape, upper_bound)

    #ax.plot_wireframe(X, Y, Z, rstride=10, cstride=100, color='g')

    lower_bound = np.amin(weights, axis=(0, 1))

    Z = np.full(weights.shape, lower_bound)

    #ax.plot_wireframe(X, Y, Z, rstride=10, cstride=100, color='g')

    ax.plot([0, 0], [0, 0], [upper_bound, lower_bound], marker="_", color='r')

    ax.plot([0], [0], [0], 'o', color='white', markersize=23)
    ax.plot([0], [0], [0], 'k', marker="$%5.04s$" % str(upper_bound - lower_bound), color='b',
                 markersize=21)

    ax.set_zlim(minz, maxz)

    plt.show()


# to demonstrate figure appearances
if __name__ == "__main__":

    avg_accs = np.random.randint(80, 96, 100)
    expansion_tasks = [4, 7, 12, 25, 30, 45, 60, 75, 80, 95]

    plot_line_avg_acc(avg_accs, expansion_tasks, 80)

    single_task_accuracies = np.random.randint(80, 96, 100)

    plot_bar_each_task_acc(single_task_accuracies)

    avg_accs_compare = np.random.randint(50, 96, (3, 100))
    labels = ['ewc', 'gem', 'sgd']

    plot_line_compare_avg_accs(avg_accs_compare, labels)

    weight_values = np.random.uniform(-2, 2, (50, 784))

    plot_wireframe_weight_surface(weight_values, -3, 3)