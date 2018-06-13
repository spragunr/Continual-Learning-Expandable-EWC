import matplotlib.pyplot as plt
import numpy as np



def plot(weights, task_post_training_weights, task_count, task_fisher_diags):
    fig = plt.figure()
    fig.suptitle('Task {}'.format(task_count))

    position = 1
    weight_count = 1
    bias_count = 1

    z_limits = [.00007, .000020, .0002]
    y_limits = [.0000016, .0000035, .00001]

    for param_index, parameter in enumerate(weights):

        if len(list(parameter.shape)) == 2:
            ax = fig.add_subplot(len(weights) / 2, 2, position, projection='3d')
            ax.set_title('weights layer {}'.format(weight_count))
            #ax.set_zlim(0, z_limits[weight_count - 1])
            ax.set_zlabel(r'$\sum_{task=1}^{T-1} F_{task,\theta}(\theta - \theta_{task})^2$')
            weight_count += 1
            x = np.arange(list(parameter.shape)[1])
            y = np.arange(list(parameter.shape)[0])
            X, Y = np.meshgrid(x, y)
            z_data = np.zeros((list(parameter.shape)[0], list(parameter.shape)[1]))

            for row in range(len(z_data)):
                for col in range(len(z_data[row])):
                    for task in range(1, task_count):
                        task_weights = task_post_training_weights.get(task)
                        task_fisher = task_fisher_diags.get(task)
                        z_data[row][col] += ((parameter.data[row][col] - task_weights[param_index][row][col]) ** 2) \
                                            * task_fisher[param_index][row][col]

            Z = z_data

            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='inferno')

        else:
            ax = fig.add_subplot(len(weights) / 2, 2, position)
            ax.set_title('bias layer {}'.format(bias_count))
            #ax.set_ylim(0, y_limits[bias_count - 1])
            ax.set_ylabel(r'$\sum_{task=1}^{T-1} F_{task,\theta}(\theta - \theta_{task})^2$')
            bias_count += 1
            x = np.arange(list(parameter.shape)[0])

            y_data = np.zeros((list(parameter.shape)[0]))


            for index in range(len(y_data)):
                for task in range(1, task_count):
                    task_weights = task_post_training_weights.get(task)
                    task_fisher = task_fisher_diags.get(task)
                    y_data[index] += ((parameter.data[index] - task_weights[param_index][index]) ** 2) * task_fisher[param_index][index]

            y = y_data

            ax.plot(x, y)

        position += 1

    fig.set_size_inches(36, 17)
    fig.savefig('../../task{}.png'.format(task_count))


