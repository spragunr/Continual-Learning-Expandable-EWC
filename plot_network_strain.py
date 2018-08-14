import argparse
from copy import deepcopy
import h5py
import matplotlib.pyplot as plt
import numpy as np




def plot_failures(failure_points, lowest, highest):
  
    plt.figure()
    
    # plt.xlim(lowest - 0.5, highest + 0.5)

    plt.hist(failure_points, align='left', bins=np.arange(lowest, highest+2), color='orange', edgecolor='k')

    plt.xlabel('Task')

    plt.show() 

def plot_strain(run_groups, metric):
    
    plt.figure()
    
    for run_group in run_groups:
        plt.plot(run_group[1], label=run_group[0])
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=3, fancybox=True, shadow=True)

    plt.ylabel(metric)
    plt.xlabel('Task')

    plt.show()

def parse_h5(filename):
    
    f = h5py.File(filename, 'r')                                                                                            
    failure = f['failure'][0]

    total = []
    st_dev = []
    avg = []
    maximum = []
    loss = []
    fisher_information = []

    for data in f['fisher_total']:
        total.append(data)

    for data in f['post_training_loss']:
        loss.append(data)

    for data in f['fisher_average']:
        avg.append(data)

    for data in f['fisher_st_dev']:
        st_dev.append(data)

    for data in f['fisher_max']:
        maximum.append(data)

    for data in f['fisher_information']:
        fisher_information.append([])
        for task in data:
            fisher_information[len(fisher_information) - 1].append(task)

    f.close()

    return (failure, total), (failure, st_dev), (failure, avg), (failure, maximum), (failure, loss), (failure, fisher_information)


def plot_fisher_dist(run_group):

    plt.figure()

    tasks = []

    for fi_data in run_group[1]:
        tasks.append(fi_data)

    plt.hist(fi_data, label=np.arange(0, run_group[0] + 1))

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=3, fancybox=True, shadow=True)

    plt.show()


def main():


    parser = argparse.ArgumentParser(description='Plotting Tool')

    parser.add_argument('--filenames',
            nargs='+', type=str, default=['NONE'], metavar='FILENAMES',
            help='names of .h5 files containing experimental result data')

    args = parser.parse_args()

    runs = []

    for filename in args.filenames:
        runs.append([])
        total, st_dev, avg, maximum, loss, fisher_information = parse_h5(filename)
        runs[len(runs) - 1].append(total)
        runs[len(runs) - 1].append(st_dev)
        runs[len(runs) - 1].append(avg)
        runs[len(runs) - 1].append(maximum)
        runs[len(runs) - 1].append(loss)
        runs[len(runs) - 1].append(fisher_information)

    failure_points = []

    for data in runs:
        failure_points.append(data[0][0])

    highest = np.amax(failure_points)
    
    lowest = np.amin(failure_points)
    
    plot_failures(failure_points, lowest, highest)


    metrics = ['Sum of Fisher Information','Standard Deviation of Fisher Information','Average of Fisher Information',
               'Maximum Fisher Information Value','Final Training Iteration Loss']

    for strain_metric in range(5):
        # strain_per_task is an array organized like so:
        # [
        # [0, []]               row 0
        # [c1, [x1, x2, x3]]    row 1: [# of runs failed at task 1,
        #                               average network strain per task (index) for runs ending at task 1]
        # [c2, [y1, y2, y3]]    row 2: [# of runs failed at task 2,
        #                               average network strain per task (index) for runs ending at task 2]
        # ...
        # ]
        strain_per_task = []

        for i in np.arange(0, highest+1):
            strain_per_task.append([0, np.zeros(i)])

        for data in runs:

            for t in range(len(data[strain_metric][1])):
                strain_per_task[data[strain_metric][0]][1][t] += data[strain_metric][1][t]

            strain_per_task[data[strain_metric][0]][0] += 1

        for row in range(len(strain_per_task)):
            for i in range(len(strain_per_task[row][1])):
                if strain_per_task[row][0] != 0:
                    strain_per_task[row][1][i] /= strain_per_task[row][0]

        print(strain_per_task)

        run_groups = []

        for row in range(len(strain_per_task)):
            if strain_per_task[row][0] > 0:
                run_groups.append((row, strain_per_task[row][1]))

        print(run_groups)

        metric = metrics[strain_metric]
        plot_strain(run_groups, metric)

    # plot summed fisher info distribution
    fisher_summed = []

    for i in np.arange(0, highest + 1):
        fisher_summed.append([0, np.zeros(i, len(runs[0][5][1][1]))])

    for data in runs:
        for t in range(len(data[5][1])):
            for fi in range(len(data[5][1][t])):
                fisher_summed[data[5][0]][1][t][fi] += data[5][1][t][fi]

        fisher_summed[data[5][0]][0] += 1

    # average fisher summed info for each task for each run group
    for row in range(len(fisher_summed)):
        for task in range(len(fisher_summed[row][1])):
            for fisher_info in range(len(task)):
                if fisher_summed[row][0] != 0:
                    fisher_summed[row][1][task][fisher_info] /= fisher_summed[row][0]

### YOU WERE HERE>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>###


    run_groups = []

    for row in range(len(fisher_summed)):
        if fisher_summed[row][0] > 0:
            run_groups.append((row, fisher_summed[row][1]))

    # run groups is now [...(failure_point, [[fisher info task 0][fi t1][fi t2]...])...]
    for group in run_groups:
        plot_fisher_dist(group)

if __name__ == '__main__':
    main()

