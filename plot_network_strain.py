import argparse
from copy import deepcopy
import h5py
import matplotlib.pyplot as plt
import numpy as np




def plot_failures(failure_points, lowest, highest):
  
    plt.figure()
    
    # plt.xlim(lowest - 0.5, highest + 0.5)

    plt.hist(failure_points, align='left', bins=np.arange(lowest, highest+2), color='orange', edgecolor='k')
    
    plt.show() 

def plot_strain(run_groups):
    
    plt.figure()
    
    for run_group in run_groups:
        plt.plot(run_group[1], label=run_group[0])
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=3, fancybox=True, shadow=True)

    plt.show()

def parse_h5(filename):
    
    f = h5py.File(filename, 'r')                                                                                            
    failure = deepcopy(f['failure'])                                                                                                           
    strain_per_task = []                                                                                                                
    for data in f['strain']:
        strain_per_task.append(data)
    
    f.close()

    return (failure[0], strain_per_task)

def main():


    # parserargparse.ArgumentParser(description='Plotting Tool')
    # 
    # parser.add_argument('--filenames', 
    #         nargs='+', type=str, default=['NONE'], metavar='FILENAMES', 
    #         help='names of .h5 files containing experimental result data')
    # 
    # args = parser.parse_args()
    # 
    # runs = []

    # for filename in args.filenames:
    #     runs.append(parse_h5(filename))

    # failure_points = []

    # for data in runs:
    #     failure_points.append(data[0])
    # 
    
    failure_points = [4, 5, 4, 4, 2, 3, 3, 3, 3, 3, 5, 6, 9, 7]
     
    highest = np.amax(failure_points)
    
    lowest = np.amin(failure_points)
    
    plot_failures(failure_points, lowest, highest)
    
    runs = [
            (3, [0, 2, 9]),
            (5, [0, 3, 4, 5, 9]),
            (5, [0, 2, 6, 8, 10]),
            (3, [0, 1, 8]),
            (9, [0, 3, 4, 5, 6, 7, 10, 15, 23]),
            (3, [0, .5, 7.5])
            ]

    # strain_per_task is an array organized like so:
    # [
    # [0, []]               row 0                     
    # [c1, [x1, x2, x3]]    row 1: (# of runs failed at task 1,
    #                               average network strain per task (index) for runs ending at task 1)
    # [c2, [y1, y2, y3]]    row 2: (# of runs failed at task 2,
    #                               average network strain per task (index) for runs ending at task 2)
    # ...                            
    # ]
    strain_per_task = []

    for i in np.arange(0, highest+1):
        strain_per_task.append([0, np.zeros(i)])
    
    for data in runs:
        
        for t in range(len(data[1])):
            strain_per_task[data[0]][1][t] += data[1][t]
        
        strain_per_task[data[0]][0] += 1
    
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

    plot_strain(run_groups)

if __name__ == '__main__':
    main()

