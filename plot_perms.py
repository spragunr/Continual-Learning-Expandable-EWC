import argparse
from copy import deepcopy
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

DIRECTORY = 'perm_results/plots/' # TODO change this as needed


def parse_h5(filename):
    
    f = h5py.File(filename, 'r')                                                                                            
    
    ewc_pens = []
    avg_accs = []

    for data in f['ewc_pen']:
        ewc_pens.append(data)

    for data in f['avg_acc']:
        avg_accs.append(data)
    
    f.close()

    return ewc_pens, avg_accs 

def plot_line_avg_acc(avg_accuracies, labels):

    plt.figure()
    
    for i, avg_acc in enumerate(avg_accuracies):
        plt.plot(avg_acc, label=labels[i])
    

    plt.ylabel('Average Accuracy on All Tasks')
    plt.xlabel('Total Task Count')
    plt.xlim(1, len(avg_accuracies[0]))

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=3, fancybox=True, shadow=True)

    plt.savefig('{}avg_acc.eps'.format(DIRECTORY), dpi=300, format='eps')


def plot_line_ewc_pen(ewc_pens, labels):

    plt.figure()
    
    for i, ewc_pen in enumerate(ewc_pens):
        plt.plot(ewc_pen, label=labels[i])
    

    plt.ylabel('Elastic Weight Consolidation Loss Penalty')
    plt.xlabel('Total Task Count')
    plt.xlim(1, len(avg_accuracies[0]))

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=3, fancybox=True, shadow=True)

    plt.savefig('{}ewc_pen.eps'.format(DIRECTORY), dpi=300, format='eps')


def main():

    #sns.set(color_codes=True)

    parser = argparse.ArgumentParser(description='Plotting Tool')

    parser.add_argument('--filenames',
            nargs='+', type=str, default=['NONE'], metavar='FILENAMES',
            help='names of .h5 files containing experimental result data')
    
    args = parser.parse_args()
    
    print(args.filenames)

    ewc_pens_list = []
    avg_accs_list = []

    for filename in args.filenames:
        ewc_pens, avg_accs = parse_h5(filename)

        ewc_pens_list.append(ewc_pens)
        avg_accs_list.append(avg_accs)
    
    labels = ['100%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
    
    plot_line_avg_acc(avg_accs_list, labels)

    plot_line_ewc_pen(ewc_pens_list, labels)

if __name__ == '__main__':
    main()
 
