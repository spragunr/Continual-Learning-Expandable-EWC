import argparse
from copy import deepcopy
import h5py
import matplotlib.pyplot as plt
import numpy as np



def plot_failures(failure_points, lowest, highest):
  
    plt.figure()
    
    plt.xlim(lowest, highest)

    plt.hist(failure_points)
    
    plt.show() 

def plot_strain():


def average_strain():


def parse_h5(filename):
    
    f = h5py.File(filename, 'r')                                                                                            
    failure = deepcopy(f['failure'])                                                                                                           
    strain_per_task = []                                                                                                                
    for data in f['strain']:
        strain_per_task.append(data)
    
    f.close()

    return (failure, strain_per_task)

def main():


    parserargparse.ArgumentParser(description='Plotting Tool')
    
    parser.add_argument('--filenames', 
            nargs='+', type=str, default=['NONE'], metavar='FILENAMES', 
            help='names of .h5 files containing experimental result data')
    
    args = parser.parse_args()
    
    runs = []

    for filename in args.filenames:
        runs.append(parse_h5(filename))

    failure_points = []

    for data in runs:
        failure_points.append(data[0])
    
    highest = np.amax(failure_points)
    
    lowest = np.amin(failure_points)

    plot_failures(failure_points, lowest, highest)
