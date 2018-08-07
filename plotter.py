import argparse 
import h5py
import matplotlib.pyplot as plt
import numpy as np


def plot_line_avg_acc(avg_accuracies, metadata, expansion_markers, threshold):

    plt.figure()
    
    for i, avg_acc in enumerate(avg_accuracies):
        plt.plot(avg_acc, label=metadata[i])
    

    plt.ylabel('Average Accuracy on All Tasks')
    plt.xlabel('Total Task Count')
    plt.xlim(1, len(avg_accuracies))
    plt.ylim(0, 100)

    for marker in expansion_markers:
        plt.axvline(x=marker)

    plt.axhline(y=threshold, linestyle='dashed')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=3, fancybox=True, shadow=True)
    
    plt.show()

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
    
    plt.show()
    
    plt.savefig('final_per_task_acc.eps', dpi=300, format='eps')


def parse_h5_file(filename):
    
    f = h5py.File(filename, 'r')
    
    avg_acc = []
    task_acc = []
    expansions = []
    metadata = []
    
    for data in f["avg_acc"]:
    
        avg_acc.append(data)
    
    
    for data in f["task_acc"]:
    
        task_acc.append(data)


    for data in f["expansions"]:
    
        expansions.append(data)
    
    for data in f["metadata"]:
        
        metadata.append(data)
    
    f.close()
    
    expansion_indices = []
    
    for i in range(len(expansions)):
        if expansions[i] == 1:
            expansion_indices.append(i)
        elif expansions[i] > 1:
            print("You'd better take a look at {}, Captain...".format(filename))

    return avg_acc, task_acc, expansion_indices, metadata

def main():
    
    
    parser = argparse.ArgumentParser(description='Plotting Tool')
    
    parser.add_argument('--filenames', nargs='+', type=str, default=['NONE'], metavar='FILENAMES',
                        help='names of .h5 files containing experimental result data')

    args = parser.parse_args()


    avg_acc_list = []
    task_acc_list = []
    expansion_indices_list = []
    metadata_list = []

    for filename in args.filenames:
        
        avg_acc, task_acc, expansion_indices, metadata = parse_h5_file(filename)
        
        avg_acc_list.append(avg_acc)
        task_acc_list.append(task_acc)
        expansion_indices_list.append(expansion_indices)
        metadata_list.append(metadata)
    
    threshold = 0

    for data in metadata_list[0]:
        if data.startswith('threshold'):
            threshold = float(data[datadata.rfind(' '):])
    
    # just for testing... 
    print(avg_acc_list[0], expansion_indices_list[0], task_acc_list[0], metadata_list[0])
    print(threshold)

if __name__ == "__main__":
    main()




