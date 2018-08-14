import argparse 
import h5py
import matplotlib.pyplot as plt
import numpy as np


def plot_line_avg_acc(avg_accuracies, expansion_markers, threshold, labels, save):

    plt.figure()
    
    for i, avg_acc in enumerate(avg_accuracies):
        plt.plot(avg_acc, label=labels[i])
    

    plt.ylabel('Average Accuracy on All Tasks')
    plt.xlabel('Total Task Count')
    plt.xlim(1, len(avg_accuracies[0]))
    plt.ylim(0, 100)

    for marker in expansion_markers:
        plt.axvline(x=marker, color='r')

    plt.axhline(y=threshold, linestyle='dashed')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=3, fancybox=True, shadow=True)

    plt.savefig('{}.eps'.format(save), dpi=300, format='eps')


def plot_bar_each_task_acc(task_accuracies, labels, save):
    
    plt.figure()

    x_values = np.arange(1, len(task_accuracies[0]) + 1)

    w = 1 / len(task_accuracies)
    offset = w / len(task_accuracies)
    
    adjusted_xs = [x_values - offset, x_values + offset]

    for i, task_acc in enumerate(task_accuracies):
        plt.bar(adjusted_xs[i], width=w, height=task_acc, align='center', edgecolor='k', label=labels[i])
    
    plt.ylabel('Accuracy')
    plt.xlabel('Task')
    plt.xlim(0, len(task_accuracies[0]) + 1)
    plt.ylim(0, 100)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=3, fancybox=True, shadow=True)
    
    plt.savefig('{}.eps'.format(save), dpi=300, format='eps')


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
    """
    NOTE: pass me the name of the file with expansion first...
    """
      
    parser = argparse.ArgumentParser(description='Plotting Tool')
    
    parser.add_argument('--filenames', nargs='+', type=str, default=['NONE'], metavar='FILENAMES',
                        help='names of .h5 files containing experimental result data')

    parser.add_argument('--labels', nargs='+', type=str, default=['NONE'], metavar='LABELS',
                        help='figure legend labels in same order as respective filenames')
    
    parser.add_argument('--line', type=str, default='NO_LINE', metavar='LINE',
                        help='filename for saved line graph (no extension)')
    
    parser.add_argument('--bar', type=str, default='NO_BAR', metavar='BAR',
                        help='filename for saved bar graph (no extension)')
    
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
        if data.startswith('accuracy_threshold'):
            threshold = float(data[data.rfind(' '):])
    
    plot_line_avg_acc(avg_acc_list, expansion_indices_list[0], threshold, args.labels, args.line)
    
    plot_bar_each_task_acc(task_acc_list, args.labels, args.bar)



if __name__ == "__main__":
    main()




