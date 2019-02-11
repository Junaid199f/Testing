import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Function to convert batch input data to use scan ops of tensorflow.
def process_batch_input_for_RNN(batch_input):
    """
    Process tensor of size [5,3,2] to [3,5,2]
    """
    batch_input_ = tf.transpose(batch_input, perm=[2, 0, 1])
    X = tf.transpose(batch_input_)

    return X

def one_hot(data):
    """Convert an iterable of indices to one-hot encoded labels."""
    nb_classes = np.unique(data).shape[0]
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def bar_plot(objects, mean_accuracies, std_accuracies, accuracies):
    """
    Show bar plot comparing results for object in objects
    """
    y_pos = np.arange(len(objects))
    plt.figure(figsize=(12,8))
    plt.rc('font', size=20)
    plt.bar(y_pos, mean_accuracies, align='center', alpha=0.5, yerr=std_accuracies, width=1/3)
    plt.xticks(y_pos, objects)
    plt.ylabel('Accuracies in %')
    plt.title('Training Mean Test Accuracies')
    plt.text(0.18, 80, 'Max LSTM test accuracy:\n' + str(np.amax(accuracies[0])))
    plt.text(0.18, 70, 'Max subLSTM test accuracy:\n' + str(np.amax(accuracies[1])))
    plt.text(0.18, 40, 'Mean LSTM test accuracy:\n' + str(np.amax(mean_accuracies[0])))
    plt.text(0.18, 30, 'Mean subLSTM test accuracy:\n' + str(np.amax(mean_accuracies[1])))