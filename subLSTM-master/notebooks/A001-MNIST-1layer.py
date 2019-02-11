
# coding: utf-8

# # Sequential MNIST results from the paper by Rui Costa et al.:<br/>"Cortical microcircuits as gated-recurrent neural networks" 
# ## Implementation done in the scope of the nurture.ai NIPS 2017 paper implementation challenge

# - nurture.ai challenge: https://nurture.ai/nips-challenge
# - Paper: http://papers.nips.cc/paper/6631-cortical-microcircuits-as-gated-recurrent-neural-networks
# - Credits:<br/>
#  Training logic based on the r2rt LSTM tutorial (https://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html).<br/>
#  Model definition based on KnHuq implementation (https://github.com/KnHuq/Dynamic-Tensorflow-Tutorial/blob/master/LSTM/LSTM.py
# ).

# ## This notebook compare the results of models with 1 layer (as done in the paper)

# ### Loading Librairies and Models

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import sys

#import LSTM and subLSTM cell models
sys.path.append('../models/')
from LSTM import *
from subLSTM import *
from parameters import *

sys.path.append('../src/common/')
import helper as hp


# ### Loading MNIST dataset

# In[2]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# ### Training Function

# In[ ]:


def train_network(g, batch_size=50, n_epoch=10, verbose=False, save=False, patience=25, min_delta=0.01):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # parameters for early stopping
        patience_cnt = 0
        max_test_accuracy = 0.0

        #Iterations to do trainning
        for epoch in range(n_epoch):

            X, Y = mnist.train.next_batch(batch_size)
            X = X.reshape(batch_size, 1, g['input_size'])

            sess.run(g['train_step'],feed_dict={g['rnn']._inputs:X, g['y']:Y})
               
            if epoch % 1000 == 0:
                Loss=str(sess.run(g['cross_entropy'],feed_dict={g['rnn']._inputs:X, g['y']:Y}))
                Train_accuracy=str(sess.run(g['accuracy'],feed_dict={g['rnn']._inputs:X, g['y']:Y}))
                X_test = mnist.test.images.reshape(10000,1,g['input_size'])
                Test_accuracy=str(sess.run(g['accuracy'],feed_dict={g['rnn']._inputs:X_test, g['y']:mnist.test.labels}))
                if verbose:
                    print("\rIteration: %s Loss: %s Train Accuracy: %s Test Accuracy: %s"%(epoch,Loss,Train_accuracy,Test_accuracy))
                    
                # early stopping
                if float(Test_accuracy) > max_test_accuracy:
                    max_test_accuracy = float(Test_accuracy)
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                if patience_cnt > patience:
                    print("early stopping at epoch: ", epoch)
                    break
                        
                if isinstance(save, str):
                    g['saver'].save(sess, save)
    
    return max_test_accuracy


# ### Building Graph Model Function

# In[ ]:


def build_graph(cell_type=None, load_parameters=False):
    # define initial parameters
    input_size = 784
    output_size = 10
    optimizer = 'Adam'
    momentum = False
    learning_rate = 0.001
    hidden_units = 10
    
    if load_parameters:
        #load parameters from file
        if cell_type == 'LSTM':
            parameters = LSTM_parameters()
        elif cell_type == 'sub_LSTM':
            parameters = SubLSTM_parameters()
        elif cell_type == 'fix_sub_LSTM':
            parameters = Fix_subLSTM_parameters()
        else:
            print("No cell_type selected! Use LSTM cell")
            parameters = LSTM_parameters()
        
        input_size = parameters.mnist['input_size']
        output_size = parameters.mnist['output_size']
        optimizer = parameters.mnist['optimizer']
        momentum = parameters.mnist['momentum']
        learning_rate = parameters.mnist['learning_rate']
        hidden_units = parameters.mnist['hidden_units']

    # reset graph
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()
    
    # Initializing rnn object
    if cell_type == 'LSTM':
        rnn = LSTM_cell(input_size, hidden_units, output_size)
    elif cell_type == 'sub_LSTM':
        rnn = subLSTM_cell(input_size, hidden_units, output_size)
    elif cell_type == 'fix_sub_LSTM':
        print("TODO!")
    else:
        rnn = LSTM_cell(input_size, hidden_units, output_size)
    
    #input label placeholder
    y = tf.placeholder(tf.float32, [None, output_size])
    
    # Getting all outputs from rnn
    outputs = rnn.get_outputs()

    # Getting final output through indexing after reversing
    last_output = outputs[-1]

    # As rnn model output the final layer through Relu activation softmax is
    # used for final output
    output = tf.nn.softmax(last_output)

    # Computing the Cross Entropy loss
    cross_entropy = -tf.reduce_sum(y * tf.log(output))

    # setting optimizer
    if optimizer == 'Adam':
        # Trainning with Adam Optimizer
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    elif optimizer == 'RMSProp':
        # Trainning with RMSProp Optimizer
        train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(cross_entropy)
    else:
        #if nothing is define use Adam optimizer
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # Calculation of correct prediction and accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
    accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100
      
    return dict(
        rnn = rnn,
        y = y,
        input_size = input_size,
        output = output,
        cross_entropy = cross_entropy,
        train_step = train_step,
        preds = correct_prediction,
        accuracy = accuracy,
        saver = tf.train.Saver()
    ) 


# ### Simulation Parameters

# In[ ]:


n_simulation = 5
batch_size = 10
n_epoch = 20


# ### LSTM training

# In[ ]:


get_ipython().run_cell_magic('time', '', 'lstm_accuracies = []\nprint(\'Traning begins for: \', n_simulation, \' simulation(s)\')\n\nfor n in range(n_simulation):\n    print(\'simulation \', n, \' running\')\n    g = build_graph(cell_type=\'LSTM\', load_parameters=True)\n    test_accuracy = train_network(g, batch_size, n_epoch, verbose=False)\n    lstm_accuracies.append(test_accuracy)\n\nlstm_mean_accuracy = np.mean(lstm_accuracies)\nlstm_std_accuracy = np.std(lstm_accuracies)\nlstm_best_accuracy = np.amax(lstm_accuracies)\n\nprint("The mean test accuracy of the simulation is:", lstm_mean_accuracy)\nprint("the standard deviation is:", lstm_std_accuracy)\nprint("The best test accuracy obtained was:", lstm_best_accuracy)')


# ### SubLSTM training

# In[ ]:


get_ipython().run_cell_magic('time', '', 'sub_lstm_accuracies = []\nprint(\'Traning begins for: \', n_simulation, \' simulation(s)\')\n\nfor n in range(n_simulation):\n    print(\'simulation \', n, \' running\')\n    g = build_graph(cell_type=\'sub_LSTM\', load_parameters=True)\n    test_accuracy = train_network(g, batch_size, n_epoch, verbose = False)\n    sub_lstm_accuracies.append(test_accuracy)\n\nsub_lstm_mean_accuracy = np.mean(sub_lstm_accuracies)\nsub_lstm_std_accuracy = np.std(sub_lstm_accuracies)\nsub_lstm_best_accuracy = np.amax(sub_lstm_accuracies)\n\nprint("The mean test accuracy of the simulation is:", sub_lstm_mean_accuracy)\nprint("the standard deviation is:", sub_lstm_std_accuracy)\nprint("The best test accuracy obtained was:", sub_lstm_best_accuracy)')


# ### Plot test mean accuracies and std

# In[ ]:


objects = ('LSTM', 'SubLSTM')
mean_accuracies = [lstm_mean_accuracy, sub_lstm_mean_accuracy]
std_accuracies = [lstm_std_accuracy, sub_lstm_std_accuracy]
accuracies = [lstm_accuracies, sub_lstm_accuracies]


# In[ ]:


hp.bar_plot(objects, mean_accuracies, std_accuracies, accuracies)

