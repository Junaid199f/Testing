# Nurture.ai Global NIPS Paper Implementation Challenge

This repo is used to replicate the results from the paper [«Cortical microcircuits as gated-recurrent neural networks»](http://papers.nips.cc/paper/6631-cortical-microcircuits-as-gated-recurrent-neural-networks) by R. Costa et al. (2017), in the scope of the [Nurture.ai Global NIPS Paper Implementation Challenge](https://nurture.ai/nips-challenge). Currently, the results are replicated on the MNIST dataset for one layer LSTM and subLSTM. Moreover, results have been added with two layers for each of those networks.

## Getting Started

You can find the anaylsis in the following notebooks:  
1-layer: [A001-MNIST-1layer.ipynb](https://github.com/michael1788/subLSTM/blob/master/notebooks/A001-MNIST-1layer.ipynb)  
2-layers: [A002-MNIST-2layers.ipynb](https://github.com/michael1788/subLSTM/blob/master/notebooks/A002-MNIST-2layers.ipynb)


### Prerequisites

Each training was done on AWS on a p2.xlarge instance with the Deep Learning AMI using Tensorflow and python 3.5.2  
If you run the notebooks without a GPU, you might want to do only one simulation (we ran 5 simulations) as well as decrease the number of epochs. Traning time are reported in the notebooks.

## Paper insight – difference between LSTM and subLSTM equations:

![alt text](https://github.com/michael1788/subLSTM/blob/master/img/lstm_vs_sublstm.jpg "LSTM vs subLSTM equations. Equations from the paper.")
Screenshot from the paper.  

**f**: forget gate   
**c**: memory cell  
**h**: LSTM state  
**i**: input  
**z**: new weighted input  
Big o dot symbol: element-wise multiplication

## Results
We trained our models on a maximum of 150k epochs with early stopping (with a patience of 25, with a patience unit increased by looking at the test accuracy every 1000 epochs). Moreover, we ran 5 simulations, i.e. we did 5 times the training process, and reported the simulation average and standard deviation as well as the best epoch of each simulation. It has to be noted that we used the traning and test sets as provided by the Tensorflow MNSIT example tutorial. Finally, we trained our models without momentum (they did in the paper, but no values given).

**Best epoch results accross all simulations:**

| 1 layer networks  |      LSTM     |   subLSTM     |
| :------------:    | :-----------: | :-----------: |
| Paper             |    97.96 %    |    97.29 %    |
| Here              |    98.08 %    |    97.86 %    |

Training times and bar plots can be found in the notebook. Simulations parameters are in the notebooks as well. Network parameters as define in the paper can be found in the file [parameter.py](https://github.com/michael1788/subLSTM/blob/master/models/parameters.py)

| 2 layers networks |      LSTM     |   subLSTM     |
| :------------:    | :-----------: | :-----------: |
| Here              |    95.35 %    |    96.95 %    |

**Mean results accross all simulations:**

| 1 layer networks  |      LSTM     |   subLSTM     |
| :------------:    | :-----------: | :-----------: |
| Here              |    94.10 %    |    94.04 %    |


| 2 layers networks |      LSTM     |   subLSTM     |
| :------------:    | :-----------: | :-----------: |
| Here              |    95.25 %    |    88.83 %    |

### Comment on the results
For the 1 layer networks, we found similar results, with a difference in test accuracy between the LSTM and the subLSTM even smaller. Concering the 2 layers networks (not tested in the paper), results are better for the subLSTM for the best value, but much lower for the mean over 5 simulations. It has to be noted that the accuracies are lower for the 1 layer network compare to the 2 layers network because the 2 layers might need more training time.

## TODO

Replicate the paper's results on the Penn Treebank and WikiText-10 datasets.  
Write the tests.  
Ask the authors about the momentum values.

## Authors

* **Michael Stettler**  - [github](https://github.com/michaelStettler)
* **Michael Moret** - [github](https://github.com/michael1788)

## License

This project is licensed under the MIT License.

## Acknowledgments

* The LSTM and subLSTM models were modified from https://github.com/KnHuq/Dynamic-Tensorflow-Tutorial/blob/master/LSTM/LSTM.py
* The training methodology was based on the r2rt LSTM tutorial (https://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html)
