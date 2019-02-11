"""
Parameters used in «Cortical microcircuits as gated-recurrent neural networks» paper.
"""


class SubLSTM_parameters():

    """
    Define all parameters used in the paper
    for the subLSTM network.

    """

    def __init__(self):
        
        self.mnist = {
                      'input_size': 784,
                      'output_size': 10,
                      'hidden_layer_size': 1,
                      'optimizer': 'RMSProp',
                      'momentum': True,
                      'learning_rate': 1e-4,
                      'hidden_units': 100
                     }
        
        self.penn_treebank_10 ={
                      'input_size': None, 
                      'hidden_layer_size': 2,
                      'input_dropout': 0.012,
                      'output_dropout': 0.045,
                      'update_dropout': 0.438,
                      'learning_rate': 0.01666,
                      'weight_decay': 0.000009,
                      'hidden_units': 10
                     }
        
        self.penn_treebank_100 ={
                      'input_size': None, 
                      'hidden_layer_size': 2,
                      'input_dropout': 0.392,
                      'output_dropout': 0.051,
                      'update_dropout': 0.246,
                      'learning_rate': 0.01186,
                      'weight_decay': 0.000157,
                      'hidden_units': 100
                     }
        
        self.penn_treebank_200 ={
                      'input_size': None, 
                      'hidden_layer_size': 2,
                      'input_dropout': 0.337,
                      'output_dropout': 0.373,
                      'update_dropout': 0.439,
                      'learning_rate': 0.01534,
                      'weight_decay': 0.000076,
                      'hidden_units': 200
                     }
        
        self.penn_treebank_650 ={
                      'input_size': None, 
                      'hidden_layer_size': 2,
                      'input_dropout': 0.562,
                      'output_dropout': 0.515,
                      'update_dropout': 0.794,
                      'learning_rate': 0.00301,
                      'weight_decay': 0.000227,
                      'hidden_units': 650
                     }
        
        self.wikitext_10 ={
                      'input_size': None, 
                      'hidden_layer_size': 2,
                      'input_dropout': 0.002,
                      'output_dropout': 0.030,
                      'update_dropout': 0.390,
                      'learning_rate': 0.00859,
                      'weight_decay': 0.000013,
                      'hidden_units': 10
                     }
        
        self.wikitext_100 ={
                      'input_size': None, 
                      'hidden_layer_size': 2,
                      'input_dropout': 0.172,
                      'output_dropout': 0.150,
                      'update_dropout': 0.009,
                      'learning_rate': 0.00635,
                      'weight_decay': 0.000177,
                      'hidden_units': 100
                     }
        
        self.wikitext_200 ={
                      'input_size': None, 
                      'hidden_layer_size': 2,
                      'input_dropout': 0.342,
                      'output_dropout': 0.269,
                      'update_dropout': 0.018,
                      'learning_rate': 0.00722,
                      'weight_decay': 0.000111,
                      'hidden_units': 200
                     }
        
        self.wikitext_650 ={
                      'input_size': None, 
                      'hidden_layer_size': 2,
                      'input_dropout': 0.633,
                      'output_dropout': 0.567,
                      'update_dropout': 0.257,
                      'learning_rate': 0.00300,
                      'weight_decay': 0.000142,
                      'hidden_units': 650
                     }
        
class Fix_subLSTM_parameters():

    """
    Define all parameters used in the paper
    for the fix_subLSTM network.

    """

    def __init__(self):
        
        self.mnist = {
                      'input_size': 784, 
                      'output_size': 10,
                      'hidden_layer_size': 1,
                      'optimizer': 'RMSProp',
                      'momentum': True,
                      'learning_rate': 1e-4,
                      'hidden_units': 100
                     }
        
        self.penn_treebank_10 ={
                      'input_size': None, 
                      'hidden_layer_size': 2,
                      'input_dropout': 0.009,
                      'output_dropout': 0.043,
                      'update_dropout': 0,
                      'learning_rate': 0.01006,
                      'weight_decay': 0.000029,
                      'hidden_units': 11
                     }
        
        self.penn_treebank_100 ={
                      'input_size': None, 
                      'hidden_layer_size': 2,
                      'input_dropout': 0.194,
                      'output_dropout': 0.148,
                      'update_dropout': 0.042,
                      'learning_rate': 0.00400,
                      'weight_decay': 0.000218,
                      'hidden_units': 115
                     }
        
        self.penn_treebank_200 ={
                      'input_size': None, 
                      'hidden_layer_size': 2,
                      'input_dropout': 0.394,
                      'output_dropout': 0.472,
                      'update_dropout': 0.161,
                      'learning_rate': 0.00382,
                      'weight_decay': 0.000066,
                      'hidden_units': 230
                     }
        
        self.penn_treebank_650 ={
                      'input_size': None, 
                      'hidden_layer_size': 2,
                      'input_dropout': 0.662,
                      'output_dropout': 0.730,
                      'update_dropout': 0.530,
                      'learning_rate': 0.00347,
                      'weight_decay': 0.000136,
                      'hidden_units': 750
                     }
        
        self.wikitext_10 ={
                      'input_size': None, 
                      'hidden_layer_size': 2,
                      'input_dropout': 0.033,
                      'output_dropout': 0.070,
                      'update_dropout': 0.013,
                      'learning_rate': 0.00875,
                      'weight_decay': 0,
                      'hidden_units': 11
                     }
        
        self.wikitext_100 ={
                      'input_size': None, 
                      'hidden_layer_size': 2,
                      'input_dropout': 0.130,
                      'output_dropout': 0.187,
                      'update_dropout': 0,
                      'learning_rate': 0.00541,
                      'weight_decay': 0.000172,
                      'hidden_units': 115
                     }
        
        self.wikitext_200 ={
                      'input_size': None, 
                      'hidden_layer_size': 2,
                      'input_dropout': 0.256,
                      'output_dropout': 0.273,
                      'update_dropout': 0,
                      'learning_rate': 0.00533,
                      'weight_decay': 0.000160,
                      'hidden_units': 230
                     }
        
        self.wikitext_650 ={
                      'input_size': None, 
                      'hidden_layer_size': 2,
                      'input_dropout': 0.656,
                      'output_dropout': 0.590,
                      'update_dropout': 0.711,
                      'learning_rate': 0.00321,
                      'weight_decay': 0.000122,
                      'hidden_units': 750
                     }
        
        
class LSTM_parameters():

    """
    Define all parameters used in the paper
    used for the LSTM network.

    """

    def __init__(self):
        
        self.mnist = {
                      'input_size': 784, 
                      'output_size': 10,
                      'hidden_layer_size': 1,
                      'optimizer': 'RMSProp',
                      'momentum': True,
                      'learning_rate': 1e-4,
                      'hidden_units': 100
                     }
        
        self.penn_treebank_10 ={
                      'input_size': None, 
                      'hidden_layer_size': 2,
                      'input_dropout': 0.026,
                      'output_dropout': 0.047,
                      'update_dropout': 0.002,
                      'learning_rate': 0.01186,
                      'weight_decay': 0.000020,
                      'hidden_units': 10
                     }
        
        self.penn_treebank_100 ={
                      'input_size': None, 
                      'hidden_layer_size': 2,
                      'input_dropout': 0.099,
                      'output_dropout': 0.074,
                      'update_dropout': 0.015,
                      'learning_rate': 0.00906,
                      'weight_decay': 0.000532,
                      'hidden_units': 100
                     }
        
        self.penn_treebank_200 ={
                      'input_size': None, 
                      'hidden_layer_size': 2,
                      'input_dropout': 0.473,
                      'output_dropout': 0.345,
                      'update_dropout': 0.013,
                      'learning_rate': 0.00496,
                      'weight_decay': 0.000191,
                      'hidden_units': 200
                     }
        
        self.penn_treebank_650 ={
                      'input_size': None, 
                      'hidden_layer_size': 2,
                      'input_dropout': 0.607,
                      'output_dropout': 0.630,
                      'update_dropout': 0.083,
                      'learning_rate': 0.00568,
                      'weight_decay': 0.000145,
                      'hidden_units': 650
                     }
        
        self.wikitext_10 ={
                      'input_size': None, 
                      'hidden_layer_size': 2,
                      'input_dropout': 0.015,
                      'output_dropout': 0.039,
                      'update_dropout': 0,
                      'learning_rate': 0.01235,
                      'weight_decay': 0,
                      'hidden_units': 10
                     }
        
        self.wikitext_100 ={
                      'input_size': None, 
                      'hidden_layer_size': 2,
                      'input_dropout': 0.198,
                      'output_dropout': 0.154,
                      'update_dropout': 0.002,
                      'learning_rate': 0.01162,
                      'weight_decay': 0.000123,
                      'hidden_units': 100
                     }
        
        self.wikitext_200 ={
                      'input_size': None, 
                      'hidden_layer_size': 2,
                      'input_dropout': 0.379,
                      'output_dropout': 0.351,
                      'update_dropout': 0,
                      'learning_rate': 0.00734,
                      'weight_decay': 0.000076,
                      'hidden_units': 200
                     }
        
        self.wikitext_650 ={
                      'input_size': None, 
                      'hidden_layer_size': 2,
                      'input_dropout': 0.572,
                      'output_dropout': 0.566,
                      'update_dropout': 0.071,
                      'learning_rate': 0.00354,
                      'weight_decay': 0.000112,
                      'hidden_units': 650
                     }
        