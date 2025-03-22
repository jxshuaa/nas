import torch
import torch.nn as nn
import torch.optim as optim

"""
This module defines the search space for Neural Architecture Search.
The search space includes:
- Number of layers
- Number of units per layer
- Activation functions
- Optimizers
- Kernel sizes (for CNN)
- Pooling types (for CNN)
- Dropout rates
"""

# MLP Search Space
MLP_SEARCH_SPACE = {
    'num_layers': [1, 2, 3, 4],
    'num_units': [16, 32, 64, 128, 256],
    'activation': [nn.ReLU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU],
    'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.5],
    'optimizer': [optim.Adam, optim.SGD, optim.RMSprop],
    'learning_rate': [0.1, 0.01, 0.001, 0.0001],
    'batch_size': [32, 64, 128, 256]
}

# CNN Search Space
CNN_SEARCH_SPACE = {
    'num_conv_layers': [1, 2, 3, 4],
    'num_filters': [16, 32, 64, 128, 256],
    'kernel_size': [3, 5, 7],
    'activation': [nn.ReLU, nn.LeakyReLU],
    'pooling': [nn.MaxPool2d, nn.AvgPool2d],
    'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.5],
    'num_fc_layers': [1, 2, 3],
    'num_fc_units': [64, 128, 256, 512],
    'optimizer': [optim.Adam, optim.SGD, optim.RMSprop],
    'learning_rate': [0.1, 0.01, 0.001, 0.0001],
    'batch_size': [32, 64, 128, 256]
}

# Search Space Configuration for different tasks
SEARCH_SPACES = {
    'mlp': MLP_SEARCH_SPACE,
    'cnn': CNN_SEARCH_SPACE
}

def get_search_space(model_type='cnn'):
    """
    Get the search space for a specific model type.
    
    Args:
        model_type (str): Type of model ('mlp' or 'cnn')
        
    Returns:
        dict: Search space configuration
    """
    if model_type not in SEARCH_SPACES:
        raise ValueError(f"Model type '{model_type}' not supported. Choose from {list(SEARCH_SPACES.keys())}")
    
    return SEARCH_SPACES[model_type] 