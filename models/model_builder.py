import torch
import torch.nn as nn
import numpy as np

class MLPModel(nn.Module):
    """
    Multilayer Perceptron model with configurable architecture.
    """
    def __init__(self, config, input_size, num_classes):
        """
        Initialize MLP model with given configuration.
        
        Args:
            config (dict): Model configuration containing architecture parameters
            input_size (int): Size of input features
            num_classes (int): Number of output classes
        """
        super(MLPModel, self).__init__()
        
        self.config = config
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Build layers
        layers = []
        prev_units = input_size
        
        # Add hidden layers
        for i in range(config['num_layers']):
            layers.append(nn.Linear(prev_units, config['num_units']))
            layers.append(config['activation']())
            
            if config['dropout_rate'] > 0:
                layers.append(nn.Dropout(config['dropout_rate']))
            
            prev_units = config['num_units']
        
        # Add output layer
        layers.append(nn.Linear(prev_units, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)


class CNNModel(nn.Module):
    """
    Convolutional Neural Network model with configurable architecture.
    """
    def __init__(self, config, input_channels, input_height, input_width, num_classes):
        """
        Initialize CNN model with given configuration.
        
        Args:
            config (dict): Model configuration containing architecture parameters
            input_channels (int): Number of input channels
            input_height (int): Height of input image
            input_width (int): Width of input image
            num_classes (int): Number of output classes
        """
        super(CNNModel, self).__init__()
        
        self.config = config
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.num_classes = num_classes
        
        # Build convolutional layers
        conv_layers = []
        in_channels = input_channels
        current_height, current_width = input_height, input_width
        
        for i in range(config['num_conv_layers']):
            out_channels = config['num_filters']
            kernel_size = config['kernel_size']
            
            # Apply padding to maintain spatial dimensions
            padding = kernel_size // 2
            
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
            conv_layers.append(config['activation']())
            
            # Apply pooling (reduces spatial dimensions by factor of 2)
            pool_kernel_size = 2
            if i < config['num_conv_layers'] - 1:  # Apply pooling to all but the last layer
                conv_layers.append(config['pooling'](pool_kernel_size))
                current_height //= pool_kernel_size
                current_width //= pool_kernel_size
            
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate the size of flattened features after convolutions
        flattened_size = in_channels * current_height * current_width
        
        # Build fully connected layers
        fc_layers = []
        prev_units = flattened_size
        
        for i in range(config['num_fc_layers']):
            fc_layers.append(nn.Linear(prev_units, config['num_fc_units']))
            fc_layers.append(config['activation']())
            
            if config['dropout_rate'] > 0:
                fc_layers.append(nn.Dropout(config['dropout_rate']))
            
            prev_units = config['num_fc_units']
        
        # Add output layer
        fc_layers.append(nn.Linear(prev_units, num_classes))
        
        self.fc_layers = nn.Sequential(*fc_layers)
    
    def forward(self, x):
        """Forward pass through the model."""
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


def build_model(config, model_type, input_shape, num_classes):
    """
    Build a neural network model based on configuration and model type.
    
    Args:
        config (dict): Model configuration
        model_type (str): Type of model to build ('mlp' or 'cnn')
        input_shape (tuple): Shape of input data
        num_classes (int): Number of output classes
        
    Returns:
        nn.Module: Neural network model
    """
    if model_type == 'mlp':
        input_size = np.prod(input_shape)
        return MLPModel(config, input_size, num_classes)
    
    elif model_type == 'cnn':
        if len(input_shape) != 3:
            raise ValueError("CNN input shape should be (channels, height, width)")
        
        input_channels, input_height, input_width = input_shape
        return CNNModel(config, input_channels, input_height, input_width, num_classes)
    
    else:
        raise ValueError(f"Model type '{model_type}' not supported. Choose from ['mlp', 'cnn']") 