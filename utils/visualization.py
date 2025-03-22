import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import json

def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss and accuracy.
    
    Args:
        history (dict): Training history including loss and accuracy metrics
        save_path (str, optional): Path to save the plot
    """
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training and validation loss
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot training and validation accuracy
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_search_results(results, algorithm='random', save_path=None):
    """
    Plot search results for different algorithms.
    
    Args:
        results (dict): Results of the search process
        algorithm (str): Search algorithm ('random', 'evolutionary', 'rl')
        save_path (str, optional): Path to save the plot
    """
    if algorithm == 'random':
        # Plot validation accuracy of sampled architectures
        val_accs = [result['val_acc'] for result in results['results']]
        
        plt.figure(figsize=(10, 6))
        plt.plot(val_accs, 'b-')
        plt.plot(val_accs, 'bo')
        plt.axhline(y=results['best_val_acc'], color='r', linestyle='--', label=f'Best Accuracy: {results["best_val_acc"]:.2f}%')
        plt.xlabel('Architecture Index')
        plt.ylabel('Validation Accuracy (%)')
        plt.title('Random Search Results')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    elif algorithm == 'evolutionary':
        # Plot best and average fitness over generations
        generations = range(len(results['fitness_history']))
        best_fitness = [gen_stats['best_fitness'] for gen_stats in results['fitness_history']]
        avg_fitness = [gen_stats['avg_fitness'] for gen_stats in results['fitness_history']]
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitness, 'r-', label='Best Fitness')
        plt.plot(generations, avg_fitness, 'b-', label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Validation Accuracy %)')
        plt.title('Evolutionary Search Results')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    elif algorithm == 'rl':
        # Plot reward (validation accuracy) over episodes
        rewards = results['rewards_history']
        episodes = range(len(rewards))
        
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, rewards, 'g-')
        plt.plot(episodes, rewards, 'go', alpha=0.3)
        
        # Plot moving average
        window_size = min(10, len(rewards))
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, 'r-', label=f'Moving Average (window={window_size})')
        
        plt.axhline(y=results['best_val_acc'], color='r', linestyle='--', label=f'Best Accuracy: {results["best_val_acc"]:.2f}%')
        plt.xlabel('Episode')
        plt.ylabel('Reward (Validation Accuracy %)')
        plt.title('Reinforcement Learning Search Results')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()

def compare_algorithms(random_results, ea_results, rl_results, save_path=None):
    """
    Compare different NAS algorithms.
    
    Args:
        random_results (dict): Results of random search
        ea_results (dict): Results of evolutionary search
        rl_results (dict): Results of reinforcement learning search
        save_path (str, optional): Path to save the plot
    """
    # Get best validation accuracies
    accuracies = [
        random_results['best_val_acc'],
        ea_results['best_val_acc'],
        rl_results['best_val_acc']
    ]
    
    # Get total search time
    times = [
        random_results['total_time'],
        ea_results['total_time'],
        rl_results['total_time']
    ]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot best validation accuracies
    algorithms = ['Random', 'Evolutionary', 'RL']
    colors = ['blue', 'orange', 'green']
    
    ax1.bar(algorithms, accuracies, color=colors)
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Best Validation Accuracy (%)')
    ax1.set_title('Best Validation Accuracy by Algorithm')
    
    # Add values on top of bars
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.5, f'{v:.2f}%', ha='center')
    
    # Plot search time
    ax2.bar(algorithms, times, color=colors)
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Search Time (seconds)')
    ax2.set_title('Search Time by Algorithm')
    
    # Add values on top of bars
    for i, v in enumerate(times):
        ax2.text(i, v + 0.5, f'{v:.2f}s', ha='center')
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def visualize_architecture(config, model_type, save_path=None):
    """
    Visualize a neural network architecture.
    
    Args:
        config (dict): Model configuration
        model_type (str): Type of model ('mlp' or 'cnn')
        save_path (str, optional): Path to save the visualization
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    if model_type == 'mlp':
        # Visualize MLP architecture
        layers = []
        units = []
        
        # Input layer
        layers.append('Input')
        units.append('?')  # Input size depends on the dataset
        
        # Hidden layers
        for i in range(config['num_layers']):
            layers.append(f'Hidden {i+1}')
            units.append(str(config['num_units']))
        
        # Output layer
        layers.append('Output')
        units.append('?')  # Output size depends on the dataset
        
        # Create a table to display the architecture
        plt.axis('off')
        table = plt.table(
            cellText=[units],
            rowLabels=['Units'],
            colLabels=layers,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        plt.title(f'MLP Architecture\nActivation: {config["activation"].__name__}, Dropout: {config["dropout_rate"]}, Optimizer: {config["optimizer"].__name__}')
    
    elif model_type == 'cnn':
        # Create layers representation
        layer_types = []
        layer_configs = []
        
        # Convolutional layers
        for i in range(config['num_conv_layers']):
            layer_types.append(f'Conv {i+1}')
            layer_configs.append(f'{config["num_filters"]} filters, {config["kernel_size"]}x{config["kernel_size"]} kernel')
            
            # Add activation
            layer_types.append(f'Activation')
            layer_configs.append(f'{config["activation"].__name__}')
            
            # Add pooling (except for the last conv layer)
            if i < config['num_conv_layers'] - 1:
                layer_types.append(f'Pooling')
                layer_configs.append(f'{config["pooling"].__name__}, 2x2')
        
        # Flatten layer
        layer_types.append('Flatten')
        layer_configs.append('')
        
        # Fully connected layers
        for i in range(config['num_fc_layers']):
            layer_types.append(f'FC {i+1}')
            layer_configs.append(f'{config["num_fc_units"]} units')
            
            # Add activation
            layer_types.append(f'Activation')
            layer_configs.append(f'{config["activation"].__name__}')
            
            # Add dropout
            if config['dropout_rate'] > 0:
                layer_types.append(f'Dropout')
                layer_configs.append(f'rate={config["dropout_rate"]}')
        
        # Output layer
        layer_types.append('Output')
        layer_configs.append('? units')  # Output size depends on the dataset
        
        # Create a table to display the architecture
        plt.axis('off')
        table = plt.table(
            cellText=[layer_configs],
            rowLabels=['Config'],
            colLabels=layer_types,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        plt.title(f'CNN Architecture\nOptimizer: {config["optimizer"].__name__}, Learning Rate: {config["learning_rate"]}')
    
    # Save visualization if path provided
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def log_to_tensorboard(results, algorithm, log_dir='logs'):
    """
    Log search results to TensorBoard.
    
    Args:
        results (dict): Results of the search process
        algorithm (str): Search algorithm ('random', 'evolutionary', 'rl')
        log_dir (str): Directory to save TensorBoard logs
    """
    log_dir = os.path.join(log_dir, algorithm)
    writer = SummaryWriter(log_dir)
    
    # Log best accuracy
    writer.add_scalar(f'{algorithm}/best_val_acc', results['best_val_acc'], 0)
    writer.add_scalar(f'{algorithm}/best_test_acc', results['best_test_acc'], 0)
    
    # Log search time
    writer.add_scalar(f'{algorithm}/search_time', results['total_time'], 0)
    
    if algorithm == 'random':
        # Log validation accuracy of sampled architectures
        for i, result in enumerate(results['results']):
            writer.add_scalar(f'{algorithm}/val_acc', result['val_acc'], i)
            writer.add_scalar(f'{algorithm}/params', result['params'], i)
    
    elif algorithm == 'evolutionary':
        # Log fitness over generations
        for i, gen_stats in enumerate(results['fitness_history']):
            writer.add_scalar(f'{algorithm}/best_fitness', gen_stats['best_fitness'], i)
            writer.add_scalar(f'{algorithm}/avg_fitness', gen_stats['avg_fitness'], i)
    
    elif algorithm == 'rl':
        # Log rewards over episodes
        for i, reward in enumerate(results['rewards_history']):
            writer.add_scalar(f'{algorithm}/reward', reward, i)
    
    writer.close()

def save_results(results, algorithm, save_dir='results'):
    """
    Save search results to JSON file.
    
    Args:
        results (dict): Results of the search process
        algorithm (str): Search algorithm ('random', 'evolutionary', 'rl')
        save_dir (str): Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a serializable copy of results
    serializable_results = {
        'best_val_acc': results['best_val_acc'],
        'best_test_acc': results['best_test_acc'],
        'total_time': results['total_time']
    }
    
    # Convert config values to strings for JSON serialization
    if results['best_config']:
        serializable_config = {}
        for key, value in results['best_config'].items():
            if hasattr(value, '__name__'):
                serializable_config[key] = value.__name__
            else:
                serializable_config[key] = value
        serializable_results['best_config'] = serializable_config
    
    # Save to JSON file
    with open(os.path.join(save_dir, f'{algorithm}_results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=4) 