import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import time
import json

from data.data_loader import get_data_loaders
from search.search_space import get_search_space
from search.random_search import RandomSearch
from search.evolutionary_search import EvolutionarySearch
from search.rl_search import RLSearch
from utils.train_utils import set_random_seed, get_device
from utils.visualization import (
    plot_search_results, compare_algorithms,
    visualize_architecture, log_to_tensorboard, save_results
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Neural Architecture Search')
    
    # General settings
    parser.add_argument('--algorithm', type=str, default='random',
                        choices=['random', 'evolutionary', 'rl', 'all'],
                        help='NAS algorithm to use')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['mnist', 'cifar10'],
                        help='Dataset to use')
    parser.add_argument('--model_type', type=str, default='cnn',
                        choices=['mlp', 'cnn'],
                        help='Type of model to search for')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for logs')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory for results')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results')
    
    # Data loading settings
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for data loading')
    
    # Search settings
    parser.add_argument('--budget', type=int, default=50,
                        help='Budget for random search (number of architectures to evaluate)')
    parser.add_argument('--population_size', type=int, default=20,
                        help='Population size for evolutionary search')
    parser.add_argument('--num_generations', type=int, default=5,
                        help='Number of generations for evolutionary search')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of episodes for RL search')
    parser.add_argument('--epochs_per_model', type=int, default=3,
                        help='Number of epochs to train each model')
    
    return parser.parse_args()

def run_random_search(search_space, model_type, input_shape, num_classes, dataset_loaders, args):
    """Run Random Search NAS."""
    print("\n=== Running Random Search ===\n")
    
    random_search = RandomSearch(
        search_space=search_space,
        model_type=model_type,
        input_shape=input_shape,
        num_classes=num_classes,
        dataset_loaders=dataset_loaders,
        budget=args.budget,
        epochs_per_model=args.epochs_per_model,
        verbose=True
    )
    
    results = random_search.search()
    
    # Visualize results
    if args.visualize:
        plot_search_results(results, algorithm='random',
                            save_path=os.path.join(args.results_dir, 'random_search_results.png'))
        visualize_architecture(results['best_config'], model_type,
                              save_path=os.path.join(args.results_dir, 'random_best_architecture.png'))
    
    # Log results to TensorBoard
    log_to_tensorboard(results, algorithm='random', log_dir=args.log_dir)
    
    # Save results
    save_results(results, algorithm='random', save_dir=args.results_dir)
    
    return results

def run_evolutionary_search(search_space, model_type, input_shape, num_classes, dataset_loaders, args):
    """Run Evolutionary Search NAS."""
    print("\n=== Running Evolutionary Search ===\n")
    
    evolutionary_search = EvolutionarySearch(
        search_space=search_space,
        model_type=model_type,
        input_shape=input_shape,
        num_classes=num_classes,
        dataset_loaders=dataset_loaders,
        population_size=args.population_size,
        num_generations=args.num_generations,
        epochs_per_model=args.epochs_per_model,
        verbose=True
    )
    
    results = evolutionary_search.search()
    
    # Visualize results
    if args.visualize:
        plot_search_results(results, algorithm='evolutionary',
                            save_path=os.path.join(args.results_dir, 'evolutionary_search_results.png'))
        visualize_architecture(results['best_config'], model_type,
                              save_path=os.path.join(args.results_dir, 'evolutionary_best_architecture.png'))
    
    # Log results to TensorBoard
    log_to_tensorboard(results, algorithm='evolutionary', log_dir=args.log_dir)
    
    # Save results
    save_results(results, algorithm='evolutionary', save_dir=args.results_dir)
    
    return results

def run_rl_search(search_space, model_type, input_shape, num_classes, dataset_loaders, args):
    """Run Reinforcement Learning Search NAS."""
    print("\n=== Running RL-based Search ===\n")
    
    rl_search = RLSearch(
        search_space=search_space,
        model_type=model_type,
        input_shape=input_shape,
        num_classes=num_classes,
        dataset_loaders=dataset_loaders,
        num_episodes=args.num_episodes,
        epochs_per_model=args.epochs_per_model,
        verbose=True
    )
    
    results = rl_search.search()
    
    # Visualize results
    if args.visualize:
        plot_search_results(results, algorithm='rl',
                            save_path=os.path.join(args.results_dir, 'rl_search_results.png'))
        visualize_architecture(results['best_config'], model_type,
                              save_path=os.path.join(args.results_dir, 'rl_best_architecture.png'))
    
    # Log results to TensorBoard
    log_to_tensorboard(results, algorithm='rl', log_dir=args.log_dir)
    
    # Save results
    save_results(results, algorithm='rl', save_dir=args.results_dir)
    
    return results

def main():
    """Main function to run Neural Architecture Search."""
    # Parse arguments
    args = parse_args()
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset_loaders = get_data_loaders(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    train_loader, val_loader, test_loader, input_shape, num_classes = dataset_loaders
    
    # Get search space
    search_space = get_search_space(args.model_type)
    
    # Run NAS algorithms
    results = {}
    
    if args.algorithm == 'random' or args.algorithm == 'all':
        results['random'] = run_random_search(
            search_space, args.model_type, input_shape, num_classes, 
            (train_loader, val_loader, test_loader), args
        )
    
    if args.algorithm == 'evolutionary' or args.algorithm == 'all':
        results['evolutionary'] = run_evolutionary_search(
            search_space, args.model_type, input_shape, num_classes, 
            (train_loader, val_loader, test_loader), args
        )
    
    if args.algorithm == 'rl' or args.algorithm == 'all':
        results['rl'] = run_rl_search(
            search_space, args.model_type, input_shape, num_classes, 
            (train_loader, val_loader, test_loader), args
        )
    
    # Compare algorithms if multiple were run
    if args.algorithm == 'all' and args.visualize:
        compare_algorithms(
            results['random'],
            results['evolutionary'],
            results['rl'],
            save_path=os.path.join(args.results_dir, 'algorithm_comparison.png')
        )
    
    print("\nNeural Architecture Search completed!")

if __name__ == '__main__':
    main() 