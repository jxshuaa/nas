import random
import time
import torch
import numpy as np
from tqdm import tqdm
import copy

from models.model_builder import build_model
from utils.train_utils import train_model, evaluate_model, count_parameters, get_device

class RandomSearch:
    """
    Random Search algorithm for Neural Architecture Search.
    This is a baseline approach that randomly samples architectures from the search space.
    """
    
    def __init__(self, search_space, model_type, input_shape, num_classes, dataset_loaders, 
                 budget=100, epochs_per_model=5, verbose=True):
        """
        Initialize Random Search.
        
        Args:
            search_space (dict): Search space for architectures
            model_type (str): Type of model to build ('mlp' or 'cnn')
            input_shape (tuple): Shape of input data
            num_classes (int): Number of output classes
            dataset_loaders (tuple): (train_loader, val_loader, test_loader)
            budget (int): Number of architectures to sample
            epochs_per_model (int): Number of epochs to train each model
            verbose (bool): Whether to print progress
        """
        self.search_space = search_space
        self.model_type = model_type
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train_loader, self.val_loader, self.test_loader = dataset_loaders
        self.budget = budget
        self.epochs_per_model = epochs_per_model
        self.verbose = verbose
        self.device = get_device()
        
        # Store results
        self.results = []
        self.best_config = None
        self.best_model = None
        self.best_val_acc = 0
        self.best_test_acc = 0
    
    def sample_config(self):
        """
        Randomly sample a configuration from the search space.
        
        Returns:
            dict: Randomly sampled configuration
        """
        config = {}
        for key, values in self.search_space.items():
            config[key] = random.choice(values)
        return config
    
    def search(self):
        """
        Perform random search over the search space.
        
        Returns:
            dict: Results of the search process, including best architecture
        """
        print(f"Starting Random Search with budget: {self.budget}")
        print(f"Model type: {self.model_type}")
        print(f"Input shape: {self.input_shape}")
        print(f"Device: {self.device}")
        
        start_time = time.time()
        
        for i in tqdm(range(self.budget), desc="Random Search Progress"):
            # Sample a random configuration
            config = self.sample_config()
            
            # Build model
            model = build_model(config, self.model_type, self.input_shape, self.num_classes)
            n_params = count_parameters(model)
            
            if self.verbose:
                print(f"\nArchitecture {i+1}/{self.budget}:")
                print(f"Config: {config}")
                print(f"Parameters: {n_params}")
            
            try:
                # Train model
                history = train_model(
                    model, 
                    self.train_loader, 
                    self.val_loader, 
                    config, 
                    device=self.device, 
                    epochs=self.epochs_per_model
                )
                
                # Get validation accuracy
                val_acc = history['val_acc'][-1]
                
                # Track results
                result = {
                    'config': copy.deepcopy(config),
                    'val_acc': val_acc,
                    'params': n_params,
                    'history': history
                }
                
                self.results.append(result)
                
                # Update best model if validation accuracy improves
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_config = copy.deepcopy(config)
                    self.best_model = copy.deepcopy(model)
                    
                    if self.verbose:
                        print(f"New best model found! Validation accuracy: {val_acc:.2f}%")
            
            except Exception as e:
                if self.verbose:
                    print(f"Error training architecture: {e}")
        
        total_time = time.time() - start_time
        
        # Evaluate the best model on test set
        if self.best_model is not None:
            test_loss, test_acc = evaluate_model(
                self.best_model,
                self.test_loader,
                device=self.device
            )
            self.best_test_acc = test_acc
        
        # Final report
        print("\nRandom Search completed!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Best test accuracy: {self.best_test_acc:.2f}%")
        print(f"Best configuration: {self.best_config}")
        
        # Return search summary
        return {
            'best_config': self.best_config,
            'best_val_acc': self.best_val_acc,
            'best_test_acc': self.best_test_acc,
            'best_model': self.best_model,
            'results': self.results,
            'total_time': total_time
        } 