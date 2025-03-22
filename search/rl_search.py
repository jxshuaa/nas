import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm
import copy
import random

from models.model_builder import build_model
from utils.train_utils import train_model, evaluate_model, count_parameters, get_device

class RNNController(nn.Module):
    """
    RNN Controller for RL-based Neural Architecture Search.
    Samples architectures from the search space using a policy network.
    """
    
    def __init__(self, search_space, hidden_size=64):
        """
        Initialize RNN Controller.
        
        Args:
            search_space (dict): Search space for architectures
            hidden_size (int): Hidden size of the RNN
        """
        super(RNNController, self).__init__()
        
        self.search_space = search_space
        self.hidden_size = hidden_size
        
        # Define input and output sizes for each search space parameter
        self.params = list(search_space.keys())
        self.param_values = {param: values for param, values in search_space.items()}
        self.param_sizes = {param: len(values) for param, values in search_space.items()}
        
        # Initialize tracking lists
        self.actions = []
        self.log_probs = []
        self.entropies = []
        
        # Calculate total number of choices
        total_choices = sum(self.param_sizes.values())
        
        # Embedding layer
        self.embedding = nn.Embedding(total_choices, hidden_size)
        
        # LSTM controller
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        
        # Output layers for each parameter
        self.output_layers = nn.ModuleDict({
            param: nn.Linear(hidden_size, self.param_sizes[param])
            for param in self.params
        })
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize the parameters with Glorot initialization."""
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
    
    def forward(self):
        """
        Forward pass of the controller to sample a configuration.
        
        Returns:
            dict: Sampled configuration
        """
        self.actions = []
        self.log_probs = []
        self.entropies = []
        
        # Initial states
        h_t = torch.zeros(1, self.hidden_size, device=next(self.parameters()).device)
        c_t = torch.zeros(1, self.hidden_size, device=next(self.parameters()).device)
        
        # Track offsets for embedding lookups
        offset = 0
        num_previous_choices = 0
        
        # Sample parameters in order
        config = {}
        for i, param in enumerate(self.search_space.keys()):
            num_choices = len(self.param_values[param])
            
            if i == 0:
                x_t = torch.zeros(1, self.hidden_size, device=next(self.parameters()).device)
            else:
                # Fix: Ensure proper tensor creation and device placement
                action_index = torch.tensor([self.actions[-1] + offset - num_previous_choices], 
                                          dtype=torch.long, 
                                          device=next(self.parameters()).device)
                x_t = self.embedding(action_index)
            
            # LSTM step
            h_t, c_t = self.lstm(x_t, (h_t, c_t))
            
            # Output layer for this parameter
            logits = self.output_layers[param](h_t)
            
            # Sample action
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(log_probs * probs).sum(1, keepdim=True)
            
            # Fix: Use proper device for multinomial sampling
            action = probs.multinomial(num_samples=1).detach()
            selected_log_prob = log_probs.gather(1, action)
            
            # Store action and log prob
            # Fix: Handle scalar value correctly
            self.actions.append(action.item())
            self.log_probs.append(selected_log_prob)
            self.entropies.append(entropy)
            
            # Update config with sampled parameter value
            action_idx = action.item()
            # Fix: Check bounds before indexing
            if 0 <= action_idx < len(self.param_values[param]):
                config[param] = self.param_values[param][action_idx]
            else:
                # Default to first value if out of bounds
                config[param] = self.param_values[param][0]
                print(f"Warning: Index {action_idx} out of bounds for parameter {param}. Using default value.")
            
            # Track offset for embedding lookup
            if i == 0:
                num_previous_choices = num_choices
                offset = 0
            else:
                offset += num_previous_choices
                num_previous_choices = num_choices
        
        return config
    
    def sample_config(self):
        """
        Sample a configuration from the controller.
        
        Returns:
            dict: Sampled configuration
        """
        self.eval()
        with torch.no_grad():
            config = self()
        return config
    
    def reset(self):
        """
        Reset the controller state between batches.
        """
        self.actions = []
        self.log_probs = []
        self.entropies = []


class RLSearch:
    """
    Reinforcement Learning-based Neural Architecture Search.
    Uses a controller RNN to generate architectures and trains it with REINFORCE.
    """
    
    def __init__(self, search_space, model_type, input_shape, num_classes, dataset_loaders,
                 controller_hidden_size=64, controller_lr=0.001, controller_optimizer=optim.Adam,
                 controller_batch_size=10, num_episodes=100, epochs_per_model=5,
                 entropy_weight=0.0001, baseline_decay=0.999, verbose=True):
        """
        Initialize RL-based NAS.
        
        Args:
            search_space (dict): Search space for architectures
            model_type (str): Type of model to build ('mlp' or 'cnn')
            input_shape (tuple): Shape of input data
            num_classes (int): Number of output classes
            dataset_loaders (tuple): (train_loader, val_loader, test_loader)
            controller_hidden_size (int): Hidden size of the controller RNN
            controller_lr (float): Learning rate for the controller
            controller_optimizer (class): Optimizer class for the controller
            controller_batch_size (int): Batch size for controller updates
            num_episodes (int): Number of episodes (architectures to sample)
            epochs_per_model (int): Number of epochs to train each model
            entropy_weight (float): Weight for entropy regularization
            baseline_decay (float): Decay rate for the baseline (exponential moving average)
            verbose (bool): Whether to print progress
        """
        self.search_space = search_space
        self.model_type = model_type
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train_loader, self.val_loader, self.test_loader = dataset_loaders
        self.controller_hidden_size = controller_hidden_size
        self.controller_lr = controller_lr
        self.controller_optimizer = controller_optimizer
        self.controller_batch_size = controller_batch_size
        self.num_episodes = num_episodes
        self.epochs_per_model = epochs_per_model
        self.entropy_weight = entropy_weight
        self.baseline_decay = baseline_decay
        self.verbose = verbose
        self.device = get_device()
        
        # Initialize controller
        self.controller = RNNController(search_space, hidden_size=controller_hidden_size).to(self.device)
        self.controller_optim = controller_optimizer(self.controller.parameters(), lr=controller_lr)
        
        # Store results
        self.sampled_architectures = []
        self.rewards_history = []
        self.baseline = 0
        self.best_config = None
        self.best_model = None
        self.best_val_acc = 0
        self.best_test_acc = 0
    
    def train_controller(self, rewards):
        """
        Train the controller using policy gradients.
        
        Args:
            rewards (list): Rewards for each sampled architecture in the batch
        """
        # Normalize rewards
        rewards = torch.tensor(rewards, device=self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Reshape controller log_probs and entropies into batches
        batch_log_probs = []
        batch_entropies = []
        
        # For each architecture in the batch
        for i in range(self.controller_batch_size):
            if i < len(self.controller.log_probs):
                batch_log_probs.append(self.controller.log_probs[i])
                batch_entropies.append(self.controller.entropies[i])
        
        # Calculate loss
        loss = 0
        for i, reward in enumerate(rewards):
            if i < len(batch_log_probs):
                # Add policy gradient loss
                loss -= batch_log_probs[i] * reward
                
                # Add entropy regularization
                loss -= self.entropy_weight * batch_entropies[i]
        
        # Backward pass and optimize
        self.controller_optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.controller.parameters(), 5.0)
        self.controller_optim.step()
        
        return loss.item()
    
    def search(self):
        """
        Perform reinforcement learning-based search over the search space.
        
        Returns:
            dict: Results of the search process, including best architecture
        """
        print(f"Starting RL-based Neural Architecture Search")
        print(f"Episodes: {self.num_episodes}")
        print(f"Model type: {self.model_type}")
        print(f"Input shape: {self.input_shape}")
        print(f"Device: {self.device}")
        
        # Determine batch size for controller updates
        num_batches = self.num_episodes // self.controller_batch_size
        remainder = self.num_episodes % self.controller_batch_size
        
        # Store start time
        start_time = time.time()
        
        # Main search loop
        for batch_idx in range(num_batches + (1 if remainder > 0 else 0)):
            batch_size = self.controller_batch_size if batch_idx < num_batches else remainder
            episode_start = batch_idx * self.controller_batch_size
            
            if self.verbose:
                print(f"\n--- Episodes {episode_start+1}-{episode_start+batch_size}/{self.num_episodes} ---")
            
            # Sample architectures
            batch_configs = []
            batch_log_probs = []
            batch_entropies = []
            
            for b in range(batch_size):
                # Reset controller state
                self.controller.reset()
                
                # Set controller to training mode
                self.controller.train()
                
                # Save current state of log_probs and entropies
                current_log_probs_len = len(self.controller.log_probs)
                current_entropies_len = len(self.controller.entropies)
                
                # Sample a configuration
                config = self.controller()
                
                # Store log_probs and entropies for this sample
                new_log_probs = self.controller.log_probs
                new_entropies = self.controller.entropies
                
                # Add to batch
                batch_configs.append(config)
                batch_log_probs.append(new_log_probs)
                batch_entropies.append(new_entropies)
                
                if self.verbose:
                    print(f"\nSampled architecture {b+1}/{batch_size}:")
                    print(f"Config: {config}")
            
            # Save current log_probs and entropies for training
            self.controller.log_probs = batch_log_probs
            self.controller.entropies = batch_entropies
            
            # Evaluate architectures
            batch_rewards = []
            for b, config in enumerate(batch_configs):
                if self.verbose:
                    print(f"\nEvaluating architecture {b+1}/{batch_size}")
                
                try:
                    # Build model
                    model = build_model(config, self.model_type, self.input_shape, self.num_classes)
                    n_params = count_parameters(model)
                    
                    if self.verbose:
                        print(f"Parameters: {n_params}")
                    
                    # Train model
                    history = train_model(
                        model, 
                        self.train_loader, 
                        self.val_loader, 
                        config, 
                        device=self.device, 
                        epochs=self.epochs_per_model
                    )
                    
                    # Get validation accuracy as reward
                    val_acc = history['val_acc'][-1]
                    reward = val_acc
                    
                    # Store results
                    self.sampled_architectures.append({
                        'config': copy.deepcopy(config),
                        'val_acc': val_acc,
                        'params': n_params,
                        'history': history
                    })
                    
                    # Update best model if validation accuracy improves
                    if val_acc > self.best_val_acc:
                        self.best_val_acc = val_acc
                        self.best_config = copy.deepcopy(config)
                        self.best_model = copy.deepcopy(model)
                        
                        if self.verbose:
                            print(f"New best model found! Validation accuracy: {val_acc:.2f}%")
                
                except Exception as e:
                    if self.verbose:
                        print(f"Error evaluating architecture: {e}")
                    
                    # Assign poor reward to failed architectures
                    reward = 0
                
                # Update baseline (exponential moving average)
                episode = episode_start + b
                if episode == 0:
                    self.baseline = reward
                else:
                    self.baseline = self.baseline * self.baseline_decay + reward * (1 - self.baseline_decay)
                
                # Store reward
                batch_rewards.append(reward)
                self.rewards_history.append(reward)
            
            # Train controller on the batch
            loss = self.train_controller(batch_rewards)
            
            if self.verbose:
                print(f"\nController loss: {loss:.4f}")
                print(f"Average reward: {np.mean(batch_rewards):.2f}")
                print(f"Best validation accuracy so far: {self.best_val_acc:.2f}%")
        
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
        print("\nRL-based Neural Architecture Search completed!")
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
            'sampled_architectures': self.sampled_architectures,
            'rewards_history': self.rewards_history,
            'total_time': total_time
        } 