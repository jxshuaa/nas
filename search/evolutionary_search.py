import random
import time
import numpy as np
import torch
from tqdm import tqdm
import copy

from models.model_builder import build_model
from utils.train_utils import train_model, evaluate_model, count_parameters, get_device

class EvolutionarySearch:
    """
    Evolutionary Algorithm-based Neural Architecture Search.
    Uses genetic algorithms to evolve neural network architectures.
    """
    
    def __init__(self, search_space, model_type, input_shape, num_classes, dataset_loaders,
                 population_size=20, num_generations=5, tournament_size=3, mutation_prob=0.2,
                 crossover_prob=0.5, epochs_per_model=5, elitism=2, verbose=True):
        """
        Initialize Evolutionary Search.
        
        Args:
            search_space (dict): Search space for architectures
            model_type (str): Type of model to build ('mlp' or 'cnn')
            input_shape (tuple): Shape of input data
            num_classes (int): Number of output classes
            dataset_loaders (tuple): (train_loader, val_loader, test_loader)
            population_size (int): Size of population in each generation
            num_generations (int): Number of generations to evolve
            tournament_size (int): Number of individuals in tournament selection
            mutation_prob (float): Probability of mutation
            crossover_prob (float): Probability of crossover
            epochs_per_model (int): Number of epochs to train each model
            elitism (int): Number of best individuals to keep in next generation
            verbose (bool): Whether to print progress
        """
        self.search_space = search_space
        self.model_type = model_type
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train_loader, self.val_loader, self.test_loader = dataset_loaders
        self.population_size = population_size
        self.num_generations = num_generations
        self.tournament_size = tournament_size
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.epochs_per_model = epochs_per_model
        self.elitism = elitism
        self.verbose = verbose
        self.device = get_device()
        
        # Store results
        self.current_population = []
        self.fitness_history = []
        self.best_individual = None
        self.best_fitness = 0
        self.best_model = None
        self.best_test_acc = 0
    
    def initialize_population(self):
        """
        Initialize a population of random individuals.
        
        Returns:
            list: List of individuals (configurations)
        """
        population = []
        
        for _ in range(self.population_size):
            # Sample random configuration
            config = {}
            for key, values in self.search_space.items():
                config[key] = random.choice(values)
            
            population.append(config)
        
        return population
    
    def evaluate_individual(self, individual):
        """
        Evaluate the fitness of an individual by training and validating the model.
        
        Args:
            individual (dict): Model configuration
            
        Returns:
            tuple: (val_acc, n_params, model, history)
        """
        # Build model
        model = build_model(individual, self.model_type, self.input_shape, self.num_classes)
        n_params = count_parameters(model)
        
        if self.verbose:
            print(f"Config: {individual}")
            print(f"Parameters: {n_params}")
        
        # Train model
        history = train_model(
            model, 
            self.train_loader, 
            self.val_loader, 
            individual, 
            device=self.device, 
            epochs=self.epochs_per_model
        )
        
        # Get validation accuracy as fitness
        val_acc = history['val_acc'][-1]
        
        return val_acc, n_params, model, history
    
    def tournament_selection(self, population, fitness_scores):
        """
        Select an individual using tournament selection.
        
        Args:
            population (list): List of individuals
            fitness_scores (list): List of fitness scores
            
        Returns:
            dict: Selected individual
        """
        # Select tournament candidates
        indices = random.sample(range(len(population)), min(self.tournament_size, len(population)))
        tournament_candidates = [population[i] for i in indices]
        tournament_fitness = [fitness_scores[i] for i in indices]
        
        # Select the best individual from the tournament
        best_idx = tournament_fitness.index(max(tournament_fitness))
        
        return copy.deepcopy(tournament_candidates[best_idx])
    
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to produce two offspring.
        
        Args:
            parent1 (dict): First parent
            parent2 (dict): Second parent
            
        Returns:
            tuple: (child1, child2)
        """
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        if random.random() < self.crossover_prob:
            # Perform uniform crossover
            for key in self.search_space.keys():
                if random.random() < 0.5:
                    child1[key], child2[key] = child2[key], child1[key]
        
        return child1, child2
    
    def mutate(self, individual):
        """
        Mutate an individual by randomly changing some of its genes.
        
        Args:
            individual (dict): Individual to mutate
            
        Returns:
            dict: Mutated individual
        """
        mutated = copy.deepcopy(individual)
        
        for key, values in self.search_space.items():
            if random.random() < self.mutation_prob:
                # Replace with a random value
                mutated[key] = random.choice(values)
        
        return mutated
    
    def search(self):
        """
        Perform evolutionary search over the search space.
        
        Returns:
            dict: Results of the search process, including best architecture
        """
        print(f"Starting Evolutionary Search with population size: {self.population_size}")
        print(f"Generations: {self.num_generations}")
        print(f"Model type: {self.model_type}")
        print(f"Input shape: {self.input_shape}")
        print(f"Device: {self.device}")
        
        start_time = time.time()
        
        # Initialize population
        self.current_population = self.initialize_population()
        
        # Evaluate initial population
        fitness_scores = []
        models = []
        param_counts = []
        histories = []
        
        for i, individual in enumerate(self.current_population):
            if self.verbose:
                print(f"\nEvaluating initial individual {i+1}/{self.population_size}")
            
            try:
                val_acc, n_params, model, history = self.evaluate_individual(individual)
                fitness_scores.append(val_acc)
                models.append(model)
                param_counts.append(n_params)
                histories.append(history)
                
                if val_acc > self.best_fitness:
                    self.best_fitness = val_acc
                    self.best_individual = copy.deepcopy(individual)
                    self.best_model = copy.deepcopy(model)
                    
                    if self.verbose:
                        print(f"New best model found! Validation accuracy: {val_acc:.2f}%")
            
            except Exception as e:
                if self.verbose:
                    print(f"Error evaluating individual: {e}")
                
                # Assign poor fitness to failed individuals
                fitness_scores.append(0)
                models.append(None)
                param_counts.append(0)
                histories.append(None)
        
        # Save initial generation stats
        gen_stats = {
            'best_fitness': max(fitness_scores) if fitness_scores else 0,
            'avg_fitness': np.mean(fitness_scores) if fitness_scores else 0,
            'best_individual': copy.deepcopy(self.best_individual)
        }
        self.fitness_history.append(gen_stats)
        
        # Evolution loop
        for gen in range(self.num_generations):
            if self.verbose:
                print(f"\n--- Generation {gen+1}/{self.num_generations} ---")
                print(f"Best fitness so far: {self.best_fitness:.2f}%")
            
            # Sort population by fitness
            sorted_indices = np.argsort(fitness_scores)[::-1]
            sorted_population = [self.current_population[i] for i in sorted_indices]
            sorted_fitness = [fitness_scores[i] for i in sorted_indices]
            sorted_models = [models[i] for i in sorted_indices]
            sorted_params = [param_counts[i] for i in sorted_indices]
            sorted_histories = [histories[i] for i in sorted_indices]
            
            # Create new population
            new_population = []
            new_fitness = []
            new_models = []
            new_params = []
            new_histories = []
            
            # Elitism: keep best individuals
            for i in range(self.elitism):
                if i < len(sorted_population):
                    new_population.append(sorted_population[i])
                    new_fitness.append(sorted_fitness[i])
                    new_models.append(sorted_models[i])
                    new_params.append(sorted_params[i])
                    new_histories.append(sorted_histories[i])
            
            # Create offspring through selection, crossover, and mutation
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self.tournament_selection(sorted_population, sorted_fitness)
                parent2 = self.tournament_selection(sorted_population, sorted_fitness)
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # Add children to new population
                for child in [child1, child2]:
                    if len(new_population) < self.population_size:
                        new_population.append(child)
            
            # Evaluate new individuals
            for i in range(len(new_fitness), len(new_population)):
                if self.verbose:
                    print(f"\nEvaluating new individual {i+1-len(new_fitness)}/{self.population_size-len(new_fitness)}")
                
                try:
                    val_acc, n_params, model, history = self.evaluate_individual(new_population[i])
                    new_fitness.append(val_acc)
                    new_models.append(model)
                    new_params.append(n_params)
                    new_histories.append(history)
                    
                    if val_acc > self.best_fitness:
                        self.best_fitness = val_acc
                        self.best_individual = copy.deepcopy(new_population[i])
                        self.best_model = copy.deepcopy(model)
                        
                        if self.verbose:
                            print(f"New best model found! Validation accuracy: {val_acc:.2f}%")
                
                except Exception as e:
                    if self.verbose:
                        print(f"Error evaluating individual: {e}")
                    
                    # Assign poor fitness to failed individuals
                    new_fitness.append(0)
                    new_models.append(None)
                    new_params.append(0)
                    new_histories.append(None)
            
            # Update population
            self.current_population = new_population
            fitness_scores = new_fitness
            models = new_models
            param_counts = new_params
            histories = new_histories
            
            # Save generation stats
            gen_stats = {
                'best_fitness': max(fitness_scores) if fitness_scores else 0,
                'avg_fitness': np.mean(fitness_scores) if fitness_scores else 0,
                'best_individual': copy.deepcopy(self.best_individual)
            }
            self.fitness_history.append(gen_stats)
        
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
        print("\nEvolutionary Search completed!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Best validation accuracy: {self.best_fitness:.2f}%")
        print(f"Best test accuracy: {self.best_test_acc:.2f}%")
        print(f"Best configuration: {self.best_individual}")
        
        # Return search summary
        return {
            'best_config': self.best_individual,
            'best_val_acc': self.best_fitness,
            'best_test_acc': self.best_test_acc,
            'best_model': self.best_model,
            'fitness_history': self.fitness_history,
            'total_time': total_time
        } 