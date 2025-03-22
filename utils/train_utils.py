import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from tqdm import tqdm

def train_model(model, train_loader, val_loader, config, device='cuda', epochs=10):
    """
    Train a neural network model.
    
    Args:
        model (nn.Module): Neural network model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        config (dict): Model configuration including optimizer, learning rate, etc.
        device (str): Device to use for training ('cuda' or 'cpu')
        epochs (int): Number of training epochs
        
    Returns:
        dict: Training history including loss and accuracy metrics
    """
    # Move model to device
    model = model.to(device)
    
    # Setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    
    optimizer_class = config['optimizer']
    optimizer = optimizer_class(model.parameters(), lr=config['learning_rate'])
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'time_per_epoch': []
    }
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track loss and accuracy
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_loss = train_loss / train_total
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Track loss and accuracy
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = 100 * val_correct / val_total
        
        epoch_time = time.time() - start_time
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['time_per_epoch'].append(epoch_time)
        
        # Print results
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - '
              f'Time: {epoch_time:.2f}s')
    
    return history

def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate a trained model on test data.
    
    Args:
        model (nn.Module): Trained neural network model
        test_loader (DataLoader): Test data loader
        device (str): Device to use for evaluation ('cuda' or 'cpu')
        
    Returns:
        tuple: (test_loss, test_accuracy)
    """
    model = model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Track loss and accuracy
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss / test_total
    test_acc = 100 * test_correct / test_total
    
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    return test_loss, test_acc

def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (nn.Module): Neural network model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device():
    """
    Get the available device (CUDA or CPU).
    
    Returns:
        str: 'cuda' if CUDA is available, otherwise 'cpu'
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def set_random_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) 