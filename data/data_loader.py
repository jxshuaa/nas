import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np

def get_data_loaders(dataset_name, batch_size=128, num_workers=2, val_split=0.1):
    """
    Get data loaders for training and evaluation.
    
    Args:
        dataset_name (str): Name of the dataset ('mnist', 'cifar10', etc.)
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        val_split (float): Fraction of training data to use for validation
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, input_shape, num_classes)
    """
    if dataset_name.lower() == 'mnist':
        return get_mnist_loaders(batch_size, num_workers, val_split)
    elif dataset_name.lower() == 'cifar10':
        return get_cifar10_loaders(batch_size, num_workers, val_split)
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Choose from ['mnist', 'cifar10']")

def get_mnist_loaders(batch_size=128, num_workers=2, val_split=0.1):
    """
    Get MNIST data loaders.
    
    Args:
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        val_split (float): Fraction of training data to use for validation
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, input_shape, num_classes)
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load the training data
    train_dataset = torchvision.datasets.MNIST(root='./data/raw', 
                                             train=True, 
                                             download=True, 
                                             transform=transform)
    
    # Split training data into train and validation sets
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Download and load the test data
    test_dataset = torchvision.datasets.MNIST(root='./data/raw', 
                                            train=False, 
                                            download=True, 
                                            transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # MNIST: 1 channel, 28x28 images, 10 classes
    input_shape = (1, 28, 28)
    num_classes = 10
    
    return train_loader, val_loader, test_loader, input_shape, num_classes

def get_cifar10_loaders(batch_size=128, num_workers=2, val_split=0.1):
    """
    Get CIFAR-10 data loaders.
    
    Args:
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        val_split (float): Fraction of training data to use for validation
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, input_shape, num_classes)
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Download and load the training data
    train_dataset = torchvision.datasets.CIFAR10(root='./data/raw', 
                                               train=True, 
                                               download=True, 
                                               transform=train_transform)
    
    # Split training data into train and validation sets
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Apply test transform to validation dataset
    val_dataset.dataset.transform = test_transform
    
    # Download and load the test data
    test_dataset = torchvision.datasets.CIFAR10(root='./data/raw', 
                                              train=False, 
                                              download=True, 
                                              transform=test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # CIFAR-10: 3 channels, 32x32 images, 10 classes
    input_shape = (3, 32, 32)
    num_classes = 10
    
    return train_loader, val_loader, test_loader, input_shape, num_classes 