"""
datasets.py
-----------
This module contains dataset loading, preprocessing, and abstraction utilities for the fed_ml_lib library.
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple, List, Optional


# Normalization values for different datasets
NORMALIZE_DICT = {
    'mnist': dict(mean=(0.1307,), std=(0.3081,)),
    'cifar': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'PILL': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'Wafer': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'HIV': None,
    'DNA': None,
}


def get_transforms(dataset_name: str, resize: Optional[int] = None) -> transforms.Compose:
    """
    Get appropriate transforms for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        resize: Target size for resizing (optional)
        
    Returns:
        Composed transforms
    """
    transform_list = []
    
    if resize is not None:
        transform_list.append(transforms.Resize((resize, resize)))
    
    transform_list.append(transforms.ToTensor())
    
    if dataset_name in NORMALIZE_DICT and NORMALIZE_DICT[dataset_name] is not None:
        transform_list.append(transforms.Normalize(**NORMALIZE_DICT[dataset_name]))
    
    return transforms.Compose(transform_list)


def load_image_dataset(dataset_name: str, data_path: str, resize: Optional[int] = None) -> Tuple[datasets.ImageFolder, datasets.ImageFolder]:
    """
    Load an image dataset from folder structure.
    
    Args:
        dataset_name: Name of the dataset
        data_path: Path to the dataset
        resize: Target size for resizing (optional)
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    transformer = get_transforms(dataset_name, resize)
    
    train_path = os.path.join(data_path, dataset_name, "Training")
    test_path = os.path.join(data_path, dataset_name, "Testing")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Dataset paths not found: {train_path} or {test_path}")
    
    trainset = datasets.ImageFolder(train_path, transform=transformer)
    testset = datasets.ImageFolder(test_path, transform=transformer)
    
    return trainset, testset


def create_data_loaders(
    dataset_name: str,
    data_path: str = "./data/",
    batch_size: int = 32,
    resize: Optional[int] = None,
    val_split: float = 0.1,
    seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        data_path: Path to the data directory
        batch_size: Batch size for data loaders
        resize: Target size for resizing images (optional)
        val_split: Fraction of training data to use for validation
        seed: Random seed for reproducibility
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    if dataset_name.lower() == 'cifar':
        # Handle CIFAR-10 dataset
        transformer = get_transforms('cifar')
        trainset = datasets.CIFAR10(os.path.join(data_path, 'cifar'), train=True, download=True, transform=transformer)
        testset = datasets.CIFAR10(os.path.join(data_path, 'cifar'), train=False, download=True, transform=transformer)
    else:
        # Handle image folder datasets
        trainset, testset = load_image_dataset(dataset_name, data_path, resize)
    
    # Split training set into train and validation
    val_size = int(len(trainset) * val_split)
    train_size = len(trainset) - val_size
    train_dataset, val_dataset = random_split(trainset, [train_size, val_size], 
                                            generator=torch.Generator().manual_seed(seed))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"Dataset: {dataset_name}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(testset)}")
    
    if hasattr(trainset, 'classes'):
        print(f"Classes: {trainset.classes}")
    
    return train_loader, val_loader, test_loader


def get_dataset_info(dataset_name: str, data_path: str = "./data/") -> dict:
    """
    Get information about a dataset.
    
    Args:
        dataset_name: Name of the dataset
        data_path: Path to the data directory
        
    Returns:
        Dictionary containing dataset information
    """
    try:
        if dataset_name.lower() == 'cifar':
            trainset = datasets.CIFAR10(os.path.join(data_path, 'cifar'), train=True, download=False)
            num_classes = 10
            classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        else:
            trainset, _ = load_image_dataset(dataset_name, data_path)
            num_classes = len(trainset.classes)
            classes = trainset.classes
        
        return {
            'num_classes': num_classes,
            'classes': classes,
            'num_samples': len(trainset)
        }
    except Exception as e:
        print(f"Error getting dataset info: {e}")
        return {'num_classes': 2, 'classes': ['class_0', 'class_1'], 'num_samples': 0}

# Add your dataset loader classes and utility functions here 