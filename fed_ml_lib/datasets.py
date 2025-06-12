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
import numpy as np


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


def create_federated_data_loaders(
    dataset_name: str,
    data_path: str = "./data/",
    num_clients: int = 10,
    batch_size: int = 32,
    resize: Optional[int] = None,
    val_split: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    partition_strategy: str = "iid"  # "iid", "non_iid", "dirichlet"
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """
    Create federated data loaders for multiple clients.
    
    Args:
        dataset_name: Name of the dataset
        data_path: Path to the data directory
        num_clients: Number of federated clients
        batch_size: Batch size for data loaders
        resize: Target size for resizing images (optional)
        val_split: Fraction of training data to use for validation
        seed: Random seed for reproducibility
        num_workers: Number of workers for data loading
        partition_strategy: Strategy for partitioning data among clients
        
    Returns:
        Tuple of (client_train_loaders, client_val_loaders, test_loader)
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Load the full dataset
    if dataset_name.lower() == 'cifar':
        transformer = get_transforms('cifar')
        trainset = datasets.CIFAR10(os.path.join(data_path, 'cifar'), train=True, download=True, transform=transformer)
        testset = datasets.CIFAR10(os.path.join(data_path, 'cifar'), train=False, download=True, transform=transformer)
    else:
        trainset, testset = load_image_dataset(dataset_name, data_path, resize)
    
    # Create test loader (shared across all clients)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Partition training data among clients
    if partition_strategy == "iid":
        client_datasets = _partition_iid(trainset, num_clients, seed)
    elif partition_strategy == "non_iid":
        client_datasets = _partition_non_iid(trainset, num_clients, seed)
    elif partition_strategy == "dirichlet":
        client_datasets = _partition_dirichlet(trainset, num_clients, seed)
    else:
        raise ValueError(f"Unknown partition strategy: {partition_strategy}")
    
    # Create train and validation loaders for each client
    client_train_loaders = []
    client_val_loaders = []
    
    for client_dataset in client_datasets:
        # Split client data into train and validation
        val_size = int(len(client_dataset) * val_split)
        train_size = len(client_dataset) - val_size
        
        if val_size > 0:
            train_dataset, val_dataset = random_split(
                client_dataset, [train_size, val_size], 
                generator=torch.Generator().manual_seed(seed)
            )
        else:
            train_dataset = client_dataset
            val_dataset = client_dataset  # Use training data for validation if no split
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        client_train_loaders.append(train_loader)
        client_val_loaders.append(val_loader)
    
    print(f"Created federated data loaders for {num_clients} clients")
    print(f"Dataset: {dataset_name}, Partition strategy: {partition_strategy}")
    print(f"Average samples per client: {len(trainset) // num_clients}")
    
    return client_train_loaders, client_val_loaders, test_loader


def _partition_iid(dataset, num_clients: int, seed: int):
    """Partition dataset into IID subsets for each client."""
    torch.manual_seed(seed)
    
    # Shuffle indices
    indices = torch.randperm(len(dataset)).tolist()
    
    # Split indices among clients
    client_datasets = []
    samples_per_client = len(dataset) // num_clients
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        if i == num_clients - 1:  # Last client gets remaining samples
            end_idx = len(dataset)
        else:
            end_idx = (i + 1) * samples_per_client
        
        client_indices = indices[start_idx:end_idx]
        client_dataset = torch.utils.data.Subset(dataset, client_indices)
        client_datasets.append(client_dataset)
    
    return client_datasets


def _partition_non_iid(dataset, num_clients: int, seed: int, classes_per_client: int = 2):
    """Partition dataset into non-IID subsets (each client gets limited classes)."""
    torch.manual_seed(seed)
    
    # Get class information
    if hasattr(dataset, 'classes'):
        num_classes = len(dataset.classes)
    else:
        # Estimate number of classes
        labels = [dataset[i][1] for i in range(min(1000, len(dataset)))]
        num_classes = len(set(labels))
    
    # Group samples by class
    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(dataset):
        if isinstance(label, torch.Tensor):
            label = label.item()
        class_indices[label].append(idx)
    
    # Assign classes to clients
    client_datasets = []
    classes_list = list(range(num_classes))
    torch.manual_seed(seed)
    
    for i in range(num_clients):
        # Randomly select classes for this client
        client_classes = torch.randperm(num_classes)[:classes_per_client].tolist()
        
        # Collect indices for selected classes
        client_indices = []
        for class_id in client_classes:
            class_samples = class_indices[class_id]
            # Take a portion of samples from each class
            samples_per_class = len(class_samples) // (num_clients // classes_per_client + 1)
            start_idx = (i % (num_clients // classes_per_client)) * samples_per_class
            end_idx = min(start_idx + samples_per_class, len(class_samples))
            client_indices.extend(class_samples[start_idx:end_idx])
        
        if client_indices:  # Only create dataset if there are indices
            client_dataset = torch.utils.data.Subset(dataset, client_indices)
            client_datasets.append(client_dataset)
    
    # If we have fewer datasets than clients, duplicate some
    while len(client_datasets) < num_clients:
        client_datasets.append(client_datasets[len(client_datasets) % len(client_datasets)])
    
    return client_datasets[:num_clients]


def _partition_dirichlet(dataset, num_clients: int, seed: int, alpha: float = 0.5):
    """Partition dataset using Dirichlet distribution for realistic non-IID setting."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get class information
    if hasattr(dataset, 'classes'):
        num_classes = len(dataset.classes)
    else:
        labels = [dataset[i][1] for i in range(min(1000, len(dataset)))]
        num_classes = len(set(labels))
    
    # Group samples by class
    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(dataset):
        if isinstance(label, torch.Tensor):
            label = label.item()
        class_indices[label].append(idx)
    
    # Generate Dirichlet distribution for each class
    client_datasets = []
    
    for class_id in range(num_classes):
        class_samples = class_indices[class_id]
        if not class_samples:
            continue
            
        # Sample from Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_clients)
        
        # Distribute samples according to proportions
        start_idx = 0
        for client_id in range(num_clients):
            if client_id >= len(client_datasets):
                client_datasets.append([])
            
            num_samples = int(proportions[client_id] * len(class_samples))
            end_idx = min(start_idx + num_samples, len(class_samples))
            
            client_datasets[client_id].extend(class_samples[start_idx:end_idx])
            start_idx = end_idx
    
    # Convert to Subset objects
    final_datasets = []
    for client_indices in client_datasets:
        if client_indices:
            client_dataset = torch.utils.data.Subset(dataset, client_indices)
            final_datasets.append(client_dataset)
    
    # Ensure we have exactly num_clients datasets
    while len(final_datasets) < num_clients:
        final_datasets.append(final_datasets[0])  # Duplicate first dataset
    
    return final_datasets[:num_clients]


class MultimodalDataset(torch.utils.data.Dataset):
    """
    Dataset for multimodal learning (e.g., images + sequences).
    
    This is a placeholder for future multimodal implementations.
    """
    
    def __init__(self, modality1_data, modality2_data, labels):
        """
        Initialize multimodal dataset.
        
        Args:
            modality1_data: First modality data (e.g., images)
            modality2_data: Second modality data (e.g., sequences)
            labels: Labels for the data
        """
        self.modality1_data = modality1_data
        self.modality2_data = modality2_data
        self.labels = labels
        
        assert len(modality1_data) == len(modality2_data) == len(labels), \
            "All modalities must have the same number of samples"
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            self.modality1_data[idx],
            self.modality2_data[idx],
            self.labels[idx]
        )


def create_multimodal_data_loaders(
    modality1_path: str,
    modality2_path: str,
    batch_size: int = 32,
    val_split: float = 0.1,
    seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for multimodal learning.
    
    This is a placeholder function for future multimodal implementations.
    
    Args:
        modality1_path: Path to first modality data
        modality2_path: Path to second modality data
        batch_size: Batch size for data loaders
        val_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # TODO: Implement multimodal data loading
    # This would involve loading and preprocessing different types of data
    # (images, sequences, etc.) and combining them appropriately
    
    raise NotImplementedError("Multimodal data loading not yet implemented")

# Add your dataset loader classes and utility functions here 