from torchvision import datasets, transforms
from torch.utils.data import random_split, TensorDataset, Dataset
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
import pickle

from fed_ml_lib.core.utils import *

"""
This file contains the partitioning functions for the different tasks. (copied from data_setup.py)
"""


def preprocess_and_split_data(au_mfcc_path):
    # Load the Audio+Vision(MP4 Video input divided into Audio and Images) data
    with open(au_mfcc_path, 'rb') as f:
        au_mfcc = pickle.load(f)

    # Initialize lists for data and labels
    data = []
    labels = []

    # Process the data
    for key in au_mfcc:
        emotion = int(key.split('-')[2]) - 1
        labels.append(emotion)
        data.append(au_mfcc[key])

    # Convert lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels).reshape(-1, 1)

    # Concatenate data and labels
    data = np.hstack((data, labels))

    # Shuffle data
    data = shuffle(data)

    # Split data and labels
    X = data[:, :-1]
    y = data[:, -1].astype(int)

    # One-hot encode labels
    num_classes = np.unique(y).size
    y_one_hot = np.zeros((y.shape[0], num_classes))
    y_one_hot[np.arange(y.shape[0]), y] = 1

    # Split into test, train, and dev sets
    test_data = X[-181:-1]
    test_labels = y_one_hot[-181:-1]
    data = X[:-180]
    labels = y_one_hot[:-180]
    train_data = X[:1020]
    train_labels = y_one_hot[:1020]
    dev_data = X[1020:]
    dev_labels = y_one_hot[1020:]

    train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
    dev_data_tensor = torch.tensor(dev_data, dtype=torch.float32)
    dev_labels_tensor = torch.tensor(dev_labels, dtype=torch.float32)
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32)
    trainset = TensorDataset(train_data_tensor, train_labels_tensor)
    devset = TensorDataset(dev_data_tensor, dev_labels_tensor)
    testset = TensorDataset(test_data_tensor, test_labels_tensor)

    return trainset, devset, testset

def partition_non_iid(dataset, num_clients: int, seed: int = 42, 
                     classes_per_client: int = 2) -> List[List[int]]:
    """
    Partition dataset in a non-IID manner by limiting classes per client.
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        seed: Random seed for reproducibility
        classes_per_client: Number of classes each client should have
        
    Returns:
        List of lists containing indices for each client
    """
    np.random.seed(seed)
    
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        # Try to extract labels from dataset
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
        labels = np.array(labels)
    
    num_classes = len(np.unique(labels))
    
    # Group indices by class
    class_indices = {}
    for class_id in range(num_classes):
        class_indices[class_id] = np.where(labels == class_id)[0].tolist()
        np.random.shuffle(class_indices[class_id])
    
    # Assign classes to clients
    client_indices = [[] for _ in range(num_clients)]
    
    # Ensure each client gets exactly classes_per_client classes
    for client_id in range(num_clients):
        # Select classes for this client
        available_classes = list(range(num_classes))
        selected_classes = np.random.choice(available_classes, 
                                          min(classes_per_client, len(available_classes)), 
                                          replace=False)
        
        # Distribute data from selected classes
        for class_id in selected_classes:
            # Calculate how much data this client gets from this class
            class_data = class_indices[class_id]
            samples_per_client = len(class_data) // (num_clients // classes_per_client + 1)
            
            # Get indices for this client
            start_idx = 0
            end_idx = min(samples_per_client, len(class_data))
            
            client_indices[client_id].extend(class_data[start_idx:end_idx])
            # Remove used indices
            class_indices[class_id] = class_data[end_idx:]
    
    # Distribute remaining data
    for class_id in range(num_classes):
        remaining_data = class_indices[class_id]
        if remaining_data:
            # Distribute remaining data evenly among clients
            samples_per_client = len(remaining_data) // num_clients
            for client_id in range(num_clients):
                start_idx = client_id * samples_per_client
                if client_id == num_clients - 1:
                    end_idx = len(remaining_data)
                else:
                    end_idx = (client_id + 1) * samples_per_client
                
                client_indices[client_id].extend(remaining_data[start_idx:end_idx])
    
    # Shuffle each client's data
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])
    
    return client_indices

def split_data_client(dataset, num_clients, seed):
    """
    This function is used to split the dataset into train and test for each client.
    :param dataset: the dataset to split (type: torch.utils.data.Dataset)
    :param num_clients: the number of clients
    :param seed: the seed for the random split
    """
    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(dataset) // num_clients
    lengths = [partition_size] * (num_clients - 1)
    lengths += [len(dataset) - sum(lengths)]
    ds = random_split(dataset, lengths, torch.Generator().manual_seed(seed))
    return ds