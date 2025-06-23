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