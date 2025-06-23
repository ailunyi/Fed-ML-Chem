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
This file contains the setup functions for the different tasks. (copied from data_setup.py)
"""


def preprocess_graph():
    """
    Preprocess the HIV dataset from MoleculeNet for a classification task.

    The HIV dataset, introduced by the Drug Therapeutics Program (DTP) AIDS Antiviral Screen, contains 
    data on over 40,000 compounds tested for their ability to inhibit HIV replication. The compounds 
    are categorized into three classes based on their activity: confirmed inactive (CI), confirmed active 
    (CA), and confirmed moderately active (CM). For this classification task, we combine the active 
    categories (CA and CM) and classify the compounds into two categories: inactive (CI) and active (CA and CM).

    The function splits the dataset into training and test sets, with 80% of the data used for training 
    and 20% for testing.

    Returns:
        tuple: A tuple containing the training set and the test set, each as a subset of the HIV dataset.
    """
    data = MoleculeNet(root="data", name="HIV")
    split_index = int(len(data) * 0.8)
    trainset, testset = data[:split_index], data[split_index:]

    return trainset, testset