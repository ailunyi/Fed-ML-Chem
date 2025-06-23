import random
import torch
import shutil
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sn
import pandas as pd
import torch.nn.functional
from collections import OrderedDict
from typing import List

"""
This file contains the utility functions for the different tasks. (copied from common.py)
"""

def create_files_train_test(path_init, path_final, splitter):
    """
    Split the dataset from path_init into two datasets : train and test in path_final
    with the splitter ratio (in %). Example : if splitter = 10, 10% of the initial dataset will be in the test dataset.

    :param path_init: path of the initial dataset
    :param path_final: path of the final dataset
    :param splitter: ratio (in %) of the initial dataset that will be in the test dataset.
    """
    # Move a file from rep1 to rep2
    for classe in os.listdir(path_init):
        list_init = os.listdir(path_init + "/" + classe)
        size_test = int(len(list_init) * splitter/100)
        print("Before : ", len(list_init))
        for _ in range(size_test):
            e = random.choice(list_init)  # random choice of the path of an image
            list_init.remove(e)
            shutil.move(path_init + classe + "/" + e, path_final + classe + "/" + e)

        print("After", path_init + classe, ":", len(os.listdir(path_init + classe)))
        print(path_final + classe, ":", len(os.listdir(path_final + classe)))

def choice_device(device):
    """
    A function to choose the device

    :param device: the device to choose (cpu, gpu or mps)
    """
    if torch.cuda.is_available() and device != "cpu":
        # on Windows, "cuda:0" if torch.cuda.is_available()
        device = "cuda:0"

    elif torch.backends.mps.is_available() and torch.backends.mps.is_built() and device != "cpu":
        """
        on Mac : 
        - torch.backends.mps.is_available() ensures that the current MacOS version is at least 12.3+
        - torch.backends.mps.is_built() ensures that the current current PyTorch installation was built with MPS activated.
        """
        device = "mps"

    else:
        device = "cpu"

    return device

def classes_string(name_dataset):
    """
    A function to get the classes of the dataset

    :param name_dataset: the name of the dataset
    :return: classes (the classes of the dataset) in a tuple
    """
    if name_dataset == "cifar":
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    elif name_dataset == "animaux":
        classes = ('cat', 'dog')

    elif name_dataset == "breast":
        classes = ('0', '1')

    elif name_dataset == "histo":
        classes = ('0', '1')

    elif name_dataset == "MRI":
        classes = ('glioma', 'meningioma', 'notumor', 'pituitary')

    elif name_dataset == "DNA":
        classes = ('0', '1', '2', '3', '4', '5', '6') 

    elif name_dataset == "PCOS":
        classes = ('0', '1')        

    elif name_dataset == "MMF":
        classes = ('happy', 'sad', 'angry', 'fearful', 'surprise', 'disgust', 'calm', 'neutral')   

    elif name_dataset == "DNA+MRI":
        classes = (('glioma', 'meningioma', 'notumor', 'pituitary'), ('0', '1', '2', '3', '4', '5', '6'))    

    elif name_dataset == "PILL":
        classes = ('bad', 'good') 
    
    elif name_dataset == "hiv":
        classes = ('confirmed inactive (CI)', 'confirmed active (CA)/confirmed moderately active (CM)')

    elif name_dataset == "Wafer":
        classes = ('Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none')
        
    else:
        print("Warning problem : unspecified dataset")
        return ()

    return classes

def supp_ds_store(path):
    """
    Delete the hidden file ".DS_Store" created on macOS

    :param path: path to the folder where the hidden file ".DS_Store" is
    """
    for i in os.listdir(path):
        if i == ".DS_Store":
            print("Deleting of the hidden file '.DS_Store'")
            os.remove(path + "/" + i)

def get_parameters2(net) -> List[np.ndarray]:
    """
    Get the parameters of the network
    :param net: network to get the parameters (weights and biases)
    :return: list of parameters (weights and biases) of the network
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    """
    Update the parameters of the network with the given parameters (weights and biases)
    :param net: network to set the parameters (weights and biases)
    :param parameters: list of parameters (weights and biases) to set
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    dico = {k: torch.Tensor(v) for k, v in params_dict}
    state_dict = OrderedDict(dico)

    net.load_state_dict(state_dict, strict=True)
    print("Updated model")

def get_model_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    """
    Get the parameters of a PyTorch model as numpy arrays.
    
    Args:
        model: PyTorch model
        
    Returns:
        List of numpy arrays containing model parameters
    """
    return [param.detach().cpu().numpy() for param in model.parameters()]


def set_model_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Set the parameters of a PyTorch model from numpy arrays.
    
    Args:
        model: PyTorch model to update
        parameters: List of numpy arrays containing new parameters
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

