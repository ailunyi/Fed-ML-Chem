import torch
import torch.nn as nn
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder

"""
This file contains the testing functions for the different tasks. (copied from engine.py)
"""

def test(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: Union[torch.nn.Module, Tuple],
         device: torch.device):
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    y_pred = []
    y_true = []
    y_proba = []
    softmax = nn.Softmax(dim=1)

    # Turn on inference context manager
    with torch.inference_mode():
        """
        torch.inference_mode is analogous to torch.no_grad : 
        gets better performance by disabling view tracking and version counter bumps
        """
        # Loop through DataLoader batches
        for images, labels in dataloader:
            # Send data to target device
            images, labels = images.to(device), labels.to(device)

            # 1. Forward pass
            output = model(images)

            # 2. Calculate and accumulate probas
            probas_output = softmax(output)
            y_proba.extend(probas_output.detach().cpu().numpy())

            # 3. Calculate and accumulate loss
            loss = loss_fn(output, labels)
            test_loss += loss.item()

            # 4. Calculate and accumulate accuracy
            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            y_pred.extend(preds)  # Save Prediction
            acc = (preds == labels).mean()
            test_acc += acc

    y_proba = np.array(y_proba)
    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc * 100, y_pred, y_true, y_proba


def test_graph(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: Union[torch.nn.Module, Tuple],
         device: torch.device):
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    y_pred = []
    y_true = []
    y_proba = []
    softmax = nn.Softmax(dim=1)

    # Turn on inference context manager
    with torch.inference_mode():
        """
        torch.inference_mode is analogous to torch.no_grad : 
        gets better performance by disabling view tracking and version counter bumps
        """
        # Loop through DataLoader batches
        for molecules in dataloader:
            # Send data to target device
            x, edge_index, batch, labels = molecules.x.float().to(device), molecules.edge_index.to(device), molecules.batch.to(device), molecules.y.to(device)
            labels = (F.one_hot(labels.squeeze().long(), num_classes=2)).float()

            # 1. Forward pass
            output = model(x, edge_index, batch)
            # 2. Calculate and accumulate probas
            probas_output = softmax(output)
            y_proba.extend(probas_output.detach().cpu().numpy())

            # 3. Calculate and accumulate loss
            loss = loss_fn(output, labels)
            test_loss += loss.item()
            
            # 4. Calculate and accumulate accuracy            
            labels = np.argmax(labels.detach().cpu().numpy(), axis=1)
            y_true.extend(labels)  # Save Truth
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            y_pred.extend(preds)  # Save Prediction
            acc = (preds == labels).mean()
            test_acc += acc

    y_proba = np.array(y_proba)
    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc * 100, y_pred, y_true, y_proba



def test_multimodal(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: Union[torch.nn.Module, Tuple],
         device: torch.device):
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    y_pred = []
    y_true = []
    y_proba = []
    softmax = nn.Softmax(dim=1)

    # Turn on inference context manager
    with torch.inference_mode():
        """
        torch.inference_mode is analogous to torch.no_grad : 
        gets better performance by disabling view tracking and version counter bumps
        """
        # Loop through DataLoader batches
        for data, labels in dataloader:
            # Send data to target device
            data = np.expand_dims(data, axis=0)
            au = torch.from_numpy(data[:, :, :35]).float().to(device)
            mfccs = torch.from_numpy(data[:, :, 35:]).float().to(device)
            labels = labels.float().to(device)
            # 1. Forward pass
            lengths = torch.LongTensor([au.shape[0]] * au.size(1))
            output = model(au, mfccs, lengths)
            # 2. Calculate and accumulate probas
            probas_output = softmax(output)
            y_proba.extend(probas_output.detach().cpu().numpy())

            # 3. Calculate and accumulate loss
            loss = loss_fn(output, labels)
            test_loss += loss.item()

            # 4. Calculate and accumulate accuracy
            labels =  np.argmax(labels.detach().cpu().numpy(), axis=1)      
            y_true.extend(labels)  # Save Truth
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            y_pred.extend(preds)  # Save Prediction
            acc = (preds == labels).mean()
            test_acc += acc

    y_proba = np.array(y_proba)
    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc * 100, y_pred, y_true, y_proba


def test_multimodal_health(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: Union[torch.nn.Module, Tuple],
         device: torch.device):
    """Tests a PyTorch model for multimodal health data.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A tuple of loss functions (one for each modality).
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple containing:
    - test_loss: Average test loss across both modalities
    - test_acc: Average test accuracy across both modalities
    - y_pred: List of predicted classes
    - y_true: List of true classes
    - y_proba: List of class probabilities
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values for both modalities
    test_loss_1, test_loss_2 = 0, 0
    test_acc_1, test_acc_2 = 0, 0
    y_pred_1, y_pred_2 = [], []
    y_true_1, y_true_2 = [], []
    y_proba_1, y_proba_2 = [], []
    softmax = nn.Softmax(dim=1)

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for (data_1, data_2), (labels_1, labels_2) in dataloader:
            # Send data to target device
            data_1, data_2 = data_1.to(device), data_2.to(device)
            labels_1, labels_2 = labels_1.to(device), labels_2.to(device)

            # 1. Forward pass
            output_1, output_2 = model(data_1, data_2)

            # 2. Calculate and accumulate probas
            probas_1, probas_2 = softmax(output_1), softmax(output_2)
            y_proba_1.extend(probas_1.detach().cpu().numpy())
            y_proba_2.extend(probas_2.detach().cpu().numpy())

            # 3. Calculate and accumulate loss
            loss_1 = loss_fn[0](output_1, labels_1)
            loss_2 = loss_fn[1](output_2, labels_2)
            test_loss_1 += loss_1.item()
            test_loss_2 += loss_2.item()

            # 4. Calculate and accumulate accuracy
            labels_1 = labels_1.data.cpu().numpy()
            labels_2 = labels_2.data.cpu().numpy()
            y_true_1.extend(labels_1)
            y_true_2.extend(labels_2)
            preds_1 = np.argmax(output_1.detach().cpu().numpy(), axis=1)
            preds_2 = np.argmax(output_2.detach().cpu().numpy(), axis=1)
            y_pred_1.extend(preds_1)
            y_pred_2.extend(preds_2)
            acc_1 = (preds_1 == labels_1).mean()
            acc_2 = (preds_2 == labels_2).mean()
            test_acc_1 += acc_1
            test_acc_2 += acc_2

    # Adjust metrics to get average loss and accuracy per batch
    test_loss_1 = test_loss_1 / len(dataloader)
    test_loss_2 = test_loss_2 / len(dataloader)
    test_acc_1 = test_acc_1 / len(dataloader) * 100
    test_acc_2 = test_acc_2 / len(dataloader) * 100

    # Convert probability lists to numpy arrays
    y_proba_1 = np.array(y_proba_1)
    y_proba_2 = np.array(y_proba_2)

    return (test_loss_1, test_loss_2), (test_acc_1, test_acc_2), (y_pred_1, y_pred_2), (y_true_1, y_true_2), (y_proba_1, y_proba_2)
