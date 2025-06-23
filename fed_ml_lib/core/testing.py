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
    test_loss_mri, test_loss_dna, test_acc_mri, test_acc_dna = 0, 0, 0, 0
    y_pred_mri, y_pred_dna = [], []
    y_true_mri, y_true_dna = [], []
    y_proba_mri, y_proba_dna = [], []

    # Turn on inference context manager
    with torch.inference_mode():
        """
        torch.inference_mode is analogous to torch.no_grad : 
        gets better performance by disabling view tracking and version counter bumps
        """
        # Loop through DataLoader batches
        for (mri_data, dna_data), (mri_labels, dna_labels) in dataloader:
            # Send data to target device
            mri_data, dna_data, mri_labels, dna_labels = [x.to(device) for x in [mri_data, dna_data, mri_labels, dna_labels]]

            # 1. Forward pass
            mri_output, dna_output = model(mri_data, dna_data)

            # 2. Calculate and accumulate probas
            y_proba_mri.extend(mri_output.detach().cpu().numpy())
            y_proba_dna.extend(dna_output.detach().cpu().numpy())

            # 3. Calculate and accumulate loss
            criterion_mri, criterion_dna = loss_fn
            loss_mri = criterion_mri(mri_output, mri_labels)
            loss_dna = criterion_dna(dna_output, dna_labels)                       
            test_loss_mri += loss_mri.item()
            test_loss_dna += loss_dna.item()

            # 4. Calculate and accumulate accuracy
            mri_labels = mri_labels.data.cpu().numpy()
            dna_labels = dna_labels.data.cpu().numpy()
            y_true_mri.extend(mri_labels)  # Save Truth
            y_true_dna.extend(dna_labels)  # Save Truth
            preds_mri = np.argmax(mri_output.detach().cpu().numpy(), axis=1)
            preds_dna = np.argmax(dna_output.detach().cpu().numpy(), axis=1)
            y_pred_mri.extend(preds_mri)  # Save Prediction
            y_pred_dna.extend(preds_dna)  # Save Prediction
            acc_mri = (preds_mri == mri_labels).mean()
            acc_dna = (preds_dna == dna_labels).mean()
            test_acc_mri += acc_mri
            test_acc_dna += acc_dna

    y_proba_mri = np.array(y_proba_mri)
    y_proba_dna = np.array(y_proba_dna)    
    # Adjust metrics to get average loss and accuracy per batch
    test_loss_mri = test_loss_mri / len(dataloader)
    test_acc_mri = test_acc_mri / len(dataloader) * 100
    test_loss_dna = test_loss_dna / len(dataloader)
    test_acc_dna = test_acc_dna / len(dataloader) * 100
    return (test_loss_mri, test_loss_dna), (test_acc_mri, test_acc_dna), (y_pred_mri, y_pred_dna), (y_true_mri, y_true_dna), (y_proba_mri, y_proba_dna)
