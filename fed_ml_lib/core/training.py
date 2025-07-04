import torch
import torch.nn as nn
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from .testing import test, test_graph, test_multimodal, test_multimodal_health

"""
This file contains the training functions for the different tasks. (copied from engine.py)
"""

def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: Union[torch.nn.Module, Tuple],
               optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in training mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (images, labels) in enumerate(dataloader):

        # Send data to target device
        images, labels = images.to(device), labels.to(device)

        # 1. Optimizer zero grad
        optimizer.zero_grad()  # Sets the gradients of all optimized torch.Tensor to zero.

        # 2. Forward pass
        output = model(images)

        # 3. Calculate  and accumulate loss
        loss = loss_fn(output, labels)
        train_loss += loss.item()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
        train_acc += (y_pred_class == labels).sum().item()/len(output)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc * 100


def train_step_graph(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: Union[torch.nn.Module, Tuple],
               optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in training mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for molecules in dataloader:

        # Send data to target device
        x, edge_index, batch, labels = molecules.x.float().to(device), molecules.edge_index.to(device), molecules.batch.to(device), molecules.y.to(device)
        labels = (F.one_hot(labels.squeeze().long(), num_classes=2)).float()        

        # 1. Optimizer zero grad
        optimizer.zero_grad()  # Sets the gradients of all optimized torch.Tensor to zero.

        # 2. Forward pass
        output = model(x, edge_index, batch)

        # 3. Calculate and accumulate loss
        loss = loss_fn(output, labels)
        train_loss += loss.item()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()
        
        labels = torch.argmax(labels, axis=1)
        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
        train_acc += (y_pred_class == labels).sum().item()/len(output)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc * 100


def train_step_multimodal(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: Union[torch.nn.Module, Tuple],
               optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in training mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for data, labels in dataloader:

        # Send data to target device
        data = np.expand_dims(data, axis=0)
        au = torch.from_numpy(data[:, :, :35]).float().to(device)
        mfccs = torch.from_numpy(data[:, :, 35:]).float().to(device)
        labels = labels.float().to(device)

        # 1. Optimizer zero grad
        optimizer.zero_grad()  # Sets the gradients of all optimized torch.Tensor to zero.

        # 2. Forward pass
        lengths = torch.LongTensor([au.shape[0]] * au.size(1))
        output = model(au, mfccs, lengths)

        # 3. Calculate  and accumulate loss
        loss = loss_fn(output, labels)
        train_loss += loss.item()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
        labels = np.argmax(labels.detach().cpu().numpy(), axis=1)
        train_acc += (y_pred_class == labels).sum().item()/len(output)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc * 100


def train_step_multimodal_health(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: Union[torch.nn.Module, Tuple],
               optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in training mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss_mri, train_loss_dna, train_acc_mri, train_acc_dna = 0, 0, 0, 0

    # Loop through data loader data batches
    for (mri_data, dna_data), (mri_labels, dna_labels) in dataloader:

        # Send data to target device
        mri_data, dna_data, mri_labels, dna_labels = [x.to(device) for x in [mri_data, dna_data, mri_labels, dna_labels]]

        # 1. Optimizer zero grad
        optimizer.zero_grad()  # Sets the gradients of all optimized torch.Tensor to zero.

        # 2. Forward pass
        mri_output, dna_output = model(mri_data, dna_data)

        # 3. Calculate  and accumulate loss
        criterion_mri, criterion_dna = loss_fn
        loss_mri = criterion_mri(mri_output, mri_labels)
        loss_dna = criterion_dna(dna_output, dna_labels)   
        train_loss_mri += loss_mri.item()
        train_loss_dna += loss_dna.item()
        loss = loss_mri + loss_dna

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class_mri = torch.argmax(mri_output, dim=1)
        y_pred_class_dna = torch.argmax(dna_output, dim=1)
        train_acc_mri += (y_pred_class_mri == mri_labels).sum().item()/len(mri_output)
        train_acc_dna += (y_pred_class_dna == dna_labels).sum().item()/len(dna_output)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss_mri = train_loss_mri / len(dataloader)
    train_acc_mri = train_acc_mri / len(dataloader) * 100
    train_loss_dna = train_loss_dna / len(dataloader)
    train_acc_dna = train_acc_dna / len(dataloader) * 100
    return (train_loss_mri, train_loss_dna), (train_acc_mri, train_acc_dna) 


def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, loss_fn: Union[torch.nn.Module, Tuple],
          epochs: int, device: torch.device, task: Optional[str] = None) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    task: An optional string indicating the task (default is None).

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  val_loss: [...],
                  val_acc: [...]}
    """
    # Create empty results dictionary
    if isinstance(loss_fn, Tuple):
        results = {"train_loss_mri": [], "train_acc_mri": [], "train_loss_dna": [], "train_acc_dna": [], "val_loss_mri": [], "val_acc_mri": [], "val_loss_dna": [], "val_acc_dna": []}
    else:
        results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs), colour="BLUE"):
        # Select functions based on the task
        if task == "multimodal_health" and isinstance(loss_fn, Tuple):
            train_step_fn = train_step_multimodal_health
            test_fn = test_multimodal_health
        elif task == "Multimodal" and isinstance(loss_fn, Tuple):
            train_step_fn = train_step_multimodal_health
            test_fn = test_multimodal_health
        elif task == "Multimodal":
            train_step_fn = train_step_multimodal
            test_fn = test_multimodal
        elif task == "Graph":
            train_step_fn = train_step_graph
            test_fn = test_graph
        else:
            train_step_fn = train_step
            test_fn = test

        # Perform training and validation
        train_loss, train_acc = train_step_fn(model=model, dataloader=train_dataloader, loss_fn=loss_fn,
                                            optimizer=optimizer, device=device)
        val_loss, val_acc, *_ = test_fn(model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device)

        if isinstance(loss_fn, Tuple):
            # Print out what's happening
            train_loss_mri, train_loss_dna = train_loss
            train_acc_mri, train_acc_dna = train_acc
            val_loss_mri, val_loss_dna = val_loss
            val_acc_mri, val_acc_dna = val_acc

            print(f"\tTrain Epoch: {epoch + 1} \t"
                f"Train_loss_mri: {train_loss_mri:.4f} | "
                f"Train_acc_mri: {train_acc_mri:.4f} | "
                f"Train_loss_dna: {train_loss_dna:.4f} | "
                f"Train_acc_dna: {train_acc_dna:.4f} | "
                f"Validation_loss_mri: {val_loss_mri:.4f} | "
                f"Validation_acc_mri: {val_acc_mri:.4f} | "
                f"Validation_loss_dna: {val_loss_dna:.4f} | "
                f"Validation_acc_dna: {val_acc_dna:.4f}"
            )

            # Update results dictionary
            results["train_loss_mri"].append(train_loss_mri)
            results["train_acc_mri"].append(train_acc_mri)
            results["train_loss_dna"].append(train_loss_dna)
            results["train_acc_dna"].append(train_acc_dna)
            results["val_loss_mri"].append(val_loss_mri)
            results["val_acc_mri"].append(val_acc_mri)
            results["val_loss_dna"].append(val_loss_dna)
            results["val_acc_dna"].append(val_acc_dna)
        else:
            
            # Print out what's happening
            print(
            f"\tTrain Epoch: {epoch + 1} \t"
            f"Train_loss: {train_loss:.4f} | "
            f"Train_acc: {train_acc:.4f} % | "
            f"Validation_loss: {val_loss:.4f} | "
            f"Validation_acc: {val_acc:.4f} %"
            )

            # Update results dictionary
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["val_loss"].append(val_loss)
            results["val_acc"].append(val_acc)

    # Return the filled results at the end of the epochs
    return results