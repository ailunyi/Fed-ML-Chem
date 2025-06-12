"""
utils.py
--------
This module contains miscellaneous utility functions for the fed_ml_lib library.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
import torch
from collections import OrderedDict


def plot_graph(x_data: List[List], y_data: List[List], x_label: str, y_label: str, 
               curve_labels: List[str], title: str, path: str):
    """
    Plot and save a graph with multiple curves.
    
    Args:
        x_data: List of x-axis data for each curve
        y_data: List of y-axis data for each curve
        x_label: Label for x-axis
        y_label: Label for y-axis
        curve_labels: Labels for each curve
        title: Title of the plot
        path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    for i, (x, y, label) in enumerate(zip(x_data, y_data, curve_labels)):
        plt.plot(x, y, label=label)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path + ".png", dpi=300, bbox_inches='tight')
    plt.close()


def save_graphs(path_save: str, local_epoch: int, results: Dict[str, List], end_file: str = ""):
    """
    Save training and validation graphs.

    Args:
        path_save: Path to save the graphs
        local_epoch: Number of epochs
        results: Results dictionary containing accuracy and loss
        end_file: Suffix for the filename
    """
    os.makedirs(path_save, exist_ok=True)
    print(f"Saving graphs in {path_save}")
    
    # Plot training curves (train and validation)
    plot_graph(
        [[*range(local_epoch)]] * 2,
        [results["train_acc"], results["val_acc"]],
        "Epochs", "Accuracy (%)",
        curve_labels=["Training accuracy", "Validation accuracy"],
        title="Accuracy curves",
        path=os.path.join(path_save, "Accuracy_curves" + end_file)
    )

    plot_graph(
        [[*range(local_epoch)]] * 2,
        [results["train_loss"], results["val_loss"]],
        "Epochs", "Loss",
        curve_labels=["Training loss", "Validation loss"], 
        title="Loss curves",
        path=os.path.join(path_save, "Loss_curves" + end_file)
    )


def get_parameters(net) -> List[np.ndarray]:
    """
    Get the parameters of the network.
    
    Args:
        net: Network to get the parameters (weights and biases)
        
    Returns:
        List of parameters (weights and biases) of the network
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    """
    Update the parameters of the network with the given parameters.
    
    Args:
        net: Network to set the parameters (weights and biases)
        parameters: List of parameters (weights and biases) to set
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    dico = {k: torch.Tensor(v) for k, v in params_dict}
    state_dict = OrderedDict(dico)
    net.load_state_dict(state_dict, strict=True)
    print("Updated model parameters")


def save_matrix(y_true, y_pred, path: str, classes: List[str]):
    """
    Save confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        path: Path to save the matrix
        classes: List of class names
    """
    try:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        print("Warning: sklearn or seaborn not available for confusion matrix plotting")


def save_roc(y_true, y_proba, path: str, num_classes: int):
    """
    Save ROC curve plot.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        path: Path to save the ROC curve
        num_classes: Number of classes
    """
    try:
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        if num_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            
    except ImportError:
        print("Warning: sklearn not available for ROC curve plotting")

# Add your utility functions here 