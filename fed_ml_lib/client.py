"""
client.py
---------
This module contains the Flower client logic for the fed_ml_lib library.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any, Union
import flwr as fl
from flwr.common import NDArrays, Scalar
import numpy as np

from .engine import train_one_epoch, evaluate_model
from .utils import save_confusion_matrix, save_roc_curve, plot_training_curves


class FedMLClient(fl.client.NumPyClient):
    """
    Flower client for federated learning with support for classical and quantum models.
    
    Supports:
    - Standard image classification
    - Quantum neural networks
    - Multimodal learning (future extension)
    """
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        save_results: Optional[str] = None,
        classes: Optional[List[str]] = None,
        task_type: str = "classification"
    ):
        """
        Initialize the federated client.
        
        Args:
            client_id: Unique identifier for this client
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to run training on
            save_results: Directory to save results (optional)
            classes: List of class names for evaluation
            task_type: Type of task ("classification", "multimodal")
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_results = save_results
        self.classes = classes or []
        self.task_type = task_type
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        if save_results:
            os.makedirs(save_results, exist_ok=True)
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Get model parameters as numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from numpy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Train the model on local data.
        
        Args:
            parameters: Global model parameters
            config: Training configuration
            
        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        # Extract configuration
        server_round = int(config.get("server_round", 1))
        local_epochs = int(config.get("local_epochs", 1))
        learning_rate = float(config.get("learning_rate", 0.001))
        
        print(f"[Client {self.client_id}, Round {server_round}] Starting training with lr={learning_rate}")
        
        # Set parameters
        self.set_parameters(parameters)
        
        # Setup optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Train for specified epochs
        epoch_results = []
        for epoch in range(local_epochs):
            train_loss, train_acc = train_one_epoch(
                model=self.model,
                train_loader=self.train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=self.device
            )
            
            # Validate
            val_loss, val_acc, _, _, _ = evaluate_model(
                model=self.model,
                data_loader=self.val_loader,
                criterion=criterion,
                device=self.device
            )
            
            epoch_results.append({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
            
            print(f"[Client {self.client_id}] Epoch {epoch+1}/{local_epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Update training history
        for result in epoch_results:
            for key, value in result.items():
                self.training_history[key].append(value)
        
        # Save training curves if requested
        if self.save_results:
            self._save_training_results(server_round)
        
        # Return updated parameters and metrics
        metrics = {
            "train_loss": float(epoch_results[-1]['train_loss']),
            "train_accuracy": float(epoch_results[-1]['train_acc']),
            "val_loss": float(epoch_results[-1]['val_loss']),
            "val_accuracy": float(epoch_results[-1]['val_acc'])
        }
        
        return self.get_parameters(config), len(self.train_loader.dataset), metrics
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the model on local validation data.
        
        Args:
            parameters: Global model parameters
            config: Evaluation configuration
            
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        print(f"[Client {self.client_id}] Starting evaluation")
        
        # Set parameters
        self.set_parameters(parameters)
        
        # Evaluate
        criterion = nn.CrossEntropyLoss()
        val_loss, val_acc, y_pred, y_true, y_proba = evaluate_model(
            model=self.model,
            data_loader=self.val_loader,
            criterion=criterion,
            device=self.device
        )
        
        # Save evaluation results if requested
        if self.save_results and len(self.classes) > 0:
            self._save_evaluation_results(y_true, y_pred, y_proba)
        
        metrics = {
            "accuracy": float(val_acc),
            "num_examples": len(self.val_loader.dataset)
        }
        
        print(f"[Client {self.client_id}] Evaluation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        
        return float(val_loss), len(self.val_loader.dataset), metrics
    
    def _save_training_results(self, round_num: int) -> None:
        """Save training curves and results."""
        try:
            # Plot training curves
            plot_training_curves(
                self.training_history,
                save_path=os.path.join(self.save_results, f"client_{self.client_id}_round_{round_num}_training.png"),
                title=f"Client {self.client_id} Training Progress"
            )
        except Exception as e:
            print(f"Warning: Could not save training results: {e}")
    
    def _save_evaluation_results(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> None:
        """Save evaluation results including confusion matrix and ROC curve."""
        try:
            # Save confusion matrix
            save_confusion_matrix(
                y_true, y_pred,
                save_path=os.path.join(self.save_results, f"client_{self.client_id}_confusion_matrix.png"),
                class_names=self.classes
            )
            
            # Save ROC curve for binary classification
            if len(self.classes) == 2:
                save_roc_curve(
                    y_true, y_proba[:, 1],
                    save_path=os.path.join(self.save_results, f"client_{self.client_id}_roc_curve.png")
                )
        except Exception as e:
            print(f"Warning: Could not save evaluation results: {e}")


class MultimodalFedMLClient(FedMLClient):
    """
    Extended client for multimodal federated learning.
    
    Handles models that take multiple inputs (e.g., images + sequences).
    """
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        modality_names: List[str],
        save_results: Optional[str] = None,
        classes: Optional[List[str]] = None
    ):
        """
        Initialize multimodal federated client.
        
        Args:
            client_id: Unique identifier for this client
            model: Multimodal PyTorch model
            train_loader: Training data loader (returns multiple inputs)
            val_loader: Validation data loader (returns multiple inputs)
            device: Device to run training on
            modality_names: Names of different modalities (e.g., ['mri', 'dna'])
            save_results: Directory to save results (optional)
            classes: List of class names for evaluation
        """
        super().__init__(client_id, model, train_loader, val_loader, device, save_results, classes, "multimodal")
        self.modality_names = modality_names
    
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train multimodal model - placeholder for future implementation."""
        # For now, fall back to standard training
        # TODO: Implement multimodal training logic
        return super().fit(parameters, config)
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate multimodal model - placeholder for future implementation."""
        # For now, fall back to standard evaluation
        # TODO: Implement multimodal evaluation logic
        return super().evaluate(parameters, config)


def create_client(
    client_id: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    client_type: str = "standard",
    **kwargs
) -> Union[FedMLClient, MultimodalFedMLClient]:
    """
    Factory function to create appropriate client type.
    
    Args:
        client_id: Unique identifier for this client
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run training on
        client_type: Type of client ("standard", "multimodal")
        **kwargs: Additional arguments for specific client types
        
    Returns:
        Appropriate client instance
    """
    if client_type == "multimodal":
        return MultimodalFedMLClient(
            client_id=client_id,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            **kwargs
        )
    else:
        return FedMLClient(
            client_id=client_id,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            **kwargs
        ) 