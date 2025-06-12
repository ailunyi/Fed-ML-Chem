"""
server.py
---------
This module contains the Flower server logic and custom strategies for the fed_ml_lib library.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable, Union
from flwr.common import (
    Metrics, EvaluateIns, EvaluateRes, FitIns, FitRes, MetricsAggregationFn,
    Scalar, NDArrays, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
import numpy as np

from .engine import evaluate_model
from .utils import save_confusion_matrix, save_roc_curve


class FedCustom(Strategy):
    """
    Custom federated learning strategy with advanced features.
    
    Features:
    - Dynamic learning rate assignment to different client groups
    - Custom aggregation methods
    - Server-side evaluation
    - Flexible client sampling
    """
    
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        learning_rate_strategy: str = "uniform",  # "uniform", "split", "adaptive"
        base_learning_rate: float = 0.001,
        higher_learning_rate: float = 0.003
    ):
        """
        Initialize the custom federated strategy.
        
        Args:
            fraction_fit: Fraction of clients to sample for training
            fraction_evaluate: Fraction of clients to sample for evaluation
            min_fit_clients: Minimum number of clients for training
            min_evaluate_clients: Minimum number of clients for evaluation
            min_available_clients: Minimum number of available clients
            evaluate_fn: Server-side evaluation function
            on_fit_config_fn: Function to configure training
            on_evaluate_config_fn: Function to configure evaluation
            accept_failures: Whether to accept failed clients
            initial_parameters: Initial model parameters
            fit_metrics_aggregation_fn: Function to aggregate training metrics
            evaluate_metrics_aggregation_fn: Function to aggregate evaluation metrics
            learning_rate_strategy: Strategy for learning rate assignment
            base_learning_rate: Base learning rate
            higher_learning_rate: Higher learning rate for split strategy
        """
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.learning_rate_strategy = learning_rate_strategy
        self.base_learning_rate = base_learning_rate
        self.higher_learning_rate = higher_learning_rate
    
    def __repr__(self) -> str:
        return f"FedCustom (accept_failures={self.accept_failures})"
    
    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters
    
    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients
    
    def num_evaluate_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        
        # Create base config
        config = {"server_round": server_round, "local_epochs": 1}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        # Apply learning rate strategy
        fit_configurations = []
        n_clients = len(clients)
        
        for idx, client in enumerate(clients):
            client_config = config.copy()
            
            if self.learning_rate_strategy == "split":
                # Split clients into two groups with different learning rates
                half_clients = n_clients // 2
                client_config["learning_rate"] = (
                    self.base_learning_rate if idx < half_clients else self.higher_learning_rate
                )
            elif self.learning_rate_strategy == "adaptive":
                # Adaptive learning rate based on client performance (placeholder)
                client_config["learning_rate"] = self.base_learning_rate
            else:  # uniform
                client_config["learning_rate"] = self.base_learning_rate
            
            fit_configurations.append((client, FitIns(parameters, client_config)))
        
        return fit_configurations
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        
        # Sample clients
        sample_size, min_num_clients = self.num_evaluate_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        
        # Create config
        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        
        # Return client/config pairs
        evaluate_ins = EvaluateIns(parameters, config)
        return [(client, evaluate_ins) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Aggregate parameters
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        
        # Aggregate custom metrics if aggregation function was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif results:
            # Simple averaging of metrics
            total_examples = sum([res.num_examples for _, res in results])
            metrics_aggregated = {}
            
            # Aggregate common metrics
            for metric_name in ["train_loss", "train_accuracy", "val_loss", "val_accuracy"]:
                if all(metric_name in res.metrics for _, res in results):
                    weighted_sum = sum([
                        res.metrics[metric_name] * res.num_examples 
                        for _, res in results
                    ])
                    metrics_aggregated[metric_name] = weighted_sum / total_examples
        
        return parameters_aggregated, metrics_aggregated
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        if not results:
            return None, {}
        
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        # Aggregate loss
        loss_aggregated = weighted_loss_avg([
            (evaluate_res.num_examples, evaluate_res.loss)
            for _, evaluate_res in results
        ])
        
        # Aggregate custom metrics if aggregation function was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif results:
            # Simple averaging of metrics
            total_examples = sum([res.num_examples for _, res in results])
            
            # Aggregate common metrics
            for metric_name in ["accuracy"]:
                if all(metric_name in res.metrics for _, res in results):
                    weighted_sum = sum([
                        res.metrics[metric_name] * res.num_examples 
                        for _, res in results
                    ])
                    metrics_aggregated[metric_name] = weighted_sum / total_examples
        
        return loss_aggregated, metrics_aggregated
    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            return None
        
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        
        loss, metrics = eval_res
        return loss, metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate metrics using weighted average.
    
    Args:
        metrics: List of (num_examples, metrics_dict) tuples
        
    Returns:
        Aggregated metrics dictionary
    """
    # Multiply accuracy of each client by number of examples used
    total_examples = sum([num_examples for num_examples, _ in metrics])
    
    # Calculate weighted averages
    weighted_metrics = {}
    
    # Get all metric names from first client
    if metrics:
        metric_names = metrics[0][1].keys()
        
        for metric_name in metric_names:
            weighted_sum = sum([
                num_examples * m[metric_name] 
                for num_examples, m in metrics 
                if metric_name in m
            ])
            weighted_metrics[metric_name] = weighted_sum / total_examples
    
    return weighted_metrics


def create_server_evaluate_fn(
    model: nn.Module,
    test_loader,
    device: torch.device,
    classes: Optional[List[str]] = None,
    save_results: Optional[str] = None
) -> Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]:
    """
    Create a server-side evaluation function.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: Test data loader
        device: Device to run evaluation on
        classes: List of class names
        save_results: Directory to save results
        
    Returns:
        Evaluation function for server-side evaluation
    """
    def evaluate_fn(
        server_round: int, 
        parameters_ndarrays: NDArrays, 
        config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Server-side evaluation function."""
        # Set model parameters
        params_dict = zip(model.state_dict().keys(), parameters_ndarrays)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        
        # Evaluate
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, y_pred, y_true, y_proba = evaluate_model(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device
        )
        
        # Save results if requested
        if save_results and classes and server_round % 5 == 0:  # Save every 5 rounds
            try:
                import os
                os.makedirs(save_results, exist_ok=True)
                
                save_confusion_matrix(
                    y_true, y_pred,
                    save_path=os.path.join(save_results, f"server_round_{server_round}_confusion_matrix.png"),
                    class_names=classes
                )
                
                if len(classes) == 2:
                    save_roc_curve(
                        y_true, y_proba[:, 1],
                        save_path=os.path.join(save_results, f"server_round_{server_round}_roc_curve.png")
                    )
            except Exception as e:
                print(f"Warning: Could not save server evaluation results: {e}")
        
        print(f"[Server] Round {server_round} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        
        return float(test_loss), {"accuracy": float(test_acc)}
    
    return evaluate_fn


def get_on_fit_config_fn(
    local_epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> Callable[[int], Dict[str, Scalar]]:
    """
    Create a function to configure training rounds.
    
    Args:
        local_epochs: Number of local epochs per round
        batch_size: Batch size for training
        learning_rate: Learning rate for training
        
    Returns:
        Configuration function
    """
    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return training configuration dict for each round."""
        config = {
            "server_round": server_round,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }
        return config
    
    return fit_config


def get_on_evaluate_config_fn() -> Callable[[int], Dict[str, Scalar]]:
    """
    Create a function to configure evaluation rounds.
    
    Returns:
        Configuration function
    """
    def evaluate_config(server_round: int) -> Dict[str, Scalar]:
        """Return evaluation configuration dict for each round."""
        return {"server_round": server_round}
    
    return evaluate_config 