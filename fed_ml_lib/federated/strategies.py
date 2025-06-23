from typing import Dict, List, Tuple, Optional, Any
from flwr.common import Metrics, Scalar
import numpy as np


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Compute weighted average of metrics from multiple clients.
    
    This function aggregates metrics from multiple clients using a weighted average
    where the weight is the number of examples each client contributed.
    
    Args:
        metrics: List of tuples containing (num_examples, metrics_dict) from each client
        
    Returns:s
        Dictionary of aggregated metrics
        
    Example:
        ```python
        client_metrics = [
            (100, {"accuracy": 0.85, "loss": 0.3}),
            (200, {"accuracy": 0.90, "loss": 0.2}),
            (150, {"accuracy": 0.88, "loss": 0.25})
        ]
        avg_metrics = weighted_average(client_metrics)
        # Returns: {"accuracy": 0.88, "loss": 0.233}
        ```
    """
    if not metrics:
        return {}
    
    # Calculate total number of examples
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    if total_examples == 0:
        return {}
    
    # Get all unique metric names
    all_metric_names = set()
    for _, client_metrics in metrics:
        all_metric_names.update(client_metrics.keys())
    
    # Calculate weighted average for each metric
    aggregated_metrics = {}
    
    for metric_name in all_metric_names:
        # Only include clients that have this metric
        relevant_metrics = [
            (num_examples, client_metrics[metric_name])
            for num_examples, client_metrics in metrics
            if metric_name in client_metrics
        ]
        
        if relevant_metrics:
            relevant_total = sum(num_examples for num_examples, _ in relevant_metrics)
            weighted_sum = sum(
                num_examples * metric_value
                for num_examples, metric_value in relevant_metrics
            )
            aggregated_metrics[metric_name] = weighted_sum / relevant_total
    
    return aggregated_metrics


def adaptive_weighted_average(
    metrics: List[Tuple[int, Metrics]], 
    performance_weight: float = 0.7,
    data_weight: float = 0.3
) -> Metrics:
    """
    Compute adaptive weighted average that considers both data size and performance.
    
    This function aggregates metrics using a weighted average that takes into account
    both the number of examples and the performance of each client.
    
    Args:
        metrics: List of tuples containing (num_examples, metrics_dict) from each client
        performance_weight: Weight given to performance-based weighting (0-1)
        data_weight: Weight given to data size-based weighting (0-1)
        
    Returns:
        Dictionary of aggregated metrics
        
    Note:
        performance_weight + data_weight should equal 1.0
    """
    if not metrics:
        return {}
    
    # Normalize weights
    total_weight = performance_weight + data_weight
    if total_weight > 0:
        performance_weight /= total_weight
        data_weight /= total_weight
    else:
        performance_weight = data_weight = 0.5
    
    # Calculate total number of examples
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    if total_examples == 0:
        return {}
    
    # Get all unique metric names
    all_metric_names = set()
    for _, client_metrics in metrics:
        all_metric_names.update(client_metrics.keys())
    
    # Calculate adaptive weighted average for each metric
    aggregated_metrics = {}
    
    for metric_name in all_metric_names:
        # Only include clients that have this metric
        relevant_data = [
            (num_examples, client_metrics)
            for num_examples, client_metrics in metrics
            if metric_name in client_metrics
        ]
        
        if not relevant_data:
            continue
        
        # Calculate weights based on data size
        data_weights = [num_examples / total_examples for num_examples, _ in relevant_data]
        
        # Calculate weights based on performance (higher accuracy = higher weight)
        performance_metrics = [client_metrics.get('accuracy', 0.0) for _, client_metrics in relevant_data]
        
        if performance_metrics and max(performance_metrics) > 0:
            # Normalize performance weights
            max_performance = max(performance_metrics)
            performance_weights = [perf / max_performance for perf in performance_metrics]
        else:
            # Equal performance weights if no accuracy information
            performance_weights = [1.0 / len(relevant_data)] * len(relevant_data)
        
        # Combine data and performance weights
        combined_weights = [
            data_weight * dw + performance_weight * pw
            for dw, pw in zip(data_weights, performance_weights)
        ]
        
        # Normalize combined weights
        total_combined_weight = sum(combined_weights)
        if total_combined_weight > 0:
            combined_weights = [w / total_combined_weight for w in combined_weights]
        
        # Calculate weighted average
        metric_values = [client_metrics[metric_name] for _, client_metrics in relevant_data]
        weighted_average_value = sum(
            weight * value
            for weight, value in zip(combined_weights, metric_values)
        )
        
        aggregated_metrics[metric_name] = weighted_average_value
    
    return aggregated_metrics


def median_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Compute median aggregation of metrics from multiple clients.
    
    This aggregation method is more robust to outliers compared to weighted averaging.
    
    Args:
        metrics: List of tuples containing (num_examples, metrics_dict) from each client
        
    Returns:
        Dictionary of aggregated metrics using median values
    """
    if not metrics:
        return {}
    
    # Get all unique metric names
    all_metric_names = set()
    for _, client_metrics in metrics:
        all_metric_names.update(client_metrics.keys())
    
    # Calculate median for each metric
    aggregated_metrics = {}
    
    for metric_name in all_metric_names:
        # Get all values for this metric
        metric_values = [
            client_metrics[metric_name]
            for _, client_metrics in metrics
            if metric_name in client_metrics
        ]
        
        if metric_values:
            aggregated_metrics[metric_name] = float(np.median(metric_values))
    
    return aggregated_metrics


def trimmed_mean_aggregation(
    metrics: List[Tuple[int, Metrics]], 
    trim_fraction: float = 0.1
) -> Metrics:
    """
    Compute trimmed mean aggregation of metrics from multiple clients.
    
    This aggregation method removes outliers by trimming a fraction of the highest
    and lowest values before computing the mean.
    
    Args:
        metrics: List of tuples containing (num_examples, metrics_dict) from each client
        trim_fraction: Fraction of values to trim from each end (0-0.5)
        
    Returns:
        Dictionary of aggregated metrics using trimmed mean
    """
    if not metrics:
        return {}
    
    # Ensure trim_fraction is valid
    trim_fraction = max(0.0, min(0.5, trim_fraction))
    
    # Get all unique metric names
    all_metric_names = set()
    for _, client_metrics in metrics:
        all_metric_names.update(client_metrics.keys())
    
    # Calculate trimmed mean for each metric
    aggregated_metrics = {}
    
    for metric_name in all_metric_names:
        # Get all values for this metric
        metric_values = [
            client_metrics[metric_name]
            for _, client_metrics in metrics
            if metric_name in client_metrics
        ]
        
        if metric_values:
            # Sort values
            sorted_values = sorted(metric_values)
            n = len(sorted_values)
            
            # Calculate number of values to trim from each end
            trim_count = int(n * trim_fraction)
            
            # Trim values
            if trim_count > 0 and trim_count < n // 2:
                trimmed_values = sorted_values[trim_count:-trim_count]
            else:
                trimmed_values = sorted_values
            
            # Calculate mean of trimmed values
            if trimmed_values:
                aggregated_metrics[metric_name] = float(np.mean(trimmed_values))
    
    return aggregated_metrics


def confidence_weighted_aggregation(
    metrics: List[Tuple[int, Metrics]], 
    confidence_key: str = "confidence"
) -> Metrics:
    """
    Compute confidence-weighted aggregation of metrics from multiple clients.
    
    This aggregation method weights client contributions based on their confidence scores.
    
    Args:
        metrics: List of tuples containing (num_examples, metrics_dict) from each client
        confidence_key: Key in metrics dict that contains confidence scores
        
    Returns:
        Dictionary of aggregated metrics weighted by confidence
    """
    if not metrics:
        return {}
    
    # Get all unique metric names
    all_metric_names = set()
    for _, client_metrics in metrics:
        all_metric_names.update(client_metrics.keys())
    
    # Calculate confidence-weighted average for each metric
    aggregated_metrics = {}
    
    for metric_name in all_metric_names:
        if metric_name == confidence_key:
            continue  # Skip the confidence key itself
        
        # Get values and confidence scores
        weighted_data = []
        for num_examples, client_metrics in metrics:
            if metric_name in client_metrics:
                metric_value = client_metrics[metric_name]
                confidence = client_metrics.get(confidence_key, 1.0)  # Default confidence = 1.0
                weighted_data.append((metric_value, confidence))
        
        if weighted_data:
            # Calculate confidence-weighted average
            total_confidence = sum(confidence for _, confidence in weighted_data)
            
            if total_confidence > 0:
                weighted_sum = sum(
                    value * confidence
                    for value, confidence in weighted_data
                )
                aggregated_metrics[metric_name] = weighted_sum / total_confidence
            else:
                # Fallback to simple average if no confidence scores
                aggregated_metrics[metric_name] = np.mean([value for value, _ in weighted_data])
    
    return aggregated_metrics


def create_custom_aggregation_fn(
    strategy: str = "weighted_average",
    **kwargs
) -> callable:
    """
    Create a custom aggregation function based on the specified strategy.
    
    Args:
        strategy: Aggregation strategy ("weighted_average", "adaptive", "median", "trimmed_mean", "confidence")
        **kwargs: Additional arguments for specific strategies
        
    Returns:
        Aggregation function that can be used in federated strategies
    """
    if strategy == "weighted_average":
        return weighted_average
    elif strategy == "adaptive":
        performance_weight = kwargs.get("performance_weight", 0.7)
        data_weight = kwargs.get("data_weight", 0.3)
        return lambda metrics: adaptive_weighted_average(metrics, performance_weight, data_weight)
    elif strategy == "median":
        return median_aggregation
    elif strategy == "trimmed_mean":
        trim_fraction = kwargs.get("trim_fraction", 0.1)
        return lambda metrics: trimmed_mean_aggregation(metrics, trim_fraction)
    elif strategy == "confidence":
        confidence_key = kwargs.get("confidence_key", "confidence")
        return lambda metrics: confidence_weighted_aggregation(metrics, confidence_key)
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy}") 