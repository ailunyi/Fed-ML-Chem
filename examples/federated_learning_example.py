"""
Federated Learning Example using fed_ml_lib
===========================================

This example demonstrates how to use the fed_ml_lib library for federated learning
with both classical and quantum models.

Features demonstrated:
- Federated data partitioning (IID, non-IID, Dirichlet)
- Custom federated strategies
- Classical and quantum model training
- Server-side evaluation
- Results visualization
"""

import os
import torch
import flwr as fl
from flwr.common import ndarrays_to_parameters
from typing import List, Dict, Any

# Import fed_ml_lib components
from fed_ml_lib.models import create_model
from fed_ml_lib.datasets import create_federated_data_loaders, get_dataset_info
from fed_ml_lib.client import create_client
from fed_ml_lib.server import FedCustom, weighted_average, create_server_evaluate_fn, get_on_fit_config_fn
from fed_ml_lib.config import load_experiment_config, ExperimentConfig
from fed_ml_lib.utils import get_device


def run_federated_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run a complete federated learning experiment.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dictionary containing experiment results
    """
    print(f"🚀 Starting Federated Learning Experiment")
    print(f"📊 Dataset: {config.data.dataset_name}")
    print(f"🤖 Model: {config.model.model_type}")
    print(f"👥 Clients: {config.training.num_clients}")
    print(f"🔄 Rounds: {config.training.num_rounds}")
    
    # Setup device
    device = get_device(config.training.device)
    print(f"💻 Device: {device}")
    
    # Get dataset information
    dataset_info = get_dataset_info(config.data.dataset_name, config.data.data_path)
    num_classes = dataset_info['num_classes']
    classes = dataset_info['classes']
    
    print(f"📈 Classes: {classes} ({num_classes} total)")
    
    # Create federated data loaders
    print("📦 Creating federated data loaders...")
    client_train_loaders, client_val_loaders, test_loader = create_federated_data_loaders(
        dataset_name=config.data.dataset_name,
        data_path=config.data.data_path,
        num_clients=config.training.num_clients,
        batch_size=config.training.batch_size,
        resize=config.data.resize,
        seed=config.training.seed,
        partition_strategy=config.data.partition_strategy
    )
    
    # Create global model for server-side evaluation
    print(f"🏗️ Creating {config.model.model_type} model...")
    global_model = create_model(
        model_type=config.model.model_type,
        num_classes=num_classes,
        **config.model.model_params
    )
    
    # Create server-side evaluation function
    evaluate_fn = create_server_evaluate_fn(
        model=global_model,
        test_loader=test_loader,
        device=device,
        classes=classes,
        save_results=config.training.save_results
    )
    
    # Create federated strategy
    print("⚙️ Setting up federated strategy...")
    strategy = FedCustom(
        fraction_fit=config.training.fraction_fit,
        fraction_evaluate=config.training.fraction_evaluate,
        min_fit_clients=config.training.min_fit_clients,
        min_evaluate_clients=config.training.min_evaluate_clients,
        min_available_clients=config.training.min_available_clients,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=get_on_fit_config_fn(
            local_epochs=config.training.local_epochs,
            batch_size=config.training.batch_size,
            learning_rate=config.training.learning_rate
        ),
        accept_failures=True,
        initial_parameters=ndarrays_to_parameters([
            val.cpu().numpy() for _, val in global_model.state_dict().items()
        ]),
        evaluate_metrics_aggregation_fn=weighted_average,
        learning_rate_strategy=config.training.learning_rate_strategy,
        base_learning_rate=config.training.learning_rate,
        higher_learning_rate=config.training.learning_rate * 3
    )
    
    # Create client function
    def client_fn(cid: str):
        """Create a client for federated learning."""
        client_id = int(cid)
        
        # Create model for this client
        model = create_model(
            model_type=config.model.model_type,
            num_classes=num_classes,
            **config.model.model_params
        )
        
        # Get client's data loaders
        train_loader = client_train_loaders[client_id]
        val_loader = client_val_loaders[client_id]
        
        # Create save directory for this client
        client_save_dir = None
        if config.training.save_results:
            client_save_dir = os.path.join(config.training.save_results, f"client_{client_id}")
        
        # Create and return client
        return create_client(
            client_id=cid,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            save_results=client_save_dir,
            classes=classes
        )
    
    # Start federated learning simulation
    print("🌐 Starting federated learning simulation...")
    print(f"🔄 Running {config.training.num_rounds} rounds...")
    
    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config.training.num_clients,
        config=fl.server.ServerConfig(num_rounds=config.training.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.1 if device.type == "cuda" else 0}
    )
    
    print("✅ Federated learning completed!")
    
    # Extract results
    results = {
        'history': history,
        'final_accuracy': history.metrics_centralized.get('accuracy', [])[-1][1] if history.metrics_centralized.get('accuracy') else 0,
        'config': config,
        'dataset_info': dataset_info
    }
    
    print(f"🎯 Final Test Accuracy: {results['final_accuracy']:.4f}")
    
    return results


def main():
    """Main function to run federated learning experiments."""
    
    # Example configurations for different scenarios
    configs = {
        "classical_iid": {
            "model": {
                "model_type": "vgg16",
                "model_params": {}
            },
            "data": {
                "dataset_name": "PILL",
                "data_path": "./data/",
                "partition_strategy": "iid",
                "resize": 224
            },
            "training": {
                "num_clients": 5,
                "num_rounds": 10,
                "local_epochs": 2,
                "batch_size": 16,
                "learning_rate": 0.001,
                "learning_rate_strategy": "uniform",
                "fraction_fit": 1.0,
                "fraction_evaluate": 0.5,
                "min_fit_clients": 3,
                "min_evaluate_clients": 2,
                "min_available_clients": 3,
                "device": "auto",
                "seed": 42,
                "save_results": "./results/federated_classical_iid"
            }
        },
        
        "quantum_non_iid": {
            "model": {
                "model_type": "hybrid_cnn_qnn",
                "model_params": {
                    "n_qubits": 4,
                    "n_layers": 2
                }
            },
            "data": {
                "dataset_name": "PILL",
                "data_path": "./data/",
                "partition_strategy": "non_iid",
                "resize": 64  # Smaller images for quantum models
            },
            "training": {
                "num_clients": 3,
                "num_rounds": 5,
                "local_epochs": 1,
                "batch_size": 8,  # Smaller batches for quantum
                "learning_rate": 0.01,
                "learning_rate_strategy": "split",
                "fraction_fit": 1.0,
                "fraction_evaluate": 1.0,
                "min_fit_clients": 2,
                "min_evaluate_clients": 2,
                "min_available_clients": 2,
                "device": "auto",
                "seed": 42,
                "save_results": "./results/federated_quantum_non_iid"
            }
        }
    }
    
    # Run experiments
    results = {}
    
    for experiment_name, config_dict in configs.items():
        print(f"\n{'='*60}")
        print(f"🧪 Running Experiment: {experiment_name}")
        print(f"{'='*60}")
        
        try:
            # Convert dict to ExperimentConfig
            config = ExperimentConfig.from_dict(config_dict)
            
            # Run experiment
            experiment_results = run_federated_experiment(config)
            results[experiment_name] = experiment_results
            
            print(f"✅ Experiment {experiment_name} completed successfully!")
            
        except Exception as e:
            print(f"❌ Experiment {experiment_name} failed: {e}")
            results[experiment_name] = {"error": str(e)}
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    for experiment_name, result in results.items():
        if "error" in result:
            print(f"❌ {experiment_name}: FAILED - {result['error']}")
        else:
            accuracy = result.get('final_accuracy', 0)
            print(f"✅ {experiment_name}: Final Accuracy = {accuracy:.4f}")
    
    print(f"\n🎉 All experiments completed!")
    
    return results


if __name__ == "__main__":
    # Check if required packages are available
    try:
        import flwr
        import torch
        print("✅ All required packages are available")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install with: pip install flwr torch torchvision")
        exit(1)
    
    # Run experiments
    results = main() 