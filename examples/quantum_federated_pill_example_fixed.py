"""
Fixed Quantum Federated PILL Training Example using Fed-ML-Lib
Properly integrated with the library's modular system, matching centralized_pill_example architecture
"""
import torch
import flwr as fl
from flwr.server.strategy import FedAvg
import os
import time

from fed_ml_lib.config import pill_cnn
from fed_ml_lib.models.modular import create_model
from fed_ml_lib.data.loaders import load_datasets
from fed_ml_lib.federated.client import FlowerClient
from fed_ml_lib.federated.utils import get_parameters2, set_parameters
from fed_ml_lib.core.visualization import save_matrix
from fed_ml_lib.core.testing import test

def client_fn(cid: str):
    """Create a Flower client with quantum neural network for PILL classification using library's modular system."""
    # Use same config as working centralized example but with quantum enabled
    config = pill_cnn(
        name="quantum_pill_federated", 
        epochs=10, 
        learning_rate=0.001, 
        batch_size=32,
        use_quantum=True,
        n_qubits=2,
        quantum_layers=['classifier'],
        quantum_circuit='basic_entangler'
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load federated datasets for PILL (same as centralized)
    trainloaders, valloaders, testloader = load_datasets(
        num_clients=4,
        batch_size=config['batch_size'],
        resize=224,  # Match centralized example
        seed=42,
        num_workers=0,
        splitter=20,
        dataset='PILL',
        data_path="data/"
    )
    
    # Create model using library's modular system (matching centralized approach)
    model_config = {
        'base_architecture': 'pretrained_cnn',  # Same as centralized
        'num_classes': 2,
        'freeze_layers': 23,  # Same as centralized
        'input_shape': (3, 224, 224),  # Same as centralized
        'use_quantum': True,  # Enable quantum
        'n_qubits': config['n_qubits'],
        'quantum_layers': config['quantum_layers'],
        'quantum_circuit': config['quantum_circuit']
    }
    
    model = create_model(**model_config).to(device)
    
    class_names = ['bad', 'good']
    
    return FlowerClient(
        cid=cid,
        net=model,
        trainloader=trainloaders[int(cid)],
        valloader=valloaders[int(cid)],
        device=device,
        batch_size=config['batch_size'],
        save_results=f"results/quantum_federated_pill_fixed/client_{cid}/",
        matrix_path="confusion_matrix.png",
        roc_path="roc.png",
        yaml_path=None,
        classes=class_names
    )

def evaluate_fn(server_round, parameters, config):
    """Server-side evaluation function with quantum model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test dataset (same as centralized)
    _, _, testloader = load_datasets(
        num_clients=1,
        batch_size=32,
        resize=224,
        seed=42,
        num_workers=0,
        splitter=20,
        dataset='PILL',
        data_path="data/"
    )
    
    # Create model using library's modular system (matching centralized approach)
    model_config = {
        'base_architecture': 'pretrained_cnn',  # Same as centralized
        'num_classes': 2,
        'freeze_layers': 23,  # Same as centralized
        'input_shape': (3, 224, 224),  # Same as centralized
        'use_quantum': True,  # Enable quantum
        'n_qubits': 2,
        'quantum_layers': ['classifier'],
        'quantum_circuit': 'basic_entangler'
    }
    
    model = create_model(**model_config).to(device)
    
    # Load aggregated parameters
    set_parameters(model, parameters)
    
    # Evaluate model
    loss, accuracy, y_pred, y_true, y_proba = test(
        model=model,
        dataloader=testloader,
        loss_fn=torch.nn.CrossEntropyLoss(),
        device=device
    )
    
    # Save evaluation results
    os.makedirs("results/quantum_federated_pill_fixed/server/", exist_ok=True)
    class_names = ['bad', 'good']
    save_matrix(y_true, y_pred, "results/quantum_federated_pill_fixed/server/confusion_matrix.png", class_names)
    
    print(f"Quantum Server evaluation - Round {server_round}: Loss {loss:.4f}, Accuracy {accuracy:.2f}%")
    return loss, {"accuracy": accuracy}

def main():
    """Main quantum federated learning orchestration."""
    # Use same config as working centralized example but with quantum enabled
    config = pill_cnn(
        name="quantum_pill_federated", 
        epochs=10, 
        learning_rate=0.001, 
        batch_size=32,
        use_quantum=True,
        n_qubits=2,
        quantum_layers=['classifier'],
        quantum_circuit='basic_entangler'
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create initial model using library's modular system (matching centralized approach)
    initial_model_config = {
        'base_architecture': 'pretrained_cnn',  # Same as centralized
        'num_classes': 2,
        'freeze_layers': 23,  # Same as centralized
        'input_shape': (3, 224, 224),  # Same as centralized
        'use_quantum': True,  # Enable quantum
        'n_qubits': config['n_qubits'],
        'quantum_layers': config['quantum_layers'],
        'quantum_circuit': config['quantum_circuit']
    }
    
    initial_model = create_model(**initial_model_config).to(device)
    
    # Configure federated learning strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=4,
        min_evaluate_clients=2,
        min_available_clients=4,
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters2(initial_model)),
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=lambda round: {
            "learning_rate": str(config['learning_rate']),
            "batch_size": str(config['batch_size']),
            "server_round": round,
            "local_epochs": config['epochs']
        }
    )
    
    # Configure client resources
    client_resources = {"num_cpus": 1}
    if device.type == "cuda":
        client_resources["num_gpus"] = 0.25
    
    print("="*50)
    print("FIXED QUANTUM FEDERATED LEARNING - PILL CLASSIFICATION")
    print("="*50)
    print(f"Device: {device}")
    print(f"Clients: 4")
    print(f"Rounds: 5")
    print(f"Dataset: PILL (Binary Classification)")
    print(f"Architecture: Pretrained CNN + Quantum Layer (Library Modular)")
    print(f"Quantum: {config['n_qubits']} qubits, {config['quantum_layers']} layers")
    print(f"Matching centralized_pill_example.py architecture")
    
    start_time = time.time()
    
    # Start federated learning
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=4,
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
        client_resources=client_resources
    )
    
    print(f"Quantum federated PILL training completed in {time.time() - start_time:.2f} seconds")
    print("Results saved to results/quantum_federated_pill_fixed/")

if __name__ == "__main__":
    main() 