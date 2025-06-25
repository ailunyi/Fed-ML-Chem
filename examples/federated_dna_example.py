"""
Federated DNA Training Example using Fed-ML-Lib
"""
import torch
import flwr as fl
from flwr.server.strategy import FedAvg
import os
import time

from fed_ml_lib.config.python_config import run_experiment
from fed_ml_lib.models.modular import create_model
from fed_ml_lib.data.loaders import load_datasets
from fed_ml_lib.federated.client import FlowerClient
from fed_ml_lib.federated.utils import get_parameters2, set_parameters
from fed_ml_lib.core.visualization import save_matrix, save_roc
from fed_ml_lib.core.testing import test

def client_fn(cid: str):
    """Create a Flower client for federated learning."""
    # Configuration for each client
    config = run_experiment(
        name="dna_federated",
        dataset="DNA",
        model="mlp",
        epochs=5,  # Local epochs per round
        learning_rate=0.001,
        batch_size=32,
        hidden_dims=[64, 32, 16, 8],
        dropout_rate=0.0,
        gpu=torch.cuda.is_available()
    )
    
    device = torch.device('cuda' if config['gpu'] else 'cpu')
    
    # Load federated datasets - data is automatically split across clients
    trainloaders, valloaders, testloader = load_datasets(
        num_clients=3,  # Split data among 3 clients
        batch_size=config['batch_size'],
        resize=None,
        seed=42,
        num_workers=0,
        splitter=10,
        dataset='DNA',
        data_path="data/"
    )
    
    # Get dataset properties for model creation
    sample_input, _ = next(iter(testloader))
    input_shape = (sample_input.shape[1],)
    
    all_labels = set()
    for _, labels in testloader:
        all_labels.update(labels.tolist())
    num_classes = len(all_labels)
    
    # Create client model using library's modular system
    model = create_model(
        base_architecture="mlp",
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_dims=config['hidden_dims'],
        dropout_rate=config['dropout_rate'],
        domain='sequence'
    ).to(device)
    
    class_names = [str(i) for i in range(num_classes)]
    
    # Return configured Flower client
    return FlowerClient(
        cid=cid,
        net=model,
        trainloader=trainloaders[int(cid)],  # Client-specific training data
        valloader=valloaders[int(cid)],      # Client-specific validation data
        device=device,
        batch_size=config['batch_size'],
        save_results=f"results/federated_dna_example/client_{cid}/",
        matrix_path="confusion_matrix.png",
        roc_path="roc.png",
        yaml_path=None,
        classes=class_names
    )

def evaluate_fn(server_round, parameters, config):
    """Server-side evaluation function called after each round."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test dataset for server evaluation
    _, _, testloader = load_datasets(
        num_clients=1,
        batch_size=32,
        resize=None,
        seed=42,
        num_workers=0,
        splitter=10,
        dataset='DNA',
        data_path="data/"
    )
    
    # Get dataset properties
    sample_input, _ = next(iter(testloader))
    input_shape = (sample_input.shape[1],)
    
    all_labels = set()
    for _, labels in testloader:
        all_labels.update(labels.tolist())
    num_classes = len(all_labels)
    
    # Create model for evaluation
    model = create_model(
        base_architecture="mlp",
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_dims=[64, 32, 16, 8],
        dropout_rate=0.0,
        domain='sequence'
    ).to(device)
    
    # Load aggregated parameters from federated training
    set_parameters(model, parameters)
    
    # Evaluate global model
    loss, accuracy, y_pred, y_true, y_proba = test(
        model=model,
        dataloader=testloader,
        loss_fn=torch.nn.CrossEntropyLoss(),
        device=device
    )
    
    # Save server evaluation results
    os.makedirs("results/federated_dna_example/server/", exist_ok=True)
    class_names = [str(i) for i in range(num_classes)]
    save_matrix(y_true, y_pred, "results/federated_dna_example/server/confusion_matrix.png", class_names)
    save_roc(y_true, y_proba, "results/federated_dna_example/server/roc.png", num_classes)
    
    print(f"Server evaluation - Round {server_round}: Loss {loss:.4f}, Accuracy {accuracy:.2f}%")
    return loss, {"accuracy": accuracy}

def main():
    """Main federated learning orchestration."""
    # Global configuration
    config = run_experiment(
        name="dna_federated",
        dataset="DNA",
        model="mlp",
        epochs=5,
        learning_rate=0.001,
        batch_size=32,
        hidden_dims=[64, 32, 16, 8],
        dropout_rate=0.0,
        gpu=torch.cuda.is_available()
    )
    
    device = torch.device('cuda' if config['gpu'] else 'cpu')
    
    # Get dataset info for initial model
    _, _, testloader = load_datasets(
        num_clients=1,
        batch_size=32,
        resize=None,
        seed=42,
        num_workers=0,
        splitter=10,
        dataset='DNA',
        data_path="data/"
    )
    
    sample_input, _ = next(iter(testloader))
    input_shape = (sample_input.shape[1],)
    
    all_labels = set()
    for _, labels in testloader:
        all_labels.update(labels.tolist())
    num_classes = len(all_labels)
    
    # Create initial global model for parameter initialization
    initial_model = create_model(
        base_architecture="mlp",
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_dims=[64, 32, 16, 8],
        dropout_rate=0.0,
        domain='sequence'
    ).to(device)
    
    # Configure federated learning strategy
    strategy = FedAvg(
        fraction_fit=1.0,           # Use all available clients for training
        fraction_evaluate=1.0,      # Use all available clients for evaluation
        min_fit_clients=3,          # Minimum clients needed for training round
        min_evaluate_clients=3,     # Minimum clients needed for evaluation round
        min_available_clients=3,    # Minimum clients that must be available
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters2(initial_model)),
        evaluate_fn=evaluate_fn,    # Server-side evaluation function
        on_fit_config_fn=lambda round: {  # Configuration sent to clients each round
            "learning_rate": str(config['learning_rate']),
            "batch_size": str(config['batch_size']),
            "server_round": round,
            "local_epochs": config['epochs']
        }
    )
    
    # Configure client resources
    client_resources = {"num_cpus": 1}
    if device.type == "cuda":
        client_resources["num_gpus"] = 0.33  # Share GPU among clients
    
    print(f"Starting federated learning with {device}")
    start_time = time.time()
    
    # Start federated learning simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,                                    # Function to create clients
        num_clients=3,                                          # Total number of clients
        config=fl.server.ServerConfig(num_rounds=3),           # 3 rounds of federated learning
        strategy=strategy,                                      # FedAvg aggregation strategy
        client_resources=client_resources                       # Resource allocation per client
    )
    
    print(f"Federated training completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 