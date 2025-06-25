"""
Quantum Federated HIV Training Example using Fed-ML-Lib and PennyLane
Based on the legacy Standard_FedQNN_HIV notebook but using modern library architecture
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
from fed_ml_lib.core.testing import test_graph
from fed_ml_lib.core.utils import choice_device

def client_fn(cid: str):
    """Create a Flower client with quantum graph neural network for HIV classification"""
    config = run_experiment(
        name="quantum_hiv_federated",
        dataset="hiv",
        model="quantum_gnn",
        epochs=10,
        learning_rate=0.001,
        batch_size=64,
        gpu=torch.cuda.is_available()
    )
    
    device = torch.device(choice_device("gpu" if config['gpu'] else "cpu"))
    
    # Load federated datasets for HIV
    trainloaders, valloaders, testloader = load_datasets(
        num_clients=10,  # Following legacy notebook setup
        batch_size=config['batch_size'],
        resize=None,
        seed=42,
        num_workers=0,
        splitter=10,  # Following legacy notebook
        dataset='hiv',
        data_path="data/"
    )
    
    # Get dataset properties from graph data
    sample_batch = next(iter(testloader))
    if hasattr(sample_batch, 'x'):
        # Graph data format
        input_dim = sample_batch.x.shape[1]  # Node feature dimension
        num_classes = 2  # Binary classification: confirmed inactive vs confirmed active
    else:
        # Fallback for non-graph data
        input_dim = 9  # Default molecular feature dimension
        num_classes = 2
    
    # Create quantum graph neural network using library's modular system
    model = create_model(
        base_architecture="gcn",             # Use "gcn" not "gnn"
        input_shape=(input_dim,),
        num_classes=num_classes,
        use_quantum=True,                    # Enable quantum processing
        hidden_dims=[64, 64, 64],            # GNN hidden dimensions (embedding_size=64)
        dropout_rate=0.0,
        domain='graph',
        n_qubits=2,                          # Following legacy: 2 qubits
        n_layers=2,                          # Following legacy: 2 layers
        quantum_circuit="basic_entangler",   # Following legacy circuit
        quantum_layers=['classifier']        # Apply quantum to final classifier
    ).to(device)
    
    class_names = ['confirmed inactive (CI)', 'confirmed active (CA)/confirmed moderately active (CM)']
    
    return FlowerClient(
        cid=cid,
        net=model,
        trainloader=trainloaders[int(cid)],
        valloader=valloaders[int(cid)],
        device=device,
        batch_size=config['batch_size'],
        save_results=f"results/quantum_federated_hiv/client_{cid}/",
        matrix_path="confusion_matrix.png",
        roc_path="roc.png",
        yaml_path=None,
        classes=class_names
    )

def evaluate_fn(server_round, parameters, config):
    """Server-side evaluation with quantum graph model"""
    device = torch.device(choice_device("gpu" if torch.cuda.is_available() else "cpu"))
    
    # Load test dataset
    _, _, testloader = load_datasets(
        num_clients=1,
        batch_size=64,
        resize=None,
        seed=42,
        num_workers=0,
        splitter=10,
        dataset='hiv',
        data_path="data/"
    )
    
    # Get dataset properties
    sample_batch = next(iter(testloader))
    if hasattr(sample_batch, 'x'):
        input_dim = sample_batch.x.shape[1]
        num_classes = 2
    else:
        input_dim = 9
        num_classes = 2
    
    # Create quantum graph model for evaluation
    model = create_model(
        base_architecture="gcn",             # Use "gcn" not "gnn"
        input_shape=(input_dim,),
        num_classes=num_classes,
        use_quantum=True,
        hidden_dims=[64, 64, 64],
        dropout_rate=0.0,
        domain='graph',
        n_qubits=2,
        n_layers=2,
        quantum_circuit="basic_entangler",
        quantum_layers=['classifier']
    ).to(device)
    
    # Load aggregated parameters
    set_parameters(model, parameters)
    
    # Custom RMSE loss as in legacy notebook
    class RMSELoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mse = torch.nn.MSELoss()
        
        def forward(self, yhat, y):
            return torch.sqrt(self.mse(yhat, y))
    
    # Evaluate quantum graph model
    loss, accuracy, y_pred, y_true, y_proba = test_graph(
        model=model,
        dataloader=testloader,
        loss_fn=RMSELoss(),
        device=device
    )
    
    # Save evaluation results
    os.makedirs("results/quantum_federated_hiv/server/", exist_ok=True)
    class_names = ['confirmed inactive (CI)', 'confirmed active (CA)/confirmed moderately active (CM)']
    save_matrix(y_true, y_pred, "results/quantum_federated_hiv/server/confusion_matrix.png", class_names)
    save_roc(y_true, y_proba, "results/quantum_federated_hiv/server/roc.png", num_classes)
    
    print(f"Quantum Server evaluation - Round {server_round}: Loss {loss:.4f}, Accuracy {accuracy:.2f}%")
    return loss, {"accuracy": accuracy}

def main():
    """Main quantum federated learning orchestration for HIV classification"""
    config = run_experiment(
        name="quantum_hiv_federated",
        dataset="hiv",
        model="quantum_gnn",
        epochs=2,
        learning_rate=0.001,
        batch_size=64,
        gpu=torch.cuda.is_available()
    )
    
    device = torch.device(choice_device("gpu" if config['gpu'] else "cpu"))
    
    # Get dataset info for initial quantum model
    _, _, testloader = load_datasets(
        num_clients=1,
        batch_size=64,
        resize=None,
        seed=42,
        num_workers=0,
        splitter=10,
        dataset='hiv',
        data_path="data/"
    )
    
    # Get dataset properties
    sample_batch = next(iter(testloader))
    if hasattr(sample_batch, 'x'):
        input_dim = sample_batch.x.shape[1]
        num_classes = 2
    else:
        input_dim = 9
        num_classes = 2
    
    # Create initial quantum graph model using library's modular system
    initial_model = create_model(
        base_architecture="gcn",             # Use "gcn" not "gnn"
        input_shape=(input_dim,),
        num_classes=num_classes,
        use_quantum=True,
        hidden_dims=[64, 64, 64],  # embedding_size=64 as in legacy
        dropout_rate=0.0,
        domain='graph',
        n_qubits=2,                # Following legacy: 2 qubits
        n_layers=2,                # Following legacy: 2 layers
        quantum_circuit="basic_entangler",
        quantum_layers=['classifier']
    ).to(device)
    
    # Configure federated learning strategy (following legacy parameters)
    strategy = FedAvg(
        fraction_fit=1.0,          # frac_fit=1.0
        fraction_evaluate=0.5,     # frac_eval=0.5
        min_fit_clients=10,        # min_fit_clients=10
        min_evaluate_clients=5,    # min_eval_clients=10//2
        min_available_clients=10,  # min_avail_clients=10
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters2(initial_model)),
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=lambda round: {
            "learning_rate": str(config['learning_rate']),  # lr=1e-3
            "batch_size": str(config['batch_size']),
            "server_round": round,
            "local_epochs": config['epochs']  # max_epochs=10 -> epochs=2 for quick demo
        }
    )
    
    # Configure client resources
    client_resources = {"num_cpus": 1}
    if device.type == "cuda":
        client_resources["num_gpus"] = 0.1  # Share GPU among clients
    
    print("="*60)
    print("QUANTUM FEDERATED LEARNING - HIV CLASSIFICATION")
    print("Following Standard_FedQNN_HIV notebook methodology")
    print("="*60)
    print(f"Device: {device}")
    print(f"Architecture: GNN + Quantum Neural Network")
    print(f"Quantum circuit: {2} qubits, {2} layers (AngleEmbedding + BasicEntanglerLayers)")
    print(f"Clients: 10")
    print(f"Dataset: HIV molecular graphs")
    print(f"Classes: Binary (CI vs CA/CM)")
    print(f"Embedding size: 64")
    
    start_time = time.time()
    
    # Start quantum federated learning simulation (following legacy: 20 rounds)
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=10,
        config=fl.server.ServerConfig(num_rounds=10),  
        strategy=strategy,
        client_resources=client_resources
    )
    
    print(f"Quantum federated HIV training completed in {time.time() - start_time:.2f} seconds")
    print("\nResults saved to results/quantum_federated_hiv/")
    print("- Server evaluation: confusion_matrix.png, roc.png")
    print("- Client results: individual client directories")

if __name__ == "__main__":
    main() 