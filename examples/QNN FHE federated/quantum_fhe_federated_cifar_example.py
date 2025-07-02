"""
Quantum FHE Federated Learning Example for CIFAR Dataset
Combines Fed-ML-Lib's built-in Quantum and FHE implementations
"""
import torch
import flwr as fl
from flwr.server.strategy import FedAvg
import os
import time

from fed_ml_lib.data.loaders import load_datasets
from fed_ml_lib.models.modular import create_model
from fed_ml_lib.federated.client import FHEFlowerClient
from fed_ml_lib.federated.utils import get_parameters2, set_parameters
from fed_ml_lib.core.utils import choice_device
from fed_ml_lib.core.visualization import save_all_results
from fed_ml_lib.core.testing import test

def client_fn(cid: str):
    """Create a Quantum FHE Flower client using the library implementation"""
    batch_size = 32
    device = torch.device(choice_device('gpu'))
    
    # Load federated datasets
    trainloaders, valloaders, testloader = load_datasets(
        num_clients=4,
        batch_size=batch_size,
        resize=32,  # CIFAR native resolution
        seed=42,
        num_workers=0,
        splitter=20,
        dataset='cifar',
        data_path="data/"
    )
    
    # Create CIFAR model with both Quantum and FHE support
    model = create_model(
        base_architecture="cnn",
        input_shape=(3, 32, 32),
        num_classes=10,
        conv_channels=[32, 64],
        hidden_dims=[128],
        dropout_rate=0.0,
        domain="vision",
        use_quantum=True,
        n_qubits=4,
        n_layers=2,
        quantum_circuit="basic_entangler",
        quantum_layers=["features"],
        use_fhe=True,
        fhe_scheme="CKKS",
        fhe_layers=["classifier"]
    ).to(device)
    
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Use the library's FHEFlowerClient (handles both quantum and FHE)
    return FHEFlowerClient(
        cid=cid,
        net=model,
        trainloader=trainloaders[int(cid)],
        valloader=valloaders[int(cid)],
        device=device,
        batch_size=batch_size,
        save_results=f"results/quantum_fhe_federated_cifar/client_{cid}/",
        matrix_path="confusion_matrix.png",
        roc_path="roc.png",
        yaml_path=None,
        classes=classes,
        fhe_scheme="CKKS",
        fhe_layers=["classifier"],
        context_path=f"quantum_fhe_context_client_{cid}.pkl"
    )

def evaluate_fn(server_round, parameters, config):
    """Server-side evaluation function"""
    device = torch.device(choice_device('gpu'))
    
    # Load test dataset
    _, _, testloader = load_datasets(
        num_clients=1,
        batch_size=32,
        resize=32,
        seed=42,
        num_workers=0,
        splitter=20,
        dataset='cifar',
        data_path="data/"
    )
    
    # Create model with same quantum and FHE configuration
    model = create_model(
        base_architecture="cnn",
        input_shape=(3, 32, 32),
        num_classes=10,
        conv_channels=[32, 64],
        hidden_dims=[128],
        dropout_rate=0.0,
        domain="vision",
        use_quantum=True,
        n_qubits=4,
        n_layers=2,
        quantum_circuit="basic_entangler",
        quantum_layers=["features"],
        use_fhe=True,
        fhe_scheme="CKKS",
        fhe_layers=["classifier"]
    ).to(device)
    
    # Load parameters
    set_parameters(model, parameters)
    
    # Evaluate
    loss, accuracy, y_pred, y_true, y_proba = test(
        model=model,
        dataloader=testloader,
        loss_fn=torch.nn.CrossEntropyLoss(),
        device=device
    )
    
    # Save results
    os.makedirs("results/quantum_fhe_federated_cifar/server/", exist_ok=True)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    save_all_results(
        train_true=y_true,
        train_pred=y_pred,
        train_proba=y_proba,
        test_true=y_true,
        test_pred=y_pred,
        test_proba=y_proba,
        training_history={},
        classes=classes,
        results_path="results/quantum_fhe_federated_cifar/server",
        config={},
        file_suffix=""
    )
    # ROC curve saved by save_all_results above)
    
    print(f"Server evaluation - Round {server_round}: Loss {loss:.4f}, Accuracy {accuracy:.2f}%")
    return loss, {"accuracy": accuracy}

def main():
    """Main federated learning orchestration"""
    device = torch.device(choice_device('gpu'))
    
    # Create initial model with both Quantum and FHE support
    initial_model = create_model(
        base_architecture="cnn",
        input_shape=(3, 32, 32),
        num_classes=10,
        conv_channels=[32, 64],
        hidden_dims=[128],
        dropout_rate=0.0,
        domain="vision",
        use_quantum=True,
        n_qubits=4,
        n_layers=2,
        quantum_circuit="basic_entangler",
        quantum_layers=["features"],
        use_fhe=True,
        fhe_scheme="CKKS",
        fhe_layers=["classifier"]
    ).to(device)
    
    print("=" * 70)
    print("QUANTUM FHE FEDERATED LEARNING - CIFAR DATASET")
    print("Using Fed-ML-Lib's Built-in Quantum + FHE Implementation")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Clients: 4")
    print(f"Rounds: 20")
    print(f"Local Epochs: 10")
    print(f"Dataset: CIFAR-10")
    print(f"Architecture: Quantum CNN with FHE")
    print(f"Quantum: 4 qubits, 2 layers, basic_entangler")
    print(f"Quantum Layers: features")
    print(f"FHE Scheme: CKKS")
    print(f"FHE Layers: classifier")
    print("=" * 70)
    
    start_time = time.time()
    
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
            "learning_rate": "0.001",
            "batch_size": "32",
            "server_round": round,
            "local_epochs": 10  # 10 local epochs as requested
        }
    )
    
    # Set up client resources (quantum + FHE requires more resources)
    client_resources = {"num_cpus": 2}
    if device.type == "cuda":
        client_resources["num_gpus"] = 1
    
    # Create results directory
    os.makedirs("results/quantum_fhe_federated_cifar", exist_ok=True)
    
    # Start federated learning
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=4,
        config=fl.server.ServerConfig(num_rounds=20),  # 20 rounds as requested
        strategy=strategy,
        client_resources=client_resources
    )
    
    print(f"Quantum FHE federated training completed in {time.time() - start_time:.2f} seconds")
    print("\nExperiment Summary:")
    print(f"- Architecture: Quantum CNN + FHE")
    print(f"- Quantum Processing: 4 qubits on feature layers")
    print(f"- FHE Encryption: CKKS on classifier layer")
    print(f"- Federated Learning: 4 clients, 20 rounds, 10 local epochs")
    print(f"- Results saved to: results/quantum_fhe_federated_cifar/")

if __name__ == "__main__":
    main() 