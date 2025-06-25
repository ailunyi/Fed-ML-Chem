"""
Quantum Federated DNA Training Example using Fed-ML-Lib and PennyLane
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
from fed_ml_lib.core.utils import choice_device

def client_fn(cid: str):
    """Create a Flower client with quantum neural network"""
    config = run_experiment(
        name="quantum_dna_federated",
        dataset="DNA",
        model="quantum_mlp",
        epochs=2,
        learning_rate=0.001,
        batch_size=32,
        gpu=torch.cuda.is_available()
    )
    
    device = torch.device(choice_device("gpu" if config['gpu'] else "cpu"))
    
    # Load federated datasets
    trainloaders, valloaders, testloader = load_datasets(
        num_clients=4,
        batch_size=config['batch_size'],
        resize=None,
        seed=42,
        num_workers=0,
        splitter=20,
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
    
    # Create quantum neural network using library's modular system
    model = create_model(
        base_architecture="mlp",
        input_shape=input_shape,
        num_classes=num_classes,
        use_quantum=True,                    # Enable quantum processing
        hidden_dims=[1024, 512, 256, 128],   # Classical hidden layers
        dropout_rate=0.0,
        domain='sequence',
        n_qubits=7,                          # Quantum circuit parameters
        n_layers=7,
        quantum_circuit="basic_entangler",
        quantum_layers=['classifier']        # Apply quantum to classifier
    ).to(device)
    
    class_names = [str(i) for i in range(num_classes)]
    
    return FlowerClient(
        cid=cid,
        net=model,
        trainloader=trainloaders[int(cid)],
        valloader=valloaders[int(cid)],
        device=device,
        batch_size=config['batch_size'],
        save_results=f"results/quantum_federated_dna/client_{cid}/",
        matrix_path="confusion_matrix.png",
        roc_path="roc.png",
        yaml_path=None,
        classes=class_names
    )

def evaluate_fn(server_round, parameters, config):
    """Server-side evaluation with quantum model"""
    device = torch.device(choice_device("gpu" if torch.cuda.is_available() else "cpu"))
    
    # Load test dataset
    _, _, testloader = load_datasets(
        num_clients=1,
        batch_size=32,
        resize=None,
        seed=42,
        num_workers=0,
        splitter=20,
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
    
    # Create quantum model for evaluation using library
    model = create_model(
        base_architecture="mlp",
        input_shape=input_shape,
        num_classes=num_classes,
        use_quantum=True,
        hidden_dims=[1024, 512, 256, 128],
        dropout_rate=0.0,
        domain='sequence',
        n_qubits=7,
        n_layers=7,
        quantum_circuit="basic_entangler",
        quantum_layers=['classifier']
    ).to(device)
    
    # Load aggregated parameters
    set_parameters(model, parameters)
    
    # Evaluate quantum model
    loss, accuracy, y_pred, y_true, y_proba = test(
        model=model,
        dataloader=testloader,
        loss_fn=torch.nn.CrossEntropyLoss(),
        device=device
    )
    
    # Save evaluation results
    os.makedirs("results/quantum_federated_dna/server/", exist_ok=True)
    class_names = [str(i) for i in range(num_classes)]
    save_matrix(y_true, y_pred, "results/quantum_federated_dna/server/confusion_matrix.png", class_names)
    save_roc(y_true, y_proba, "results/quantum_federated_dna/server/roc.png", num_classes)
    
    print(f"Quantum Server evaluation - Round {server_round}: Loss {loss:.4f}, Accuracy {accuracy:.2f}%")
    return loss, {"accuracy": accuracy}

def main():
    """Main quantum federated learning orchestration"""
    config = run_experiment(
        name="quantum_dna_federated",
        dataset="DNA",
        model="quantum_mlp",
        epochs=2,
        learning_rate=0.001,
        batch_size=32,
        gpu=torch.cuda.is_available()
    )
    
    device = torch.device(choice_device("gpu" if config['gpu'] else "cpu"))
    
    # Get dataset info for initial quantum model
    _, _, testloader = load_datasets(
        num_clients=1,
        batch_size=32,
        resize=None,
        seed=42,
        num_workers=0,
        splitter=20,
        dataset='DNA',
        data_path="data/"
    )
    
    sample_input, _ = next(iter(testloader))
    input_shape = (sample_input.shape[1],)
    
    all_labels = set()
    for _, labels in testloader:
        all_labels.update(labels.tolist())
    num_classes = len(all_labels)
    
    # Create initial quantum model using library's modular system
    initial_model = create_model(
        base_architecture="mlp",
        input_shape=input_shape,
        num_classes=num_classes,
        use_quantum=True,
        hidden_dims=[1024, 512, 256, 128],
        dropout_rate=0.0,
        domain='sequence',
        n_qubits=7,
        n_layers=7,
        quantum_circuit="basic_entangler",
        quantum_layers=['classifier']
    ).to(device)
    
    # Configure federated learning strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
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
    
    print("="*60)
    print("QUANTUM FEDERATED LEARNING - DNA CLASSIFICATION")
    print("="*60)
    print(f"Device: {device}")
    print(f"Architecture: MLP with Quantum Enhancement")
    print(f"Quantum circuit: {7} qubits, {7} layers")
    print(f"Clients: 4")
    print(f"Dataset: DNA sequences")
    
    start_time = time.time()
    
    # Start quantum federated learning simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=4,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
        client_resources=client_resources
    )
    
    print(f"Quantum federated training completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 