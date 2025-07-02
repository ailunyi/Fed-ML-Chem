"""
Quantum FHE Federated Learning Example for MRI Brain Tumor Dataset
Uses Fed-ML-Lib's built-in Quantum and FHE implementations
Based on legacy FHE_FedQNN_MRI.py architecture
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
    
    # Load federated MRI datasets
    trainloaders, valloaders, testloader = load_datasets(
        num_clients=4,
        batch_size=batch_size,
        resize=224,  # MRI images resized to 224x224 as in legacy code
        seed=42,
        num_workers=0,
        splitter=10,
        dataset='MRI',
        data_path="data/"
    )
    
    # Create MRI model with Quantum and FHE support (based on legacy architecture)
    # Legacy: Conv(3,16) -> Conv(16,32) -> Linear(32*56*56, 128) -> Linear(128, n_qubits) -> QNN -> Linear(n_qubits, 4)
    model = create_model(
        base_architecture="cnn",
        input_shape=(3, 224, 224),  # MRI input shape
        num_classes=4,  # MRI classes: glioma, meningioma, notumor, pituitary
        conv_channels=[16, 32],  # Match legacy architecture
        hidden_dims=[128],  # Match legacy architecture
        dropout_rate=0.0,
        domain="vision",
        use_quantum=True,
        n_qubits=4,  # Match legacy: 4 qubits
        n_layers=6,  # Match legacy: 6 quantum layers
        quantum_circuit="basic_entangler",  # Match legacy: BasicEntanglerLayers
        quantum_layers=["classifier"],  # Apply quantum to classifier as in legacy
        use_fhe=True,
        fhe_scheme="CKKS",  # Match legacy TenSEAL CKKS scheme
        fhe_layers=["classifier"]  # Apply FHE to classifier
    ).to(device)
    
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    # Use the library's FHEFlowerClient (handles both quantum and FHE)
    return FHEFlowerClient(
        cid=cid,
        net=model,
        trainloader=trainloaders[int(cid)],
        valloader=valloaders[int(cid)],
        device=device,
        batch_size=batch_size,
        save_results=f"results/quantum_fhe_federated_mri/client_{cid}/",
        matrix_path="confusion_matrix.png",
        roc_path="roc.png",
        yaml_path=None,
        classes=classes,
        fhe_scheme="CKKS",
        fhe_layers=["classifier"],
        context_path=f"quantum_fhe_mri_context_client_{cid}.pkl"
    )

def evaluate_fn(server_round, parameters, config):
    """Server-side evaluation function"""
    device = torch.device(choice_device('gpu'))
    
    # Load test dataset
    _, _, testloader = load_datasets(
        num_clients=1,
        batch_size=32,
        resize=224,  # MRI resolution
        seed=42,
        num_workers=0,
        splitter=10,
        dataset='MRI',
        data_path="data/"
    )
    
    # Create model with same quantum and FHE configuration
    model = create_model(
        base_architecture="cnn",
        input_shape=(3, 224, 224),
        num_classes=4,
        conv_channels=[16, 32],
        hidden_dims=[128],
        dropout_rate=0.0,
        domain="vision",
        use_quantum=True,
        n_qubits=4,
        n_layers=6,
        quantum_circuit="basic_entangler",
        quantum_layers=["classifier"],
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
    os.makedirs("results/quantum_fhe_federated_mri/server/", exist_ok=True)
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    save_all_results(
        train_true=y_true,
        train_pred=y_pred,
        train_proba=y_proba,
        test_true=y_true,
        test_pred=y_pred,
        test_proba=y_proba,
        training_history={},
        classes=classes,
        results_path="results/quantum_fhe_federated_mri/server",
        config={},
        file_suffix=""
    )
    # ROC curve saved by save_all_results above)
    
    print(f"Server evaluation - Round {server_round}: Loss {loss:.4f}, Accuracy {accuracy:.2f}%")
    return loss, {"accuracy": accuracy}

def main():
    """Main federated learning orchestration"""
    device = torch.device(choice_device('gpu'))
    
    # Create initial model with Quantum and FHE support
    initial_model = create_model(
        base_architecture="cnn",
        input_shape=(3, 224, 224),
        num_classes=4,
        conv_channels=[16, 32],
        hidden_dims=[128],
        dropout_rate=0.0,
        domain="vision",
        use_quantum=True,
        n_qubits=4,
        n_layers=6,
        quantum_circuit="basic_entangler",
        quantum_layers=["classifier"],
        use_fhe=True,
        fhe_scheme="CKKS",
        fhe_layers=["classifier"]
    ).to(device)
    
    print("=" * 80)
    print("QUANTUM FHE FEDERATED LEARNING - MRI BRAIN TUMOR DATASET")
    print("Using Fed-ML-Lib's Built-in Quantum + FHE Implementation")
    print("Based on Legacy FHE_FedQNN_MRI.py Architecture")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Clients: 4")
    print(f"Rounds: 20")
    print(f"Local Epochs: 10")
    print(f"Dataset: MRI Brain Tumor (4 classes)")
    print(f"Classes: glioma, meningioma, notumor, pituitary")
    print(f"Input Resolution: 224x224 (as in legacy)")
    print(f"Architecture: Quantum CNN with FHE")
    print(f"CNN Layers: Conv(3→16) → Conv(16→32) → Linear(128)")
    print(f"Quantum: 4 qubits, 6 layers, basic_entangler")
    print(f"Quantum Layers: classifier")
    print(f"FHE Scheme: CKKS (TenSEAL compatible)")
    print(f"FHE Layers: classifier")
    print("=" * 80)
    
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
            "learning_rate": "0.001",  # Match legacy lr=1e-3
            "batch_size": "32",
            "server_round": round,
            "local_epochs": 10  # Match legacy max_epochs=10
        }
    )
    
    # Set up client resources (quantum + FHE requires more resources)
    client_resources = {"num_cpus": 2}
    if device.type == "cuda":
        client_resources["num_gpus"] = 1
    
    # Create results directory
    os.makedirs("results/quantum_fhe_federated_mri", exist_ok=True)
    
    # Start federated learning
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=4,
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
        client_resources=client_resources
    )
    
    print(f"Quantum FHE MRI federated training completed in {time.time() - start_time:.2f} seconds")
    print("\nExperiment Summary:")
    print(f"- Dataset: MRI Brain Tumor Classification")
    print(f"- Architecture: Quantum CNN + FHE (based on legacy)")
    print(f"- Input: 224x224 RGB images")
    print(f"- CNN: Conv(3→16) → Conv(16→32) → Linear(128)")
    print(f"- Quantum Processing: 4 qubits, 6 layers on classifier")
    print(f"- FHE Encryption: CKKS on classifier layer")
    print(f"- Federated Learning: 4 clients, 20 rounds, 10 local epochs")
    print(f"- Classes: glioma, meningioma, notumor, pituitary")
    print(f"- Results saved to: results/quantum_fhe_federated_mri/")

if __name__ == "__main__":
    main() 