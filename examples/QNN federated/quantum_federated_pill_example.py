"""
Quantum Federated PILL Training Example using Fed-ML-Lib
Matching the exact legacy architecture with VGG16 + Quantum Layer
"""
import torch
import flwr as fl
from flwr.server.strategy import FedAvg
import os
import time
import torch.nn as nn
import torchvision.models as models
import pennylane as qml

from fed_ml_lib.config.python_config import run_experiment
from fed_ml_lib.data.loaders import load_datasets
from fed_ml_lib.federated.client import FlowerClient
from fed_ml_lib.federated.utils import get_parameters2, set_parameters
from fed_ml_lib.core.visualization import save_all_results
from fed_ml_lib.core.testing import test

# Quantum circuit parameters (matching legacy)
n_qubits = 2
n_layers = 2
weight_shapes = {"weights": (n_layers, n_qubits)}
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface='torch')
def quantum_net(inputs, weights):
    """Quantum circuit exactly matching the legacy implementation"""
    qml.AngleEmbedding(inputs, wires=range(n_qubits)) 
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class Net(nn.Module):
    """
    Exact replica of legacy quantum CNN model with VGG16 backbone
    A CNN model with VGG16 feature extractor and quantum neural network
    """
    def __init__(self, num_classes=2):
        super(Net, self).__init__()
        # Use exact same VGG16 feature extractor as legacy
        self.feature_extractor = models.vgg16(weights='IMAGENET1K_V1').features[:-1]
        
        # Exact same classification head as legacy
        self.classification_head = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=(224 // 2 ** 5, 224 // 2 ** 5)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=num_classes),  # VGG16 last conv has 512 channels
        )
        
        # Direct quantum layer integration (exact match to legacy)
        self.qnn = qml.qnn.TorchLayer(quantum_net, weight_shapes)
        
        # Freeze VGG16 parameters exactly like legacy (first 23 layers)
        for param in self.feature_extractor[:23].parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass matching legacy exactly
        """
        feature_maps = self.feature_extractor(x)
        scores = self.classification_head(feature_maps)
        scores = self.qnn(scores)  # Direct quantum processing, no residual connection
        return scores

def client_fn(cid: str):
    """Create a Flower client with exact legacy quantum neural network for PILL classification"""
    config = run_experiment(
        name="quantum_pill_federated",
        dataset="PILL",
        model="quantum_cnn",
        epochs=2,
        learning_rate=0.001,
        batch_size=32,
        gpu=torch.cuda.is_available()
    )
    
    device = torch.device('cuda' if config['gpu'] else 'cpu')
    
    # Load federated datasets for PILL
    trainloaders, valloaders, testloader = load_datasets(
        num_clients=4,
        batch_size=config['batch_size'],
        resize=224,  # PILL is an image dataset, needs proper resize
        seed=42,
        num_workers=0,
        splitter=20,
        dataset='PILL',
        data_path="data/"
    )
    
    # PILL is an image dataset with known dimensions from config
    input_shape = (3, 224, 224)  # RGB images, 224x224
    num_classes = 2  # PILL is binary classification: bad/good pills
    
    print(f"Client {cid}: Input shape: {input_shape}, Classes: {num_classes}")
    
    # Create exact legacy model architecture
    model = Net(num_classes=num_classes).to(device)
    
    class_names = ['bad', 'good']  # Actual PILL class names
    
    return FlowerClient(
        cid=cid,
        net=model,
        trainloader=trainloaders[int(cid)],
        valloader=valloaders[int(cid)],
        device=device,
        batch_size=config['batch_size'],
        save_results=f"results/quantum_federated_pill/client_{cid}/",
        matrix_path="confusion_matrix.png",
        roc_path="roc.png",
        yaml_path=None,
        classes=class_names
    )

def evaluate_fn(server_round, parameters, config):
    """Server-side evaluation with exact legacy quantum model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test dataset
    _, _, testloader = load_datasets(
        num_clients=1,
        batch_size=32,
        resize=224,  # PILL is an image dataset
        seed=42,
        num_workers=0,
        splitter=20,
        dataset='PILL',
        data_path="data/"
    )
    
    # PILL is an image dataset with known dimensions
    input_shape = (3, 224, 224)
    num_classes = 2  # PILL is binary classification: bad/good pills
    
    print(f"Server evaluation: Input shape: {input_shape}, Classes: {num_classes}")
    
    # Create exact legacy model architecture
    model = Net(num_classes=num_classes).to(device)
    
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
    os.makedirs("results/quantum_federated_pill/server/", exist_ok=True)
    class_names = ['bad', 'good']  # Actual PILL class names
    save_all_results(
        train_true=y_true,
        train_pred=y_pred,
        train_proba=y_proba,
        test_true=y_true,
        test_pred=y_pred,
        test_proba=y_proba,
        training_history={},
        classes=class_names,
        results_path="results/quantum_federated_pill/server",
        config={},
        file_suffix=""
    )
    
    print(f"Quantum Server evaluation - Round {server_round}: Loss {loss:.4f}, Accuracy {accuracy:.2f}%")
    return loss, {"accuracy": accuracy}

def main():
    """Main quantum federated learning orchestration for PILL classification using legacy architecture"""
    config = run_experiment(
        name="quantum_pill_federated",
        dataset="PILL",
        model="quantum_cnn",
        epochs=2,
        learning_rate=0.001,
        batch_size=32,
        gpu=torch.cuda.is_available()
    )
    
    device = torch.device('cuda' if config['gpu'] else 'cpu')
    
    # PILL dataset has known configuration
    input_shape = (3, 224, 224)
    num_classes = 2  # PILL is binary classification: bad/good pills
    
    print(f"Main: Input shape: {input_shape}, Classes: {num_classes}")
    
    # Create initial legacy quantum model
    initial_model = Net(num_classes=num_classes).to(device)
    
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
    
    # Configure client resources - VGG16 needs more memory
    client_resources = {"num_cpus": 1}
    if device.type == "cuda":
        client_resources["num_gpus"] = 0.3  # Increased for VGG16 pretrained model
    
    print("="*60)
    print("QUANTUM FEDERATED LEARNING - PILL CLASSIFICATION (LEGACY ARCHITECTURE)")
    print("="*60)
    print(f"Device: {device}")
    print(f"Architecture: VGG16 + Quantum Layer (Legacy Match)")
    print(f"Input shape: {input_shape}")
    print(f"Quantum circuit: {n_qubits} qubits, {n_layers} layers")
    print(f"Clients: 4")
    print(f"Dataset: PILL (Images) - Binary Classification")
    
    start_time = time.time()
    
    # Start quantum federated learning simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=4,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
        client_resources=client_resources
    )
    
    print(f"Quantum federated PILL training completed in {time.time() - start_time:.2f} seconds")
    print("\nResults saved to results/quantum_federated_pill/")

if __name__ == "__main__":
    main() 