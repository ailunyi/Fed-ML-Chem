"""
Federated Quantum DNA+MRI Training Example using Fed-ML-Lib
Modernized version of the legacy Standard_FedQNN_DNA+MRI notebook using the Fed-ML-Lib library
Properly using library's built-in models and client classes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import flwr as fl
from flwr.server.strategy import FedAvg
import os
import time
import pennylane as qml
import numpy as np

from fed_ml_lib.config.python_config import run_experiment
from fed_ml_lib.models.modular import create_model, ModularConfig
from fed_ml_lib.data.loaders import load_datasets
from fed_ml_lib.federated.client import MultimodalFlowerClient
from fed_ml_lib.federated.utils import get_parameters2, set_parameters
from fed_ml_lib.core.visualization import save_all_results
from fed_ml_lib.core.testing import test
from fed_ml_lib.core.training import train

class QuantumMultimodalNet(nn.Module):
    """Combined quantum-enhanced multimodal network for DNA+MRI using library components"""
    def __init__(self, num_classes_mri, num_classes_dna, dna_input_size):
        super(QuantumMultimodalNet, self).__init__()
        
        # Create MRI model using library's modular system with quantum enhancement
        self.mri_net = create_model(
            base_architecture="cnn",
            input_shape=(3, 224, 224),
            num_classes=4,  # Intermediate quantum features
            use_quantum=True,
            n_qubits=4,
            n_layers=6,
            quantum_circuit="basic_entangler",
            conv_channels=[16, 32],
            hidden_dims=[128],
            domain="vision"
        )
        
        # Create DNA model using library's modular system with quantum enhancement  
        self.dna_net = create_model(
            base_architecture="mlp",
            input_shape=(dna_input_size,),  # Use actual DNA input size
            num_classes=7,  # Intermediate quantum features
            use_quantum=True,
            n_qubits=7,
            n_layers=6,
            quantum_circuit="basic_entangler",
            hidden_dims=[512, 256, 128],
            dropout_rate=0.5,
            domain="sequence"
        )
        
        # Feature fusion with attention
        self.feature_dim = 4 + 7  # MRI qubits + DNA qubits
        self.attention = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=self.feature_dim)
        
        # Final classification layers
        self.fc1 = nn.Linear(self.feature_dim, 128)
        self.fc2_mri = nn.Linear(128, num_classes_mri)
        self.fc2_dna = nn.Linear(128, num_classes_dna)
        
    def forward(self, mri_input, dna_input):
        # Extract quantum features from each modality
        mri_features = self.mri_net(mri_input)
        dna_features = self.dna_net(dna_input)
        
        # Combine features
        combined_features = torch.cat((mri_features, dna_features), dim=1)
        combined_features = combined_features.unsqueeze(0)
        
        # Apply attention mechanism
        attn_output, _ = self.attention(combined_features, combined_features, combined_features)
        attn_output = attn_output.squeeze(0)
        
        # Final classification
        x = F.relu(self.fc1(attn_output))
        mri_output = self.fc2_mri(x)
        dna_output = self.fc2_dna(x)
        
        return mri_output, dna_output

def client_fn(cid: str):
    """Create a Flower client for quantum multimodal learning using library components"""
    config = run_experiment(
        name="quantum_dna_mri_federated",
        dataset="DNA+MRI",
        model="quantum_multimodal",
        epochs=5,
        learning_rate=0.001,
        batch_size=32,
        gpu=torch.cuda.is_available()
    )
    
    device = torch.device('cuda' if config['gpu'] else 'cpu')
    
    # Load federated datasets using library
    trainloaders, valloaders, testloader = load_datasets(
        num_clients=4,
        batch_size=config['batch_size'],
        resize=224,
        seed=42,
        num_workers=0,
        splitter=20,
        dataset='DNA+MRI',
        data_path="data/"
    )
    
    # Get actual DNA input size from the data (like in centralized example)
    _, dna_input_size = next(iter(testloader))[0][1].shape
    print(f"[Client {cid}] DNA input size: {dna_input_size}")
    
    # Get class information
    mri_classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    dna_classes = ['0', '1', '2', '3', '4', '5', '6']
    classes = (mri_classes, dna_classes)
    
    # Create quantum multimodal model using library components with correct DNA input size
    model = QuantumMultimodalNet(len(mri_classes), len(dna_classes), dna_input_size).to(device)
    
    # Use library's MultimodalFlowerClient instead of custom implementation
    return MultimodalFlowerClient(
        cid=cid,
        net=model,
        trainloader=trainloaders[int(cid)],
        valloader=valloaders[int(cid)],
        device=device,
        batch_size=config['batch_size'],
        matrix_path="confusion_matrix.png",
        roc_path="roc.png",
        save_results=f"results/quantum_dna_mri_federated/",
        yaml_path="./results/quantum_dna_mri_federated/results.yml",
        classes=classes
    )

def main():
    """Main federated learning orchestration using library components"""
    config = run_experiment(
        name="quantum_dna_mri_federated",
        dataset="DNA+MRI", 
        model="quantum_multimodal",
        epochs=5,
        learning_rate=0.001,
        batch_size=32,
        gpu=torch.cuda.is_available()
    )
    
    device = torch.device('cuda' if config['gpu'] else 'cpu')
    
    # Load datasets to get DNA input size (like in centralized example)
    _, _, testloader = load_datasets(
        num_clients=1,
        batch_size=config['batch_size'],
        resize=224,
        seed=42,
        num_workers=0,
        splitter=20,
        dataset='DNA+MRI',
        data_path="data/"
    )
    
    # Get actual DNA input size from the data
    _, dna_input_size = next(iter(testloader))[0][1].shape
    print(f"DNA input size: {dna_input_size}")
    
    # Create initial model using library components
    mri_classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    dna_classes = ['0', '1', '2', '3', '4', '5', '6']
    initial_model = QuantumMultimodalNet(len(mri_classes), len(dna_classes), dna_input_size).to(device)
    
    print("=" * 70)
    print("FEDERATED QUANTUM MULTIMODAL LEARNING - DNA+MRI (LIBRARY VERSION)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Clients: 4")
    print(f"Rounds: 10")
    print(f"Dataset: DNA+MRI (Multimodal)")
    print(f"Architecture: Library's Quantum-Enhanced Modular Network")
    print(f"MRI Classes: {len(mri_classes)} - {mri_classes}")
    print(f"DNA Classes: {len(dna_classes)} - {dna_classes}")
    print(f"DNA Input Size: {dna_input_size} features (TF-IDF)")
    print(f"Quantum Features: MRI (4 qubits), DNA (7 qubits)")
    print(f"Using: Fed-ML-Lib's MultimodalFlowerClient & ModularModel")
    
    start_time = time.time()
    
    # Configure federated learning strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=4,
        min_evaluate_clients=2,
        min_available_clients=4,
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters2(initial_model)),
        on_fit_config_fn=lambda round: {
            "learning_rate": str(config['learning_rate']),
            "batch_size": str(config['batch_size']),
            "server_round": round,
            "local_epochs": config['epochs']
        }
    )
    
    # Set up client resources
    client_resources = {'num_cpus': 1}
    if device.type == "cuda":
        client_resources["num_gpus"] = 1
    
    # Create results directory
    os.makedirs("results/quantum_dna_mri_federated", exist_ok=True)
    
    print("Starting federated learning simulation...")
    print("Using Fed-ML-Lib's built-in components:")
    print("  • MultimodalFlowerClient for federated training")
    print("  • ModularModel with quantum enhancements")
    print("  • Library's training, testing, and visualization functions")
    
    # Start federated learning simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=4,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
        client_resources=client_resources
    )
    
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print("Results saved to results/quantum_dna_mri_federated/ folder")

if __name__ == "__main__":
    main() 