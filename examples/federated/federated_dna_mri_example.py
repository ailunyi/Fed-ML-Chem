"""
Simple DNA+MRI Federated Learning Example
Uses the same logic as legacy Standard_FedNN_DNA+MRI but with fed_ml_lib functions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import flwr as fl
from flwr.server.strategy import FedAvg
import os
import time

from fed_ml_lib.data.loaders import load_datasets
from fed_ml_lib.models.modular import create_model
from fed_ml_lib.federated.client import MultimodalFlowerClient
from fed_ml_lib.federated.utils import get_parameters2
from fed_ml_lib.core.utils import choice_device

class MultimodalNet(nn.Module):
    """Multimodal network using library's create_model function"""
    def __init__(self, num_classes_mri, num_classes_dna, dna_input_size):
        super(MultimodalNet, self).__init__()
        
        # Expert vector size calculation from legacy
        expert_vector = 6  # (4 + 7) // 2 + 1
        
        # Create MRI model using library's modular system
        self.mri_net = create_model(
            base_architecture="cnn",
            input_shape=(3, 224, 224),
            num_classes=expert_vector,
            conv_channels=[16, 32],
            hidden_dims=[128],
            dropout_rate=0.0,
            domain="vision"
        )
        
        # Create DNA model using library's modular system
        self.dna_net = create_model(
            base_architecture="mlp",
            input_shape=(dna_input_size,),
            num_classes=expert_vector,
            hidden_dims=[64, 32, 16, 8],
            dropout_rate=0.0,
            domain="sequence"
        )
        
        self.feature_dim = expert_vector
        self.num_heads = expert_vector
        
        # Attention and gating mechanism from legacy
        self.attention = nn.MultiheadAttention(embed_dim=2*self.feature_dim, num_heads=self.num_heads)
        self.fc_gate = nn.Linear(2*self.feature_dim, 2) 
        self.fc2_mri = nn.Linear(self.feature_dim, num_classes_mri)
        self.fc2_dna = nn.Linear(self.feature_dim, num_classes_dna)
        
    def forward(self, mri_input, dna_input):
        # Extract features using library models
        mri_features = self.mri_net(mri_input)
        dna_features = self.dna_net(dna_input)
        
        # Combine features
        combined_features = torch.cat((mri_features, dna_features), dim=1)
        combined_features = combined_features.unsqueeze(0)
        
        # Apply attention mechanism
        attn_output, _ = self.attention(combined_features, combined_features, combined_features)
        attn_output = attn_output.squeeze(0)
        
        # Gating mechanism from legacy
        gates = F.softmax(self.fc_gate(attn_output), dim=1)
        combined_output = (gates[:, 0].unsqueeze(1) * mri_features + 
                           gates[:, 1].unsqueeze(1) * dna_features)
        
        # Final classification
        mri_output = self.fc2_mri(combined_output)
        dna_output = self.fc2_dna(combined_output)
        return mri_output, dna_output

def client_fn(cid: str):
    """Create a Flower client using library components"""
    # Configuration
    batch_size = 32
    device = torch.device(choice_device('gpu'))
    
    # Load federated datasets using library
    trainloaders, valloaders, testloader = load_datasets(
        num_clients=4,
        batch_size=batch_size,
        resize=224,
        seed=42,
        num_workers=0,
        splitter=20,
        dataset='DNA+MRI',
        data_path="data/"
    )
    
    # Get DNA input size from actual data
    _, dna_input_size = next(iter(testloader))[0][1].shape
    
    # Class definitions from legacy
    mri_classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    dna_classes = ['0', '1', '2', '3', '4', '5', '6']
    classes = (mri_classes, dna_classes)
    
    # Create model with legacy architecture
    model = MultimodalNet(len(mri_classes), len(dna_classes), dna_input_size).to(device)
    
    # Use library's MultimodalFlowerClient
    return MultimodalFlowerClient(
        cid=cid,
        net=model,
        trainloader=trainloaders[int(cid)],
        valloader=valloaders[int(cid)],
        device=device,
        batch_size=batch_size,
        matrix_path="confusion_matrix.png",
        roc_path="roc.png",
        save_results="results/federated_dna_mri/",
        yaml_path="./results/federated_dna_mri/results.yml",
        classes=classes
    )

def main():
    """Main federated learning orchestration"""
    # Configuration
    batch_size = 32
    device = torch.device(choice_device('gpu'))
    
    # Load datasets to get model parameters
    _, _, testloader = load_datasets(
        num_clients=1,
        batch_size=batch_size,
        resize=224,
        seed=42,
        num_workers=0,
        splitter=20,
        dataset='DNA+MRI',
        data_path="data/"
    )
    
    # Get DNA input size
    _, dna_input_size = next(iter(testloader))[0][1].shape
    
    # Classes from legacy
    mri_classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    dna_classes = ['0', '1', '2', '3', '4', '5', '6']
    
    # Create initial model
    initial_model = MultimodalNet(len(mri_classes), len(dna_classes), dna_input_size).to(device)
    
    print("=" * 50)
    print("FEDERATED DNA+MRI LEARNING")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Clients: 4")
    print(f"Rounds: 10")
    print(f"Dataset: DNA+MRI (Multimodal)")
    
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
            "learning_rate": "0.001",
            "batch_size": str(batch_size),
            "server_round": round,
            "local_epochs": 5
        }
    )
    
    # Set up client resources
    client_resources = {'num_cpus': 1}
    if device.type == "cuda":
        client_resources["num_gpus"] = 1
    
    # Create results directory
    os.makedirs("results/federated_dna_mri", exist_ok=True)
    
    # Start federated learning simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=4,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
        client_resources=client_resources
    )
    
    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 