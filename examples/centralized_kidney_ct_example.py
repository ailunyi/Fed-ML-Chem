"""
Federated Quantum Kidney CT Classification Example using Fed-ML-Lib
Following the paper methodology: FedQTNs with PCA preprocessing and quantum encoding
"""
import torch
import torch.nn as nn
import numpy as np
import os
import time
import flwr as fl
from flwr.server.strategy import FedAvg
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, confusion_matrix

from fed_ml_lib.config.python_config import run_experiment
from fed_ml_lib.models.modular import create_model
from fed_ml_lib.data.loaders import load_datasets
from fed_ml_lib.federated.client import FlowerClient
from fed_ml_lib.federated.utils import get_parameters2, set_parameters
from fed_ml_lib.core.training import train
from fed_ml_lib.core.testing import test
from fed_ml_lib.core.visualization import save_matrix, save_graphs
from fed_ml_lib.core.utils import choice_device

def pca_preprocess_images(dataloader, n_components=324):
    """
    Apply PCA preprocessing to reduce images to 18x18 (324 components)
    Following paper methodology: maintain ~99% variance
    """
    # Collect all images
    all_images = []
    all_labels = []
    
    for images, labels in dataloader:
        # Flatten images for PCA
        batch_size = images.size(0)
        flattened = images.view(batch_size, -1)  # Flatten to [batch, pixels]
        all_images.append(flattened)
        all_labels.append(labels)
    
    # Concatenate all data
    X = torch.cat(all_images, dim=0).numpy()
    y = torch.cat(all_labels, dim=0).numpy()
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    print(f"PCA: Reduced from {X.shape[1]} to {n_components} components")
    print(f"PCA: Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Reshape back to 18x18 images
    X_pca_reshaped = X_pca.reshape(-1, 1, 18, 18)  # [N, 1, 18, 18]
    
    # Create new dataset
    class PCADataset(torch.utils.data.Dataset):
        def __init__(self, data, labels):
            self.data = torch.FloatTensor(data)
            self.labels = torch.LongTensor(labels)
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
    
    # Create new dataloader
    pca_dataset = PCADataset(X_pca_reshaped, y)
    pca_dataloader = torch.utils.data.DataLoader(
        pca_dataset,
        batch_size=dataloader.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    return pca_dataloader, pca

def remap_labels_to_binary(dataloader, label_mapping=None):
    """
    Remap dataset labels to binary classification (0=Normal, 1=Malignant)
    If label_mapping is None, automatically map: lowest values -> 0, higher values -> 1
    """
    if label_mapping is None:
        # Get all unique labels
        all_labels = set()
        for _, labels in dataloader:
            all_labels.update(labels.tolist())
        
        sorted_labels = sorted(all_labels)
        print(f"Auto-mapping labels: {sorted_labels[:len(sorted_labels)//2]} -> Normal (0)")
        print(f"Auto-mapping labels: {sorted_labels[len(sorted_labels)//2:]} -> Malignant (1)")
        
        # Map first half to 0 (Normal), second half to 1 (Malignant)
        label_mapping = {}
        mid_point = len(sorted_labels) // 2
        for i, label in enumerate(sorted_labels):
            label_mapping[label] = 0 if i < mid_point else 1
    
    # Apply mapping to dataset
    original_dataset = dataloader.dataset
    
    # Create new dataset with remapped labels
    class BinaryDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset, label_mapping):
            self.original_dataset = original_dataset
            self.label_mapping = label_mapping
        
        def __len__(self):
            return len(self.original_dataset)
        
        def __getitem__(self, idx):
            data, label = self.original_dataset[idx]
            # Remap label to binary
            binary_label = self.label_mapping.get(label.item() if torch.is_tensor(label) else label, label)
            return data, torch.tensor(binary_label, dtype=torch.long)
    
    # Create new dataloader with binary labels
    binary_dataset = BinaryDataset(original_dataset, label_mapping)
    binary_dataloader = torch.utils.data.DataLoader(
        binary_dataset,
        batch_size=dataloader.batch_size,
        shuffle=hasattr(dataloader, 'shuffle') and dataloader.shuffle,
        num_workers=dataloader.num_workers
    )
    
    return binary_dataloader, label_mapping

def calculate_comprehensive_metrics(y_true, y_pred, y_proba):
    """Calculate AUC, specificity, sensitivity following paper methodology"""
    # Convert to numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)
    
    # AUC calculation
    if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        auc = roc_auc_score(y_true, y_proba)
    
    # Confusion matrix for specificity/sensitivity
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    else:
        specificity = sensitivity = 0.0
    
    return {
        'auc': auc,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'accuracy': (y_true == y_pred).mean()
    }

def client_fn(cid: str):
    """Create a Flower client for federated kidney CT classification"""
    config = run_experiment(
        name="kidney_ct_federated",
        dataset="kidney_ct",
        model="quantum_cnn",
        epochs=5,  # Local epochs per round
        learning_rate=0.001,
        batch_size=32,
        gpu=torch.cuda.is_available()
    )
    

    device = torch.device('cuda' if config['gpu'] else 'cpu')    
    
    # Load federated datasets for 4 hospitals
    trainloaders, valloaders, testloader = load_datasets(
        num_clients=4,  # 4 hospitals as per paper
        batch_size=config['batch_size'],
        resize=512,     # Paper uses 512x512 initial size
        seed=42,
        num_workers=0,
        splitter=20,
        dataset='kidney_ct',
        data_path="data/"
    )
    
    # Get client's data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    
    # Apply binary remapping
    trainloader, label_mapping = remap_labels_to_binary(trainloader)
    valloader, _ = remap_labels_to_binary(valloader, label_mapping)
    
    # Apply PCA preprocessing (512x512 -> 18x18)
    trainloader, pca_model = pca_preprocess_images(trainloader)
    valloader, _ = pca_preprocess_images(valloader)
    
    # Create quantum CNN model for 18x18 images with 2x2 patches -> 4 qubits
    model = create_model(
        base_architecture="cnn",
        input_shape=(1, 18, 18),         # PCA reduced size
        num_classes=2,
        use_quantum=True,                # Enable quantum processing
        conv_channels=[16, 32],          # Smaller channels for 18x18
        hidden_dims=[64, 32],            # Adjusted for smaller input
        dropout_rate=0.3,
        domain='vision',
        n_qubits=4,                      # 4 qubits for 2x2 patches
        n_layers=2,
        quantum_circuit="basic_entangler",
        quantum_layers=['classifier']
    ).to(device)
    
    class_names = ['Normal', 'Malignant']
    
    return FlowerClient(
        cid=cid,
        net=model,
        trainloader=trainloader,
        valloader=valloader,
        device=device,
        batch_size=config['batch_size'],
        save_results=f"results/federated_kidney_ct/hospital_{cid}/",
        matrix_path="confusion_matrix.png",
        roc_path="roc.png",
        yaml_path=None,
        classes=class_names
    )

def evaluate_fn(server_round, parameters, config):
    """Server-side evaluation following paper methodology"""
    
    device = torch.device('cuda' if config['gpu'] else 'cpu')
    
    # Load test dataset
    _, _, testloader = load_datasets(
        num_clients=1,
        batch_size=64,
        resize=512,     # Paper uses 512x512 initial
        seed=42,
        num_workers=0,
        splitter=20,
        dataset='kidney_ct',
        data_path="data/"
    )
    
    # Apply same preprocessing as training
    testloader, _ = remap_labels_to_binary(testloader)
    testloader, _ = pca_preprocess_images(testloader)
    
    # Create model for evaluation
    model = create_model(
        base_architecture="cnn",
        input_shape=(1, 18, 18),
        num_classes=2,
        use_quantum=True,
        conv_channels=[16, 32],
        hidden_dims=[64, 32],
        dropout_rate=0.3,
        domain='vision',
        n_qubits=4,
        n_layers=2,
        quantum_circuit="basic_entangler",
        quantum_layers=['classifier']
    ).to(device)
    
    # Load aggregated parameters
    set_parameters(model, parameters)
    
    # Test model
    loss_fn = torch.nn.CrossEntropyLoss()
    loss, accuracy, y_pred, y_true, y_proba = test(
        model=model,
        dataloader=testloader,
        loss_fn=loss_fn,
        device=device
    )
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(y_true, y_pred, y_proba)
    
    # Save results
    os.makedirs("results/federated_kidney_ct/server/", exist_ok=True)
    class_names = ['Normal', 'Malignant']
    save_matrix(y_true, y_pred, "results/federated_kidney_ct/server/confusion_matrix.png", class_names)
    
    print(f"Server evaluation - Round {server_round}:")
    print(f"  Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.1%}")
    print(f"  Sensitivity: {metrics['sensitivity']:.1%}")
    
    return loss, {"accuracy": accuracy, "auc": metrics['auc'], 
                  "specificity": metrics['specificity'], 
                  "sensitivity": metrics['sensitivity']}

def main():
    """Main function for federated quantum kidney CT classification following paper methodology"""
    
    # Configuration following paper
    config = run_experiment(
        name="kidney_ct_federated",
        dataset="kidney_ct",
        model="quantum_cnn",
        epochs=5,
        learning_rate=0.001,
        batch_size=32,
        gpu=torch.cuda.is_available()
    )
    
    device = torch.device('cuda' if config['gpu'] else 'cpu')
    
    # Get dataset info for initial model
    _, _, testloader = load_datasets(
        num_clients=1,
        batch_size=64,
        resize=512,     # Paper methodology: start with 512x512
        seed=42,
        num_workers=0,
        splitter=20,
        dataset='kidney_ct',
        data_path="data/"
    )
    
    # Apply preprocessing to determine final input shape
    testloader, _ = remap_labels_to_binary(testloader)
    testloader, _ = pca_preprocess_images(testloader)
    
    # Create initial quantum model (18x18 after PCA)
    initial_model = create_model(
        base_architecture="cnn",
        input_shape=(1, 18, 18),         # After PCA: 18x18 grayscale
        num_classes=2,
        use_quantum=True,                # Enable quantum processing
        conv_channels=[16, 32],          # Smaller for 18x18 input
        hidden_dims=[64, 32],
        dropout_rate=0.3,
        domain='vision',
        n_qubits=4,                      # 4 qubits for 2x2 patches
        n_layers=2,
        quantum_circuit="basic_entangler",
        quantum_layers=['classifier']
    ).to(device)
    
    # Configure federated learning strategy
    strategy = FedAvg(
        fraction_fit=1.0,               # Use all 4 hospitals
        fraction_evaluate=0.5,          # Use 2 hospitals for evaluation
        min_fit_clients=4,              # All 4 hospitals
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
        client_resources["num_gpus"] = 0.25  # Share GPU among 4 hospitals
    
    print("="*60)
    print("FEDERATED QUANTUM KIDNEY CT CLASSIFICATION")
    print("Following paper methodology: FedQTNs with PCA + Quantum encoding")
    print("="*60)
    print(f"Device: {device}")
    print(f"Architecture: Quantum CNN with 4 qubits (2x2 patches)")
    print(f"Preprocessing: 512x512 → PCA → 18x18 (324 components)")
    print(f"Hospitals (clients): 4")
    print(f"Dataset: Kidney CT (Normal vs Malignant)")
    print(f"Target metrics: AUC ~0.967, Specificity ~88.6%, Sensitivity ~92.5%")
    
    start_time = time.time()
    
    # Start federated quantum learning simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=4,                   # 4 hospitals as per paper
        config=fl.server.ServerConfig(num_rounds=10),  # Multiple rounds for convergence
        strategy=strategy,
        client_resources=client_resources
    )
    
    print(f"\nFederated quantum kidney CT training completed in {time.time() - start_time:.2f} seconds")
    print("\nResults saved to results/federated_kidney_ct/")
    print("- Server evaluation: confusion matrices and metrics")  
    print("- Hospital results: individual client directories")
    print("- Targeting paper metrics: AUC=0.9672, Spec=88.6%, Sens=92.5%")

if __name__ == "__main__":
    main() 