"""
models.py
---------
This module contains quantum, hybrid, and classical model definitions for the fed_ml_lib library.
"""

# Add your model classes and factory functions here 

import torch
from torch import nn
import torchvision.models as models
import pennylane as qml
import numpy as np
from typing import Optional

class VGG16Classifier(nn.Module):
    """
    A CNN model based on VGG16 for image classification.
    Args:
        num_classes: An integer indicating the number of output classes.
    """
    def __init__(self, num_classes=10) -> None:
        super().__init__()
        self.feature_extractor = models.vgg16(weights='IMAGENET1K_V1').features[:-1]
        # Get the number of output channels from the last Conv2d layer
        out_channels = 512  # VGG16's last conv layer has 512 output channels
        self.classification_head = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=(224 // 2 ** 5, 224 // 2 ** 5)),
            nn.Flatten(),
            nn.Linear(in_features=out_channels, out_features=num_classes),
        )
        # Freeze the first 23 modules' parameters using an index-based loop
        for idx, layer in enumerate(self.feature_extractor.children()):
            if idx < 23:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                break

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_maps = self.feature_extractor(x)
        scores = self.classification_head(feature_maps)
        return scores


class QuantumNet(nn.Module):
    """
    A hybrid quantum-classical neural network for classification.
    
    Args:
        input_size: Size of the input features
        num_classes: Number of output classes
        n_qubits: Number of qubits in the quantum circuit
        n_layers: Number of variational layers in the quantum circuit
        device_name: PennyLane device name (default: 'default.qubit')
    """
    def __init__(self, input_size: int, num_classes: int, n_qubits: int = 4, 
                 n_layers: int = 2, device_name: str = 'default.qubit'):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Classical preprocessing layers
        self.classical_layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_qubits)  # Reduce to number of qubits
        )
        
        # Quantum device and circuit
        self.dev = qml.device(device_name, wires=n_qubits)
        self.weight_shapes = {"weights": (n_layers, n_qubits)}
        
        # Create quantum node
        @qml.qnode(self.dev, interface='torch')
        def quantum_circuit(inputs, weights):
            # Encode classical data into quantum state
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            
            # Variational layers
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            
            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, self.weight_shapes)
        
        # Classical post-processing
        self.output_layer = nn.Linear(n_qubits, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
            
        # Classical preprocessing
        x = self.classical_layers(x)
        
        # Quantum processing
        x = self.quantum_layer(x)
        
        # Classical output
        x = self.output_layer(x)
        
        return x


class HybridCNN_QNN(nn.Module):
    """
    A hybrid CNN-Quantum neural network for image classification.
    
    Args:
        num_classes: Number of output classes
        n_qubits: Number of qubits in the quantum circuit
        n_layers: Number of variational layers in the quantum circuit
        device_name: PennyLane device name (default: 'default.qubit')
    """
    def __init__(self, num_classes: int, n_qubits: int = 4, 
                 n_layers: int = 2, device_name: str = 'default.qubit'):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # CNN feature extractor (simplified VGG-like)
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Adaptive pooling to fixed size
            nn.Flatten()
        )
        
        # Calculate the size after CNN layers
        cnn_output_size = 128 * 4 * 4  # 128 channels * 4 * 4 spatial dimensions
        
        # Classical layers to prepare for quantum processing
        self.classical_prep = nn.Sequential(
            nn.Linear(cnn_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_qubits)
        )
        
        # Quantum device and circuit
        self.dev = qml.device(device_name, wires=n_qubits)
        self.weight_shapes = {"weights": (n_layers, n_qubits)}
        
        # Create quantum node
        @qml.qnode(self.dev, interface='torch')
        def quantum_circuit(inputs, weights):
            # Encode classical data into quantum state
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            
            # Variational layers with entanglement
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            
            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, self.weight_shapes)
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        x = self.cnn_layers(x)
        
        # Classical preprocessing for quantum layer
        x = self.classical_prep(x)
        
        # Quantum processing
        x = self.quantum_layer(x)
        
        # Final classification
        x = self.classifier(x)
        
        return x


class VariationalQuantumClassifier(nn.Module):
    """
    A pure variational quantum classifier.
    
    Args:
        input_size: Size of the input features (should match n_qubits)
        num_classes: Number of output classes
        n_qubits: Number of qubits in the quantum circuit
        n_layers: Number of variational layers
        device_name: PennyLane device name (default: 'default.qubit')
    """
    def __init__(self, input_size: int, num_classes: int, n_qubits: int = 4, 
                 n_layers: int = 3, device_name: str = 'default.qubit'):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.input_size = input_size
        
        # Input preprocessing to match qubit count
        if input_size != n_qubits:
            self.input_prep = nn.Linear(input_size, n_qubits)
        else:
            self.input_prep = nn.Identity()
        
        # Quantum device
        self.dev = qml.device(device_name, wires=n_qubits)
        
        # Weight shapes for the variational circuit
        self.weight_shapes = {
            "weights": (n_layers, n_qubits, 3),  # 3 parameters per qubit per layer
        }
        
        # Create quantum node
        @qml.qnode(self.dev, interface='torch')
        def variational_circuit(inputs, weights):
            # Data encoding
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # Variational layers
            for layer in range(n_layers):
                # Parameterized rotations
                for i in range(n_qubits):
                    qml.RX(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                    qml.RZ(weights[layer, i, 2], wires=i)
                
                # Entangling gates
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if n_qubits > 2:
                    qml.CNOT(wires=[n_qubits - 1, 0])  # Circular entanglement
            
            # Measurements
            return [qml.expval(qml.PauliZ(i)) for i in range(min(num_classes, n_qubits))]
        
        self.quantum_layer = qml.qnn.TorchLayer(variational_circuit, self.weight_shapes)
        
        # Output processing if needed
        if num_classes > n_qubits:
            self.output_layer = nn.Linear(n_qubits, num_classes)
        else:
            self.output_layer = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        # Prepare input for quantum circuit
        x = self.input_prep(x)
        
        # Apply quantum circuit
        x = self.quantum_layer(x)
        
        # Final output processing
        x = self.output_layer(x)
        
        return x


def create_model(model_type: str, num_classes: int, **kwargs) -> nn.Module:
    """
    Factory function to create different types of models.
    
    Args:
        model_type: Type of model ('vgg16', 'quantum', 'hybrid_cnn_qnn', 'vqc')
        num_classes: Number of output classes
        **kwargs: Additional arguments for specific models
        
    Returns:
        PyTorch model instance
    """
    if model_type.lower() == 'vgg16':
        return VGG16Classifier(num_classes=num_classes)
    
    elif model_type.lower() == 'quantum':
        input_size = kwargs.get('input_size', 784)  # Default for flattened 28x28 images
        n_qubits = kwargs.get('n_qubits', 4)
        n_layers = kwargs.get('n_layers', 2)
        return QuantumNet(input_size, num_classes, n_qubits, n_layers)
    
    elif model_type.lower() == 'hybrid_cnn_qnn':
        n_qubits = kwargs.get('n_qubits', 4)
        n_layers = kwargs.get('n_layers', 2)
        return HybridCNN_QNN(num_classes, n_qubits, n_layers)
    
    elif model_type.lower() == 'vqc':
        input_size = kwargs.get('input_size', 4)
        n_qubits = kwargs.get('n_qubits', 4)
        n_layers = kwargs.get('n_layers', 3)
        return VariationalQuantumClassifier(input_size, num_classes, n_qubits, n_layers)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}") 