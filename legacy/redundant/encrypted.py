import os
import pickle
import time
from collections import OrderedDict
from typing import (
    List, Tuple, Dict, Optional, Callable, Union, cast
)
import tenseal as ts
from io import BytesIO

import numpy as np
import torchvision
import torch
from torch import nn
import torch.nn.functional as F
import flwr as fl
from flwr.common import (
    Metrics, EvaluateIns, EvaluateRes, FitIns, FitRes, MetricsAggregationFn, 
    Scalar, logger, Parameters, NDArray, NDArrays
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import weighted_loss_avg
from logging import WARNING
import pennylane as qml

from .base import BaseModel, ModelConfig

# dev = qml.device("default.qubit", wires=n_qubits)
    
# @qml.qnode(dev, interface='torch')
# def quantum_net(inputs, weights):
#     qml.AngleEmbedding(inputs, wires=range(n_qubits)) 
#     qml.BasicEntanglerLayers(weights,wires=range(n_qubits))
#     return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# class FHE_CIFARNet(nn.Module):
#     """
#     A simple CNN model

#     Args:
#         num_classes: An integer indicating the number of classes in the dataset.
#     """
#     def __init__(self, num_classes=10) -> None:
#         super(Net, self).__init__()
#         self.network = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

#             nn.Flatten(), 
#             nn.Linear(256*4*4, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, num_classes))              

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of the neural network
#         """
#         x = self.network(x)
#         return x
    

# class FHE_MRINet(nn.Module):
#     """
#     A simple CNN model

#     Args:
#         num_classes: An integer indicating the number of classes in the dataset.
#     """
#     def __init__(self, num_classes=10) -> None:
#         super(Net, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(32 * 56 * 56, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of the neural network
#         """
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x 
    

# class FHE_QNN_CIFARNet(nn.Module):
#     """
#     A simple CNN model

#     Args:
#         num_classes: An integer indicating the number of classes in the dataset.
#     """
#     def __init__(self, num_classes=10) -> None:
#         super(Net, self).__init__()
#         self.network = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

#             nn.Flatten(), 
#             nn.Linear(256*4*4, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, n_qubits))        
#         self.qnn = qml.qnn.TorchLayer(quantum_net, weight_shapes)
#         self.fc4 = nn.Linear(n_qubits, num_classes)        

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of the neural network
#         """
#         x = self.network(x)
#         x = self.qnn(x)
#         x = self.fc4(x)
#         return x
    
    

# class FHE_QNN_MRINet(nn.Module):
#     """
#     A simple CNN model

#     Args:
#         num_classes: An integer indicating the number of classes in the dataset.
#     """
#     def __init__(self, num_classes=10) -> None:
#         super(Net, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(32 * 56 * 56, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, n_qubits),
#             qml.qnn.TorchLayer(quantum_net, weight_shapes=weight_shapes),
#             nn.Linear(n_qubits, num_classes)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of the neural network
#         """
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x 
    

class FHEConfig(ModelConfig):
    """Configuration for FHE models."""
    
    # FHE-specific parameters
    encryption_scheme: str = "CKKS"  # CKKS, BFV, TFHE
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: List[int] = None
    scale: float = 2.0**40
    
    # Quantization parameters
    n_bits: int = 8
    use_quantization: bool = True
    
    # Security parameters
    security_level: int = 128
    
    def __post_init__(self):
        if self.coeff_mod_bit_sizes is None:
            self.coeff_mod_bit_sizes = [60, 40, 40, 60]


class BaseFHEModel(BaseModel):
    """Base class for FHE-enabled models."""
    
    def __init__(self, config: FHEConfig):
        self.fhe_config = config
        super().__init__(config)
        self._setup_encryption()
    
    def _setup_encryption(self):
        self.encryption_context = self._create_encryption_context()
        self.encryption_enabled = True
    
    def _create_encryption_context(self):
        """Create encryption context based on configuration."""
        # This would be implemented with actual FHE library
        # Placeholder implementation
        return {
            'scheme': self.fhe_config.encryption_scheme,
            'poly_modulus_degree': self.fhe_config.poly_modulus_degree,
            'scale': self.fhe_config.scale,
            'security_level': self.fhe_config.security_level
        }
    
    def encrypt_weights(self):
        """Encrypt model weights for FHE computation."""
        if not self.encryption_enabled:
            print("Warning: Encryption not available")
            return
        
        # Placeholder for weight encryption
        for name, param in self.named_parameters():
            if param.requires_grad:
                # In practice, this would encrypt the weights
                param.data = self._encrypt_tensor(param.data)
    
    def _encrypt_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Encrypt a tensor using FHE."""
        # Placeholder - would use actual FHE library
        return tensor  # Return original for now
    
    def _decrypt_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Decrypt a tensor from FHE."""
        # Placeholder - would use actual FHE library
        return tensor  # Return original for now


class FHECNNClassifier(BaseFHEModel):
    """FHE-enabled CNN for image classification."""
    
    def _build_model(self):
        # Build CNN architecture similar to classical version
        channels = self.config.conv_channels or [16, 32, 64]
        in_channels = self.config.input_shape[0]
        
        # Convolutional layers with FHE-friendly activations
        conv_layers = []
        for out_channels in channels:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                # Use polynomial activation for FHE compatibility
                self._get_fhe_activation(),
                nn.AvgPool2d(kernel_size=2, stride=2)  # AvgPool is FHE-friendly
            ])
            in_channels = out_channels
        
        self.features = nn.Sequential(*conv_layers)
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.config.input_shape)
            dummy_output = self.features(dummy_input)
            flattened_size = dummy_output.numel()
        
        # FHE-friendly classifier
        hidden_dims = self.config.hidden_dims or [128]
        classifier_layers = []
        
        in_features = flattened_size
        for hidden_dim in hidden_dims:
            classifier_layers.extend([
                nn.Linear(in_features, hidden_dim),
                self._get_fhe_activation()
            ])
            in_features = hidden_dim
        
        classifier_layers.append(nn.Linear(in_features, self.config.num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
    
    def _get_fhe_activation(self):
        """Get FHE-compatible activation function."""
        # FHE works better with polynomial activations
        if self.fhe_config.use_quantization:
            return SquareActivation()  # x^2 is FHE-friendly
        else:
            return nn.ReLU()  # Fallback to ReLU
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        # Apply FHE computation if enabled
        if self.encryption_enabled and self.training:
            x = self._fhe_forward(x)
        
        return x
    
    def _fhe_forward(self, x):
        """Forward pass with FHE computation."""
        # Placeholder for FHE computation
        return x


class FHEMLPClassifier(BaseFHEModel):
    """FHE-enabled MLP for sequence/tabular data."""
    
    def _build_model(self):
        input_size = self.config.input_shape[0]
        hidden_dims = self.config.hidden_dims or [64, 32]
        
        layers = []
        in_features = input_size
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                self._get_fhe_activation()
            ])
            in_features = hidden_dim
        
        layers.append(nn.Linear(in_features, self.config.num_classes))
        self.network = nn.Sequential(*layers)
    
    def _get_fhe_activation(self):
        """Get FHE-compatible activation function."""
        return SquareActivation() if self.fhe_config.use_quantization else nn.ReLU()
    
    def forward(self, x):
        return self.network(x)


class FHEQuantumCNN(BaseFHEModel):
    """FHE-enabled Quantum-Classical Hybrid CNN."""
    
    def __init__(self, config: FHEConfig):
        super().__init__(config)
    
    def _build_model(self):
        # Classical CNN feature extractor
        channels = self.config.conv_channels or [32, 64]
        in_channels = self.config.input_shape[0]
        
        conv_layers = []
        for out_channels in channels:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                self._get_fhe_activation(),
                nn.AvgPool2d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels
        
        self.features = nn.Sequential(*conv_layers)
        
        # Calculate size for quantum layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.config.input_shape)
            dummy_output = self.features(dummy_input)
            flattened_size = dummy_output.numel()
        
        # Classical-to-quantum interface
        self.classical_to_quantum = nn.Linear(flattened_size, self.config.n_qubits)
        
        # Quantum layer
        self._setup_quantum_layer()
        
        # Quantum-to-classical interface
        self.quantum_to_classical = nn.Linear(self.config.n_qubits, self.config.num_classes)
    
    def _setup_quantum_layer(self):
        """Setup quantum circuit layer."""
        n_qubits = self.config.n_qubits
        n_layers = self.config.n_layers
        
        # Create quantum device
        self.qdevice = qml.device("default.qubit", wires=n_qubits)
        
        # Define quantum circuit
        @qml.qnode(self.qdevice, interface='torch')
        def quantum_circuit(inputs, weights):
            # Angle embedding
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            
            # Variational layers
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            
            # Measurements
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        # Weight shapes for quantum circuit
        weight_shapes = {"weights": (n_layers, n_qubits)}
        
        # Create quantum layer
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
    
    def _get_fhe_activation(self):
        return SquareActivation() if self.fhe_config.use_quantization else nn.ReLU()
    
    def forward(self, x):
        # Classical feature extraction
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        # Classical-to-quantum
        x = torch.tanh(self.classical_to_quantum(x))  # Normalize for quantum
        
        # Quantum processing
        x = self.quantum_layer(x)
        
        # Quantum-to-classical
        x = self.quantum_to_classical(x)
        
        return x


class SquareActivation(nn.Module):
    """Square activation function (FHE-friendly)."""
    
    def forward(self, x):
        return x * x


class FHEPretrainedCNN(BaseFHEModel):
    """FHE-enabled pretrained CNN with encrypted fine-tuning."""
    
    def _build_model(self):
        import torchvision.models as models
        
        # Load pretrained model
        backbone_name = getattr(self.config, 'backbone', 'resnet18')
        
        if backbone_name == 'resnet18':
            backbone = models.resnet18(weights='IMAGENET1K_V1')
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            feature_dim = backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone for FHE: {backbone_name}")
        
        # Freeze pretrained layers (only fine-tune classifier with FHE)
        for param in self.features.parameters():
            param.requires_grad = False
        
        # FHE-enabled classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 128),
            self._get_fhe_activation(),
            nn.Linear(128, self.config.num_classes)
        )
    
    def _get_fhe_activation(self):
        return SquareActivation() if self.fhe_config.use_quantization else nn.ReLU()
    
    def forward(self, x):
        # Pretrained features (not encrypted)
        with torch.no_grad():
            features = self.features(x)
        
        # FHE-enabled classification
        return self.classifier(features)


# Model registry for encrypted models
FHE_MODEL_REGISTRY = {
    'fhe_cnn': FHECNNClassifier,
    'fhe_mlp': FHEMLPClassifier,
    'fhe_quantum_cnn': FHEQuantumCNN,
    'fhe_pretrained_cnn': FHEPretrainedCNN,
}


def create_fhe_model(model_type: str, **kwargs) -> BaseFHEModel:
    """Factory function to create FHE models."""
    
    if model_type not in FHE_MODEL_REGISTRY:
        raise ValueError(f"Unknown FHE model type: {model_type}")
    
    # Create FHE config
    config = FHEConfig(
        model_type=model_type,
        **kwargs
    )
    
    # Create and return model
    model_class = FHE_MODEL_REGISTRY[model_type]
    return model_class(config)


# Utility functions for FHE operations
class FHEUtils:
    """Utility functions for FHE operations."""
    
    @staticmethod
    def quantize_model(model: nn.Module, n_bits: int = 8):
        """Quantize model for FHE compatibility."""
        # Placeholder for quantization
        return model
    
    @staticmethod
    def encrypt_model_weights(model: BaseFHEModel):
        """Encrypt all model weights."""
        if hasattr(model, 'encrypt_weights'):
            model.encrypt_weights()
        else:
            print("Warning: Model does not support weight encryption")
    
    @staticmethod
    def setup_fhe_training(model: BaseFHEModel, dataloader):
        """Setup FHE-compatible training loop."""
        # Placeholder for FHE training setup
        return model, dataloader



FHE_CIFARNet = lambda num_classes=10: create_fhe_model(
    'fhe_cnn', 
    input_shape=(3, 32, 32), 
    num_classes=num_classes,
    conv_channels=[32, 64, 128]
)

FHE_MRINet = lambda num_classes=4: create_fhe_model(
    'fhe_cnn',
    input_shape=(3, 224, 224),
    num_classes=num_classes,
    conv_channels=[16, 32]
)

FHE_QNN_CIFARNet = lambda num_classes=10: create_fhe_model(
    'fhe_quantum_cnn',
    input_shape=(3, 32, 32),
    num_classes=num_classes,
    n_qubits=4,
    n_layers=2
)

FHE_QNN_MRINet = lambda num_classes=4: create_fhe_model(
    'fhe_quantum_cnn',
    input_shape=(3, 224, 224),
    num_classes=num_classes,
    n_qubits=4,
    n_layers=2
)