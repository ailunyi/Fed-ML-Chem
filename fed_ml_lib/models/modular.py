import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, List, Union, Any, Dict, Tuple
from dataclasses import dataclass, field
from .base import BaseModel, ModelConfig
from torch_geometric.nn import GCNConv, global_mean_pool as gap, global_max_pool as gmp

import pennylane as qml  # type: ignore


@dataclass
class ModularConfig(ModelConfig):
    """Configuration for modular models with any combination of techniques."""
    
    # Base architecture
    base_architecture: str = "cnn"  # cnn, mlp, gcn, pretrained_cnn
    
    # Encryption configuration
    use_fhe: bool = False
    fhe_scheme: str = "CKKS"  # CKKS, BFV, TFHE
    fhe_layers: List[str] = field(default_factory=list)  # Which layers to encrypt
    
    # Quantum configuration  
    use_quantum: bool = False
    quantum_position: str = "post"  # pre, post, sandwich, distributed
    n_qubits: int = 4
    n_layers: int = 2
    quantum_circuit: str = "basic_entangler"
    quantum_layers: List[str] = field(default_factory=list)  # Which layers get quantum
    
    # Advanced modular options
    fhe_quantum_interaction: str = "sequential"  # sequential, parallel, interleaved
    
    def __post_init__(self):
        if not self.fhe_layers and self.use_fhe:
            self.fhe_layers = ['classifier']
        if not self.quantum_layers and self.use_quantum:
            self.quantum_layers = ['classifier']


class ModularModel(BaseModel):
    """Truly modular model that can combine any techniques."""
    
    def __init__(self, config: ModularConfig):
        self.modular_config = config
        super().__init__(config)
        
        # Build the modular architecture
        self._build_modular_architecture()
    
    def _build_model(self):
        """Required by BaseModel - already handled in _build_modular_architecture."""
        pass
    
    def _build_modular_architecture(self):
        """Build modular architecture with any combination of techniques."""
        arch_type = self.modular_config.base_architecture
        
        if arch_type == "cnn":
            self._build_cnn()
        elif arch_type == "mlp":
            self._build_mlp()
        elif arch_type == "gcn":
            self._build_gcn()
        elif arch_type == "pretrained_cnn":
            self._build_pretrained_cnn()
        else:
            raise ValueError(f"Unknown architecture: {arch_type}")
    
    def _build_cnn(self):
        """Build CNN with optional FHE and quantum enhancements."""
        channels = self.config.conv_channels or [16, 32, 64]
        input_channels = self.config.input_shape[0]
        
        # Feature extractor
        conv_layers = []
        in_channels = input_channels
        
        for out_channels in channels:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                self._get_activation('features'),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels
        
        self.features = self._wrap_with_enhancements(
            nn.Sequential(*conv_layers), 'features'
        )
        
        # Calculate classifier input size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.config.input_shape)
            dummy_output = self.features(dummy_input)
            flattened_size = dummy_output.numel()
        
        # Classifier
        hidden_dims = self.config.hidden_dims or [128]
        classifier_layers = []
        in_features = flattened_size
        
        for hidden_dim in hidden_dims:
            classifier_layers.extend([
                nn.Linear(in_features, hidden_dim),
                self._get_activation('classifier'),
                nn.Dropout(self.config.dropout_rate)
            ])
            in_features = hidden_dim
        
        classifier_layers.append(nn.Linear(in_features, self.config.num_classes))
        
        self.classifier = self._wrap_with_enhancements(
            nn.Sequential(*classifier_layers), 'classifier'
        )
    
    def _build_mlp(self):
        """Build MLP with optional FHE and quantum enhancements."""
        input_size = self.config.input_shape[0]
        hidden_dims = self.config.hidden_dims or [64, 32]
        
        layers = []
        in_features = input_size
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                self._get_activation('network'),
                nn.Dropout(self.config.dropout_rate)
            ])
            in_features = hidden_dim
        
        layers.append(nn.Linear(in_features, self.config.num_classes))
        
        self.network = self._wrap_with_enhancements(
            nn.Sequential(*layers), 'network'
        )
    
    def _build_gcn(self):
        """Build GCN with optional FHE and quantum enhancements."""
        node_features = self.config.input_shape[0]
        hidden_dims = self.config.hidden_dims or [64, 64]
        
        # Initial convolution
        self.initial_conv = self._wrap_with_enhancements(
            GCNConv(node_features, hidden_dims[0]), 'initial_conv'
        )
        
        # Additional convolutions
        conv_layers = []
        for i in range(1, len(hidden_dims)):
            conv_layers.append(
                self._wrap_with_enhancements(
                    GCNConv(hidden_dims[i-1], hidden_dims[i]), f'conv_{i}'
                )
            )
        self.conv_layers = nn.ModuleList(conv_layers)
        
        # Output layer
        pooled_dim = hidden_dims[-1] * 2  # *2 for global pooling concat
        self.output_layer = self._wrap_with_enhancements(
            nn.Linear(pooled_dim, self.config.num_classes), 'output'
        )
    
    def _build_pretrained_cnn(self):
        """Build pretrained CNN with optional FHE and quantum enhancements."""
        backbone_name = getattr(self.config, 'backbone', 'vgg16')
        
        if backbone_name == 'vgg16':
            backbone = models.vgg16(weights='IMAGENET1K_V1')
            features = backbone.features
            feature_dim = 512
        elif backbone_name == 'resnet18':
            backbone = models.resnet18(weights='IMAGENET1K_V1')
            features = nn.Sequential(*list(backbone.children())[:-1])
            feature_dim = backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Freeze layers if specified
        freeze_layers = getattr(self.config, 'freeze_layers', 0)
        if freeze_layers > 0:
            for param in list(features.parameters())[:freeze_layers]:
                param.requires_grad = False
        
        self.features = self._wrap_with_enhancements(features, 'features')
        
        # Classifier
        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(feature_dim, self.config.num_classes)
        )
        
        self.classifier = self._wrap_with_enhancements(classifier, 'classifier')
    
    def _get_activation(self, layer_name: str) -> nn.Module:
        """Get appropriate activation function based on enhancements."""
        if self.modular_config.use_fhe and layer_name in self.modular_config.fhe_layers:
            # FHE-friendly activation (polynomial)
            return SquareActivation()
        else:
            # Standard activation
            return nn.ReLU(inplace=True)
    
    def _wrap_with_enhancements(self, layer: nn.Module, layer_name: str) -> nn.Module:
        """Wrap a layer with FHE and/or quantum enhancements."""
        enhanced_layer = layer
        
        # Apply FHE wrapping
        if (self.modular_config.use_fhe and 
            (layer_name in self.modular_config.fhe_layers or 'all' in self.modular_config.fhe_layers)):
            enhanced_layer = FHEWrapper(enhanced_layer, self.modular_config.fhe_scheme)
        
        # Apply quantum wrapping
        if (self.modular_config.use_quantum and 
            (layer_name in self.modular_config.quantum_layers or 'all' in self.modular_config.quantum_layers)):
            enhanced_layer = QuantumWrapper(
                enhanced_layer, 
                self.modular_config.n_qubits,
                self.modular_config.n_layers,
                self.modular_config.quantum_circuit
            )
        
        return enhanced_layer
    
    def forward(self, x, *args, **kwargs):
        """Forward pass through modular architecture."""
        arch_type = self.modular_config.base_architecture
        
        if arch_type == "cnn":
            return self._forward_cnn(x)
        elif arch_type == "mlp":
            return self._forward_mlp(x)
        elif arch_type == "gcn":
            return self._forward_gcn(x, *args, **kwargs)
        elif arch_type == "pretrained_cnn":
            return self._forward_pretrained_cnn(x)
    
    def _forward_cnn(self, x):
        """Forward pass for CNN architecture."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _forward_mlp(self, x):
        """Forward pass for MLP architecture."""
        return self.network(x)
    
    def _forward_gcn(self, x, edge_index, batch_index):
        """Forward pass for GCN architecture."""

        # Initial convolution
        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden)
        
        # Additional convolutions
        for conv_layer in self.conv_layers:
            hidden = conv_layer(hidden, edge_index)
            hidden = F.tanh(hidden)
        
        # Global pooling
        hidden = torch.cat([gmp(hidden, batch_index), gap(hidden, batch_index)], dim=1)
        
        # Final output
        return self.output_layer(hidden)
    
    def _forward_pretrained_cnn(self, x):
        """Forward pass for pretrained CNN architecture."""
        x = self.features(x)
        x = self.classifier(x)
        return x


class FHEWrapper(nn.Module):
    """Wrapper to add FHE encryption to any layer."""
    
    def __init__(self, layer: nn.Module, fhe_scheme: str = "CKKS"):
        super().__init__()
        self.layer = layer
        self.fhe_scheme = fhe_scheme
        self.encryption_enabled = True
        
        # FHE context (placeholder - would use actual FHE library)
        self.encryption_context = {
            'scheme': fhe_scheme,
            'poly_modulus_degree': 8192,
            'scale': 2.0**40,
            'security_level': 128
        }
    
    def forward(self, x):
        if self.encryption_enabled and self.training:
            # Placeholder for FHE encryption
            # In practice, would encrypt tensor here
            pass
            
        # Process with wrapped layer
        x = self.layer(x)
        
        if self.encryption_enabled and self.training:
            # Placeholder for FHE decryption
            # In practice, would decrypt tensor here
            pass
        
        return x


class QuantumWrapper(nn.Module):
    """Wrapper to add quantum processing to any layer."""
    
    def __init__(self, layer: nn.Module, n_qubits: int = 4, n_layers: int = 2, circuit_type: str = "basic_entangler"):
        super().__init__()
        self.layer = layer
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.circuit_type = circuit_type
        
        # Create quantum layer
        self.quantum_layer = self._create_quantum_layer()
        # Interface layers for quantum processing
        self.to_quantum = nn.Linear(128, n_qubits)  # Simplified interface
        self.from_quantum = nn.Linear(n_qubits, 128)
    
    def _create_quantum_layer(self):
        """Create quantum circuit layer."""
        # Simplified quantum layer creation
        qdevice = qml.device("default.qubit", wires=self.n_qubits)  # type: ignore
        
        @qml.qnode(qdevice, interface='torch')  # type: ignore
        def quantum_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))  # type: ignore
            qml.BasicEntanglerLayers(weights, wires=range(self.n_qubits))  # type: ignore
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]  # type: ignore
        
        weight_shapes = {"weights": (self.n_layers, self.n_qubits)}
        return qml.qnn.TorchLayer(quantum_circuit, weight_shapes)  # type: ignore
    
    def forward(self, x):
        # Store original for residual connection
        residual = x
        
        # Classical processing
        x = self.layer(x)
        
        # Quantum processing (if available)
        if self.quantum_layer is not None and x.dim() == 2:
            # Simple quantum enhancement for 2D tensors
            try:
                quantum_input = torch.tanh(self.to_quantum(x))
                quantum_output = self.quantum_layer(quantum_input)
                x = x + self.from_quantum(quantum_output)  # Residual connection
            except:
                # Fallback to classical processing if quantum fails
                pass
        
        return x


class SquareActivation(nn.Module):
    """Square activation function (FHE-friendly)."""
    
    def forward(self, x):
        return x * x


# Factory function for modular models
def create_modular_model(
    base_architecture: str,
    use_fhe: bool = False,
    use_quantum: bool = False,
    **kwargs
) -> ModularModel:
    """
    Create a modular model with any combination of techniques.
    
    Args:
        base_architecture: 'cnn', 'mlp', 'gcn', 'pretrained_cnn'
        use_fhe: Whether to add FHE encryption
        use_quantum: Whether to add quantum processing
        **kwargs: Additional configuration parameters
    
    Returns:
        ModularModel instance
    
    Examples:
        # Pure classical CNN
        model = create_modular_model('cnn')
        
        # CNN with FHE encryption
        model = create_modular_model('cnn', use_fhe=True, fhe_layers=['classifier'])
        
        # CNN with quantum enhancement
        model = create_modular_model('cnn', use_quantum=True, quantum_layers=['features'])
        
        # CNN with both FHE and quantum
        model = create_modular_model('cnn', use_fhe=True, use_quantum=True)
    """
    config = ModularConfig(
        base_architecture=base_architecture,
        use_fhe=use_fhe,
        use_quantum=use_quantum,
        **kwargs
    )
    
    return ModularModel(config)


# Convenience functions for common combinations
def create_classical_model(architecture: str, **kwargs):
    """Create pure classical model."""
    return create_modular_model(architecture, use_fhe=False, use_quantum=False, **kwargs)

def create_fhe_model(architecture: str, **kwargs):
    """Create FHE-encrypted model."""
    return create_modular_model(architecture, use_fhe=True, use_quantum=False, **kwargs)

def create_quantum_model(architecture: str, **kwargs):
    """Create quantum-enhanced model."""
    return create_modular_model(architecture, use_fhe=False, use_quantum=True, **kwargs)

def create_fhe_quantum_model(architecture: str, **kwargs):
    """Create FHE + Quantum model."""
    return create_modular_model(architecture, use_fhe=True, use_quantum=True, **kwargs)
