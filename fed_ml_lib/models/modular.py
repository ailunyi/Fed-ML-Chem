import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import tenseal as ts
from fed_ml_lib.core import security
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
    
    # Required by ModelConfig
    input_shape: Tuple[int, ...] = (3, 224, 224)  # Default for vision tasks
    num_classes: int = 2  # Default binary classification
    hidden_dims: Optional[List[int]] = None
    conv_channels: Optional[List[int]] = None
    dropout_rate: float = 0.0
    activation: str = "relu"
    domain: str = "vision"
    use_pretrained: bool = False
    freeze_layers: int = 0
    
    # CNN-specific customization parameters
    kernel_size: int = 3  # Convolutional kernel size (default 3x3)
    pooling_type: str = "max"  # "max", "avg", or "none"
    output_activation: str = "none"  # "sigmoid", "softmax", "relu", or "none"
    
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
        kernel_size = getattr(self.modular_config, 'kernel_size', 3)
        pooling_type = getattr(self.modular_config, 'pooling_type', 'max')
        
        # Feature extractor
        conv_layers = []
        in_channels = input_channels
        
        for out_channels in channels:
            # Add convolution layer
            padding = kernel_size // 2  # Maintain spatial dimensions
            conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            )
            conv_layers.append(self._get_activation('features'))
            
            # Add pooling layer based on configuration
            if pooling_type == "max":
                conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif pooling_type == "avg":
                conv_layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            # If pooling_type == "none", don't add any pooling
            
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
        
        # Final output layer
        classifier_layers.append(nn.Linear(in_features, self.config.num_classes))
        
        # Add output activation if specified
        output_activation = getattr(self.modular_config, 'output_activation', 'none')
        if output_activation == "sigmoid":
            classifier_layers.append(nn.Sigmoid())
        elif output_activation == "softmax":
            classifier_layers.append(nn.Softmax(dim=1))
        elif output_activation == "relu":
            classifier_layers.append(nn.ReLU())
        # If output_activation == "none", don't add any activation
        
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
            features = backbone.features[:-1]  # Remove last maxpool to match legacy
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
        
        # Build classification head
        if hasattr(self.config, 'classification_head'):
            # Custom classification head from config
            classifier_layers = []
            for layer_type, layer_params in self.config.classification_head:
                if layer_type == 'MaxPool2d':
                    classifier_layers.append(nn.MaxPool2d(**layer_params))
                elif layer_type == 'AvgPool2d':
                    classifier_layers.append(nn.AvgPool2d(**layer_params))
                elif layer_type == 'Flatten':
                    classifier_layers.append(nn.Flatten())
                elif layer_type == 'Linear':
                    classifier_layers.append(nn.Linear(**layer_params))
                elif layer_type == 'Dropout':
                    classifier_layers.append(nn.Dropout(**layer_params))
            classifier = nn.Sequential(*classifier_layers)
        else:
            # Default classification head
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
        
        
        self.security = security
        self.ts = ts
        
        # FHE context - will be set during first forward pass
        self.fhe_context = None
        self.encrypted_weights = {}
        
    def _initialize_fhe_context(self):
        """Initialize FHE context on first use."""
        if self.fhe_context is not None:
            return
            
        try:
            self.fhe_context = self.security.context()
            print(f"FHEWrapper: Initialized {self.fhe_scheme} context")
        except Exception as e:
            print(f"FHEWrapper: Failed to initialize FHE context: {e}")
            raise e
    
    def _encrypt_weights(self):
        """Encrypt layer weights for homomorphic operations."""
        if not self.fhe_context:
            return
            
        try:
            for name, param in self.layer.named_parameters():
                if param.requires_grad:
                    # Convert to numpy and encrypt
                    weight_np = param.detach().cpu().numpy()
                    encrypted_weight = self.ts.ckks_tensor(self.fhe_context, weight_np)
                    self.encrypted_weights[name] = encrypted_weight
                    
        except Exception as e:
            print(f"FHEWrapper: Weight encryption failed: {e}")
            raise e
    
    def _decrypt_weights(self):
        """Decrypt weights back to PyTorch tensors."""
        if not self.encrypted_weights:
            return
            
        try:
            for name, param in self.layer.named_parameters():
                if name in self.encrypted_weights:
                    # Decrypt and convert back to tensor
                    decrypted_weight = self.encrypted_weights[name].decrypt()
                    param.data = torch.tensor(decrypted_weight, dtype=param.dtype, device=param.device)
                    
        except Exception as e:
            print(f"FHEWrapper: Weight decryption failed: {e}")
            raise e
    
    def forward(self, x):
        """Forward pass with optional FHE operations."""
        if self.encryption_enabled and self.training:
            # Initialize FHE context if needed
            if self.fhe_context is None:
                self._initialize_fhe_context()
            
            # For training, we can encrypt weights for secure aggregation
            # but still need to use regular forward pass for gradient computation
            if self.fhe_context is not None:
                # Store current state for potential encryption during parameter sharing
                pass
        
        # Process with wrapped layer (always use regular computation for now)
        # In a full FHE implementation, this would use homomorphic operations
        x = self.layer(x)
        
        return x
    
    def get_encrypted_parameters(self):
        """Get encrypted parameters for secure federated aggregation."""
        self._encrypt_weights()
        return {name: encrypted.serialize() 
               for name, encrypted in self.encrypted_weights.items()}
    
    def set_encrypted_parameters(self, encrypted_params):
        """Set parameters from encrypted data."""
        try:
            # Deserialize and decrypt parameters
            for name, encrypted_data in encrypted_params.items():
                if isinstance(encrypted_data, bytes):
                    # Deserialize encrypted tensor
                    encrypted_tensor = self.ts.ckks_tensor_from(self.fhe_context, encrypted_data)
                    decrypted_data = encrypted_tensor.decrypt()
                    
                    # Set parameter
                    for param_name, param in self.layer.named_parameters():
                        if param_name == name:
                            param.data = torch.tensor(decrypted_data, 
                                                    dtype=param.dtype, device=param.device)
                            break
                            
        except Exception as e:
            print(f"FHEWrapper: Failed to set encrypted parameters: {e}")
            raise e
    
    def enable_encryption(self):
        """Enable FHE encryption for this layer."""
        self.encryption_enabled = True
        
    def disable_encryption(self):
        """Disable FHE encryption for this layer."""
        self.encryption_enabled = False


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


def create_model(
    base_architecture: str,
    input_shape: Tuple[int, ...],
    num_classes: int,
    use_fhe: bool = False,
    use_quantum: bool = False,
    hidden_dims: Optional[List[int]] = None,
    conv_channels: Optional[List[int]] = None,
    dropout_rate: float = 0.0,
    activation: str = "relu",
    domain: str = "vision",
    use_pretrained: bool = False,
    freeze_layers: int = 0,
    **kwargs
) -> ModularModel:
    """
    Create a modular model with any combination of techniques.
    
    Args:
        base_architecture: 'cnn', 'mlp', 'gcn', 'pretrained_cnn'
        input_shape: Shape of input data (e.g., (3, 224, 224) for RGB images)
        num_classes: Number of output classes
        use_fhe: Whether to add FHE encryption
        use_quantum: Whether to add quantum processing
        hidden_dims: List of hidden layer dimensions
        conv_channels: List of convolutional channel sizes
        dropout_rate: Dropout probability
        activation: Activation function to use
        domain: Domain of the task (vision, sequence, graph, medical, molecular)
        use_pretrained: Whether to use pretrained weights
        freeze_layers: Number of layers to freeze
        **kwargs: Additional configuration parameters
    
    Returns:
        ModularModel instance
    """
    config = ModularConfig(
        base_architecture=base_architecture,
        input_shape=input_shape,
        num_classes=num_classes,
        use_fhe=use_fhe,
        use_quantum=use_quantum,
        hidden_dims=hidden_dims,
        conv_channels=conv_channels,
        dropout_rate=dropout_rate,
        activation=activation,
        domain=domain,
        use_pretrained=use_pretrained,
        freeze_layers=freeze_layers,
        **kwargs
    )
    
    return ModularModel(config)

