import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch_geometric.nn import GCNConv, global_mean_pool as gap, global_max_pool as gmp
from typing import Optional, List, Union, Any, Dict
from .base import BaseModel, ModelConfig

import pennylane as qml


class HybridConfig(ModelConfig):
    """Configuration for hybrid quantum-classical models."""
    
    # Quantum circuit parameters
    n_qubits: int = 4
    n_layers: int = 2
    quantum_device: str = "default.qubit"
    
    # Quantum circuit type
    circuit_type: str = "basic_entangler"  # basic_entangler, strongly_entangling, custom
    
    # Hybrid architecture type
    hybrid_type: str = "post_processing"  # post_processing, pre_processing, sandwich
    
    # Multi-modal parameters (for fusion models)
    modalities: List[str] = None
    fusion_type: str = "attention"  # attention, concatenation, gating
    
    def __post_init__(self):
        if self.modalities is None:
            self.modalities = []


class BaseHybridModel(BaseModel):
    """Base class for hybrid quantum-classical models."""
    
    def __init__(self, config: HybridConfig):
        self.hybrid_config = config
        super().__init__(config)
        self._setup_quantum_components()
    
    def _setup_quantum_components(self):
        """Setup quantum device and circuits."""
        self.qdevice = qml.device(
            self.hybrid_config.quantum_device, 
            wires=self.hybrid_config.n_qubits
        )
        self._create_quantum_circuit()
    
    def _create_quantum_circuit(self):
        """Create quantum circuit based on configuration."""
        n_qubits = self.hybrid_config.n_qubits
        n_layers = self.hybrid_config.n_layers
        circuit_type = self.hybrid_config.circuit_type
        
        @qml.qnode(self.qdevice, interface='torch')
        def quantum_circuit(inputs, weights):
            # Data encoding
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            
            # Variational layers
            if circuit_type == "basic_entangler":
                qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            elif circuit_type == "strongly_entangling":
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            else:  # custom circuit
                self._custom_quantum_circuit(weights)
            
            # Measurements
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        # Weight shapes
        if circuit_type == "basic_entangler":
            weight_shapes = {"weights": (n_layers, n_qubits)}
        elif circuit_type == "strongly_entangling":
            weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        else:
            weight_shapes = self._get_custom_weight_shapes()
        
        # Create quantum layer
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
    
    def _custom_quantum_circuit(self, weights):
        """Override this for custom quantum circuits."""
        # Default to basic entangler
        qml.BasicEntanglerLayers(weights, wires=range(self.hybrid_config.n_qubits))
    
    def _get_custom_weight_shapes(self):
        """Override this for custom weight shapes."""
        return {"weights": (self.hybrid_config.n_layers, self.hybrid_config.n_qubits)}


class HybridCNNClassifier(BaseHybridModel):
    """Hybrid CNN with quantum post-processing."""
    
    def _build_model(self):
        # Classical CNN feature extractor
        channels = self.config.conv_channels or [16, 32, 64]
        in_channels = self.config.input_shape[0]
        
        conv_layers = []
        for out_channels in channels:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels
        
        self.features = nn.Sequential(*conv_layers)
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.config.input_shape)
            dummy_output = self.features(dummy_input)
            flattened_size = dummy_output.numel()
        
        # Classical layers
        hidden_dims = self.config.hidden_dims or [128]
        
        if self.hybrid_config.hybrid_type == "post_processing":
            # Classical → Quantum
            self.classical_layers = self._build_classical_layers(
                flattened_size, hidden_dims + [self.hybrid_config.n_qubits]
            )
            self.quantum_to_output = nn.Linear(self.hybrid_config.n_qubits, self.config.num_classes)
            
        elif self.hybrid_config.hybrid_type == "pre_processing":
            # Quantum → Classical
            self.input_to_quantum = nn.Linear(flattened_size, self.hybrid_config.n_qubits)
            self.classical_layers = self._build_classical_layers(
                self.hybrid_config.n_qubits, hidden_dims + [self.config.num_classes]
            )
            
        else:  # sandwich
            # Classical → Quantum → Classical
            mid_dim = hidden_dims[0] if hidden_dims else 64
            self.pre_quantum = self._build_classical_layers(flattened_size, [mid_dim, self.hybrid_config.n_qubits])
            self.post_quantum = self._build_classical_layers(self.hybrid_config.n_qubits, [mid_dim, self.config.num_classes])
    
    def _build_classical_layers(self, input_dim: int, layer_dims: List[int]) -> nn.Sequential:
        """Build classical neural network layers."""
        layers = []
        in_dim = input_dim
        
        for i, out_dim in enumerate(layer_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(layer_dims) - 1:  # No activation on last layer
                layers.extend([
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.config.dropout_rate)
                ])
            in_dim = out_dim
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        # Hybrid processing
        if self.hybrid_config.hybrid_type == "post_processing":
            x = self.classical_layers(x)
            x = torch.tanh(x)  # Normalize for quantum layer
            x = self.quantum_layer(x)
            x = self.quantum_to_output(x)
            
        elif self.hybrid_config.hybrid_type == "pre_processing":
            x = torch.tanh(self.input_to_quantum(x))
            x = self.quantum_layer(x)
            x = self.classical_layers(x)
            
        else:  # sandwich
            x = self.pre_quantum(x)
            x = torch.tanh(x)
            x = self.quantum_layer(x)
            x = self.post_quantum(x)
        
        return x


class HybridMLPClassifier(BaseHybridModel):
    """Hybrid MLP with quantum components."""
    
    def _build_model(self):
        input_size = self.config.input_shape[0]
        hidden_dims = self.config.hidden_dims or [64, 32]
        
        if self.hybrid_config.hybrid_type == "post_processing":
            # Classical → Quantum
            self.classical_layers = self._build_classical_layers(
                input_size, hidden_dims + [self.hybrid_config.n_qubits]
            )
            self.quantum_to_output = nn.Linear(self.hybrid_config.n_qubits, self.config.num_classes)
            
        elif self.hybrid_config.hybrid_type == "pre_processing":
            # Quantum → Classical  
            self.input_to_quantum = nn.Linear(input_size, self.hybrid_config.n_qubits)
            self.classical_layers = self._build_classical_layers(
                self.hybrid_config.n_qubits, hidden_dims + [self.config.num_classes]
            )
            
        else:  # sandwich
            mid_dim = hidden_dims[0] if hidden_dims else 32
            self.pre_quantum = self._build_classical_layers(input_size, [mid_dim, self.hybrid_config.n_qubits])
            self.post_quantum = self._build_classical_layers(self.hybrid_config.n_qubits, [mid_dim, self.config.num_classes])
    
    def _build_classical_layers(self, input_dim: int, layer_dims: List[int]) -> nn.Sequential:
        """Build classical neural network layers."""
        layers = []
        in_dim = input_dim
        
        for i, out_dim in enumerate(layer_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(layer_dims) - 1:
                layers.extend([
                    self._get_activation(),
                    nn.Dropout(self.config.dropout_rate)
                ])
            in_dim = out_dim
        
        return nn.Sequential(*layers)
    
    def _get_activation(self):
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "gelu": nn.GELU()
        }
        return activations.get(self.config.activation, nn.ReLU())
    
    def forward(self, x):
        if self.hybrid_config.hybrid_type == "post_processing":
            x = self.classical_layers(x)
            x = torch.tanh(x)
            x = self.quantum_layer(x)
            x = self.quantum_to_output(x)
            
        elif self.hybrid_config.hybrid_type == "pre_processing":
            x = torch.tanh(self.input_to_quantum(x))
            x = self.quantum_layer(x)
            x = self.classical_layers(x)
            
        else:  # sandwich
            x = self.pre_quantum(x)
            x = torch.tanh(x)
            x = self.quantum_layer(x)
            x = self.post_quantum(x)
        
        return x


class HybridGCNClassifier(BaseHybridModel):
    """Hybrid Graph Convolutional Network with quantum components."""
    
    def _build_model(self):
        node_features = self.config.input_shape[0]
        hidden_dims = self.config.hidden_dims or [64, 64]
        
        # GCN layers
        self.initial_conv = GCNConv(node_features, hidden_dims[0])
        self.conv_layers = nn.ModuleList()
        
        for i in range(1, len(hidden_dims)):
            self.conv_layers.append(GCNConv(hidden_dims[i-1], hidden_dims[i]))
        
        # Hybrid processing
        pooled_dim = hidden_dims[-1] * 2  # *2 for global pooling concat
        
        if self.hybrid_config.hybrid_type == "post_processing":
            self.graph_to_quantum = nn.Linear(pooled_dim, self.hybrid_config.n_qubits)
            self.quantum_to_output = nn.Linear(self.hybrid_config.n_qubits, self.config.num_classes)
        else:
            self.output_layer = nn.Linear(pooled_dim, self.config.num_classes)
    
    def forward(self, x, edge_index, batch_index):
        # GCN processing
        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden)
        
        for conv_layer in self.conv_layers:
            hidden = conv_layer(hidden, edge_index)
            hidden = F.tanh(hidden)
        
        # Global pooling
        hidden = torch.cat([gmp(hidden, batch_index), gap(hidden, batch_index)], dim=1)
        
        # Hybrid processing
        if self.hybrid_config.hybrid_type == "post_processing":
            hidden = torch.tanh(self.graph_to_quantum(hidden))
            hidden = self.quantum_layer(hidden)
            output = self.quantum_to_output(hidden)
        else:
            output = self.output_layer(hidden)
        
        return output


class HybridPretrainedCNN(BaseHybridModel):
    """Hybrid pretrained CNN with quantum components."""
    
    def _build_model(self):
        # Load pretrained backbone
        backbone_name = getattr(self.config, 'backbone', 'vgg16')
        
        if backbone_name == 'vgg16':
            backbone = models.vgg16(weights='IMAGENET1K_V1')
            self.features = backbone.features[:-1]
            feature_dim = 512  # VGG16 feature dimension
        elif backbone_name == 'resnet18':
            backbone = models.resnet18(weights='IMAGENET1K_V1')
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            feature_dim = backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Freeze pretrained layers
        freeze_layers = getattr(self.config, 'freeze_layers', 0)
        if freeze_layers > 0:
            for param in list(self.features.parameters())[:freeze_layers]:
                param.requires_grad = False
        
        # Hybrid classifier
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        if self.hybrid_config.hybrid_type == "post_processing":
            self.classical_to_quantum = nn.Linear(feature_dim, self.hybrid_config.n_qubits)
            self.quantum_to_output = nn.Linear(self.hybrid_config.n_qubits, self.config.num_classes)
        else:
            self.classifier = nn.Linear(feature_dim, self.config.num_classes)
    
    def forward(self, x):
        # Pretrained feature extraction
        features = self.features(x)
        features = self.adaptive_pool(features)
        features = self.flatten(features)
        
        # Hybrid processing
        if self.hybrid_config.hybrid_type == "post_processing":
            features = torch.tanh(self.classical_to_quantum(features))
            features = self.quantum_layer(features)
            output = self.quantum_to_output(features)
        else:
            output = self.classifier(features)
        
        return output


class MultimodalHybridModel(BaseHybridModel):
    """Hybrid model for multimodal data fusion."""
    
    def __init__(self, config: HybridConfig):
        if not config.modalities:
            raise ValueError("Modalities must be specified for multimodal models")
        super().__init__(config)
    
    def _build_model(self):
        self.modality_encoders = nn.ModuleDict()
        self.modality_to_quantum = nn.ModuleDict()
        
        # Build encoders for each modality
        for modality in self.hybrid_config.modalities:
            encoder = self._build_modality_encoder(modality)
            self.modality_encoders[modality] = encoder
            
            # Modality-specific quantum interface
            encoder_dim = self._get_encoder_output_dim(modality)
            self.modality_to_quantum[modality] = nn.Linear(encoder_dim, self.hybrid_config.n_qubits)
        
        # Fusion mechanism
        if self.hybrid_config.fusion_type == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=self.hybrid_config.n_qubits * len(self.hybrid_config.modalities),
                num_heads=self.hybrid_config.n_qubits
            )
        
        # Final classifier
        fused_dim = self.hybrid_config.n_qubits * len(self.hybrid_config.modalities)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.config.num_classes)
        )
    
    def _build_modality_encoder(self, modality: str) -> nn.Module:
        """Build encoder for specific modality."""
        if modality == "vision":
            return nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten()
            )
        elif modality == "sequence":
            return nn.Sequential(
                nn.Linear(384, 512),  # DNA sequence length
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
        else:
            raise ValueError(f"Unknown modality: {modality}")
    
    def _get_encoder_output_dim(self, modality: str) -> int:
        """Get output dimension of modality encoder."""
        if modality == "vision":
            return 32 * 7 * 7  # 1568
        elif modality == "sequence":
            return 128
        else:
            return 64  # Default
    
    def forward(self, **inputs):
        """Forward pass with multiple modality inputs."""
        modality_features = []
        
        # Process each modality
        for modality in self.hybrid_config.modalities:
            if modality not in inputs:
                raise ValueError(f"Missing input for modality: {modality}")
            
            # Encode modality
            features = self.modality_encoders[modality](inputs[modality])
            
            # Classical to quantum
            quantum_input = torch.tanh(self.modality_to_quantum[modality](features))
            quantum_output = self.quantum_layer(quantum_input)
            
            modality_features.append(quantum_output)
        
        # Fusion
        if self.hybrid_config.fusion_type == "attention":
            combined = torch.cat(modality_features, dim=1)
            combined = combined.unsqueeze(0)  # Add sequence dimension
            fused, _ = self.attention(combined, combined, combined)
            fused = fused.squeeze(0)
        else:  # concatenation
            fused = torch.cat(modality_features, dim=1)
        
        # Final classification
        output = self.classifier(fused)
        return output


# Model registry for hybrid models
HYBRID_MODEL_REGISTRY = {
    'hybrid_cnn': HybridCNNClassifier,
    'hybrid_mlp': HybridMLPClassifier,
    'hybrid_gcn': HybridGCNClassifier,
    'hybrid_pretrained_cnn': HybridPretrainedCNN,
    'hybrid_multimodal': MultimodalHybridModel,
}


def create_hybrid_model(model_type: str, **kwargs) -> BaseHybridModel:
    """Factory function to create hybrid models."""
    
    if model_type not in HYBRID_MODEL_REGISTRY:
        raise ValueError(f"Unknown hybrid model type: {model_type}")
    
    # Create hybrid config
    config = HybridConfig(
        model_type=model_type,
        **kwargs
    )
    
    # Create and return model
    model_class = HYBRID_MODEL_REGISTRY[model_type]
    return model_class(config)


def infer_hybrid_model_type(dataset_name: str, input_shape: tuple) -> str:
    """Automatically infer hybrid model type from dataset characteristics."""
    
    # Graph datasets
    if dataset_name.lower() in ['hiv', 'molecules', 'proteins']:
        return 'hybrid_gcn'
    
    # Sequence/tabular datasets
    if len(input_shape) == 1 or dataset_name.lower() in ['dna', 'genomics', 'tabular']:
        return 'hybrid_mlp'
    
    # Image datasets
    if len(input_shape) == 3:
        # Use pretrained for complex datasets
        if input_shape[1] >= 224 or dataset_name.lower() in ['pill', 'medical', 'imagenet']:
            return 'hybrid_pretrained_cnn'
        else:
            return 'hybrid_cnn'
    
    # Default fallback
    return 'hybrid_mlp'


# Legacy model compatibility (using factory functions)
# def MRINet():
#     return create_hybrid_model(
#         'hybrid_cnn',
#         input_shape=(3, 224, 224),
#         num_classes=4,
#         conv_channels=[16, 32],
#         n_qubits=4,
#         n_layers=6,
#         hybrid_type="post_processing"
#     )

# def DNANet():
#     return create_hybrid_model(
#         'hybrid_mlp',
#         input_shape=(384,),
#         num_classes=7,
#         hidden_dims=[512, 256, 128],
#         n_qubits=7,
#         n_layers=7,
#         hybrid_type="post_processing",
#         activation="leaky_relu",
#         dropout_rate=0.5
#     )

# def HIVNet(num_classes):
#     return create_hybrid_model(
#         'hybrid_gcn',
#         input_shape=(9,),
#         num_classes=num_classes,
#         hidden_dims=[64, 64, 64, 64],
#         n_qubits=2,
#         n_layers=2,
#         hybrid_type="post_processing"
#     )

# def PILLNet(num_classes=10):
#     return create_hybrid_model(
#         'hybrid_pretrained_cnn',
#         input_shape=(3, 224, 224),
#         num_classes=num_classes,
#         backbone='vgg16',
#         freeze_layers=23,
#         n_qubits=2,
#         n_layers=2,
#         hybrid_type="post_processing"
#     )

# def WaferNet(num_classes=10):
#     return create_hybrid_model(
#         'hybrid_cnn',
#         input_shape=(3, 64, 64),
#         num_classes=num_classes,
#         conv_channels=[16, 32],
#         hidden_dims=[128],
#         n_qubits=9,
#         n_layers=9,
#         hybrid_type="post_processing"
#     )

# def MRI_DNA_Net(num_classes_mri, num_classes_dna):
    return create_hybrid_model(
        'hybrid_multimodal',
        input_shape=(3, 224, 224),  # Primary modality shape
        num_classes=max(num_classes_mri, num_classes_dna),  # Use max for compatibility
        modalities=['vision', 'sequence'],
        fusion_type='attention',
        n_qubits=4,
        n_layers=6
    ) 