import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torch_geometric.nn import GCNConv, global_mean_pool as gap, global_max_pool as gmp
from .base import BaseModel, ModelConfig

"""
This file contains the classical model classes 
"""

# # Quantum device definitions for MRI and DNA
# mri_n_qubits = 4
# dna_n_qubits = 7
# expert_vector = (mri_n_qubits+dna_n_qubits) // 2 + 1
# num_of_expert = 2

# # Define the MRI network
# class MRINet(nn.Module):
#     def __init__(self):
#         super(MRINet, self).__init__()
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
#             nn.Linear(128, expert_vector),
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x

# class DNANet(nn.Module):
#     def __init__(self):
#         super(DNANet, self).__init__()        
#         self.fc1 = nn.Linear(input_sp, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 16)
#         self.fc4 = nn.Linear(16, 8)
#         self.fc5 = nn.Linear(8, expert_vector)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of the neural network
#         """
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         x = self.fc5(x)
#         return x

# class DNA_MRINet(nn.Module):
    
#     def __init__(self, num_classes_mri, num_classes_dna):
#         super(DNA_MRINet, self).__init__()
#         self.mri_net = MRINet()
#         self.dna_net = DNANet()
        
#         self.feature_dim = expert_vector
#         self.num_heads = expert_vector
        
#         self.attention = nn.MultiheadAttention(embed_dim=num_of_expert*self.feature_dim, num_heads=self.num_heads)
#         self.fc_gate = nn.Linear(num_of_expert*self.feature_dim, 2) 
#         self.fc2_mri = nn.Linear(self.feature_dim, num_classes_mri)
#         self.fc2_dna = nn.Linear(self.feature_dim, num_classes_dna)
        
#     def forward(self, mri_input, dna_input):
#         mri_features = self.mri_net(mri_input)
#         dna_features = self.dna_net(dna_input)
#         combined_features = torch.cat((mri_features, dna_features), dim=1)
#         combined_features = combined_features.unsqueeze(0)
#         attn_output, _ = self.attention(combined_features, combined_features, combined_features)
#         attn_output = attn_output.squeeze(0)
#         gates = F.softmax(self.fc_gate(attn_output), dim=1)
#         combined_output = (gates[:, 0].unsqueeze(1) * mri_features + 
#                            gates[:, 1].unsqueeze(1) * dna_features)
#         mri_output = self.fc2_mri(combined_output)
#         dna_output = self.fc2_dna(combined_output)
#         return mri_output, dna_output
    
# embedding_size = batch_size = 64
# class HIVNet(nn.Module):
#     """
#     Args:
#         num_classes: An integer indicating the number of classes in the dataset.
#     """
#     def __init__(self, num_classes):
#         super(HIVNet, self).__init__()
#         self.initial_conv = GCNConv(9, embedding_size)
#         self.conv1 = GCNConv(embedding_size, embedding_size)
#         self.conv2 = GCNConv(embedding_size, embedding_size)
#         self.conv3 = GCNConv(embedding_size, embedding_size)

#         self.out = nn.Linear(embedding_size * 2, num_classes)

#     def forward(self, x, edge_index, batch_index):
#         """
#         Forward pass of the neural network
#         """
#         hidden = self.initial_conv(x, edge_index)
#         hidden = F.tanh(hidden)
#         hidden = self.conv1(hidden, edge_index)
#         hidden = F.tanh(hidden)
#         hidden = self.conv2(hidden, edge_index)
#         hidden = F.tanh(hidden)
#         hidden = self.conv3(hidden, edge_index)
#         hidden = F.tanh(hidden)

#         # Global Pooling (stack different aggregations)
#         hidden = torch.cat([gmp(hidden, batch_index),
#                             gap(hidden, batch_index)], dim=1)

#         # Apply a final (linear) classifier
#         out = self.out(hidden)

#         return out
    
# class PILLNet(nn.Module):
#     """
#     A CNN model with increased channels for better performance.

#     Args:
#         num_classes: An integer indicating the number of classes in the dataset.
#     """
#     def __init__(self, num_classes=10) -> None:
#         super(PILLNet, self).__init__()
#         self.feature_extractor = models.vgg16(weights='IMAGENET1K_V1').features[:-1]
#         self.classification_head = nn.Sequential(
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.AvgPool2d(kernel_size=(224 // 2 ** 5, 224 // 2 ** 5)),
#             nn.Flatten(),
#             nn.Linear(in_features=self.feature_extractor[-2].out_channels, out_features=num_classes),
#         )
#         for param in self.feature_extractor[:23].parameters():
#             param.requires_grad = False

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of the neural network
#         """
#         feature_maps = self.feature_extractor(x)
#         scores = self.classification_head(feature_maps)
#         return scores



class CNNClassifier(BaseModel):
    """Generic CNN for image classification."""
    
    def _build_model(self):
        # Default conv channels if not provided
        if self.config.conv_channels is None:
            self.config.conv_channels = [16, 32, 64]
        
        channels = self.config.conv_channels
        in_channels = self.config.input_shape[0]  # RGB=3, grayscale=1
        
        # Build convolutional layers
        conv_layers = []
        for i, out_channels in enumerate(channels):
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
        
        # Build classifier
        hidden_dims = self.config.hidden_dims or [128]
        classifier_layers = []
        
        in_features = flattened_size
        for hidden_dim in hidden_dims:
            classifier_layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.config.dropout_rate)
            ])
            in_features = hidden_dim
        
        classifier_layers.append(nn.Linear(in_features, self.config.num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MLPClassifier(BaseModel):
    """Generic MLP for sequence/tabular data."""
    
    def _build_model(self):
        input_size = self.config.input_shape[0]  # Flattened input size
        hidden_dims = self.config.hidden_dims or [64, 32, 16]
        
        layers = []
        in_features = input_size
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                self._get_activation(),
                nn.Dropout(self.config.dropout_rate)
            ])
            in_features = hidden_dim
        
        layers.append(nn.Linear(in_features, self.config.num_classes))
        self.network = nn.Sequential(*layers)
    
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
        return self.network(x)


class GCNClassifier(BaseModel):
    """Generic Graph Convolutional Network."""
    
    def _build_model(self):
        node_features = self.config.input_shape[0]  # Number of node features
        hidden_dims = self.config.hidden_dims or [64, 64, 64]
        
        # Build GCN layers
        self.initial_conv = GCNConv(node_features, hidden_dims[0])
        
        self.conv_layers = nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            self.conv_layers.append(GCNConv(hidden_dims[i-1], hidden_dims[i]))
        
        # Output layer
        self.out = nn.Linear(hidden_dims[-1] * 2, self.config.num_classes)  # *2 for pooling concat
    
    def forward(self, x, edge_index, batch_index):
        # Initial convolution
        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden)
        
        # Additional convolutions
        for conv_layer in self.conv_layers:
            hidden = conv_layer(hidden, edge_index)
            hidden = F.tanh(hidden)
        
        # Global pooling
        hidden = torch.cat([gmp(hidden, batch_index), gap(hidden, batch_index)], dim=1)
        
        # Final classification
        out = self.out(hidden)
        return out


class PretrainedCNN(BaseModel):
    """Generic pretrained CNN (VGG, ResNet, etc.)."""
    
    def _build_model(self):
        # Choose backbone
        backbone_name = getattr(self.config, 'backbone', 'vgg16')
        
        if backbone_name == 'vgg16':
            backbone = models.vgg16(weights='IMAGENET1K_V1')
            self.features = backbone.features[:-1]  # Remove last pooling
            feature_dim = self.features[-2].out_channels if hasattr(self.features[-2], 'out_channels') else 512
        elif backbone_name == 'resnet18':
            backbone = models.resnet18(weights='IMAGENET1K_V1')
            self.features = nn.Sequential(*list(backbone.children())[:-1])  # Remove FC layer
            feature_dim = backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Freeze layers if specified
        if self.config.freeze_layers > 0:
            for param in list(self.features.parameters())[:self.config.freeze_layers]:
                param.requires_grad = False
        
        # Build classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)) if backbone_name == 'vgg16' else nn.Identity(),
            nn.Flatten(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(feature_dim, self.config.num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)