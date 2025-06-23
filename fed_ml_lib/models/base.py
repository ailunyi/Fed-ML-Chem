import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

"""
This file contains the base model classes for the models.
"""

@dataclass
class ModelConfig:
    """Configuration for model architectures."""
    input_shape: Tuple[int, ...]
    num_classes: int
    model_type: str
    
    # Architecture parameters
    hidden_dims: Optional[List[int]] = None
    conv_channels: Optional[List[int]] = None
    dropout_rate: float = 0.0
    activation: str = "relu"
    
    # Quantum parameters  
    n_qubits: int = 4
    n_layers: int = 2
    
    # Domain-specific parameters
    domain: str = "vision"  # vision, sequence, graph, medical, molecular
    use_pretrained: bool = False
    freeze_layers: int = 0

class BaseModel(nn.Module, ABC):
    """Abstract base class for all models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self._build_model()
    
    @abstractmethod
    def _build_model(self):
        """Build the model architecture based on config."""
        pass
    
    def get_num_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)