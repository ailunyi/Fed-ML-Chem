"""Model architectures for Fed-ML-Lib."""

from .base import BaseModel, ModelConfig
from .modular import (
    ModularModel, ModularConfig, create_model
)

# Export everything from the modular system
__all__ = [
    # Base classes
    'BaseModel', 'ModelConfig',
    
    # Modular system - the main interface
    'ModularModel', 'ModularConfig', 'create_modular_model',
    
    # Convenience functions for specific combinations
    'create_classical_model', 'create_fhe_model',
    'create_quantum_model', 'create_fhe_quantum_model',
    
    # Main model creation function
    'create_model',
]