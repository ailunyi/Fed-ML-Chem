"""Model architectures for Fed-ML-Lib."""

from .base import BaseModel, ModelConfig
from .modular import (
    ModularModel, ModularConfig, create_modular_model,
    create_classical_model, create_fhe_model,
    create_quantum_model, create_fhe_quantum_model
)

# Simplified model creation using the modular system
def create_model(model_type: str, **kwargs):
    """
    Create a model using the modular system.
    
    Args:
        model_type: Type of model ('cnn', 'mlp', 'gcn', 'pretrained_cnn')
        **kwargs: Additional configuration parameters
    
    Returns:
        ModularModel instance
    
    Examples:
        # Create a classical CNN
        model = create_model('cnn', input_shape=(3, 224, 224), num_classes=10)
        
        # Create a quantum-enhanced MLP
        model = create_model('mlp', input_shape=(180,), num_classes=7, 
                           use_quantum=True, n_qubits=4)
        
        # Create an FHE-encrypted CNN
        model = create_model('cnn', input_shape=(3, 32, 32), num_classes=10,
                           use_fhe=True, fhe_layers=['classifier'])
    """
    return create_modular_model(
        base_architecture=model_type,
        **kwargs
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