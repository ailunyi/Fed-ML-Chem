"""
Fed-ML-Lib: A Complete Federated Learning Library
================================================

A comprehensive library for federated learning with support for:
- Classical and quantum neural networks
- Multiple data partitioning strategies
- Custom federated aggregation strategies
- Multimodal learning capabilities
- Comprehensive configuration management

Quick Start:
-----------
```python
from fed_ml_lib.config import run_experiment, pill_cnn, quantum_mri
from fed_ml_lib.models import create_model, create_quantum_model

# Simple experiment configuration
config = run_experiment(
    name="my_experiment",
    dataset="PILL",
    model="cnn",
    epochs=25,
    use_quantum=True
)

# Create models directly
model = create_model('cnn', input_shape=(3, 224, 224), num_classes=10)
quantum_model = create_quantum_model('cnn', input_shape=(3, 224, 224), 
                                   num_classes=10, n_qubits=4)

# See examples/python_config_example.py for complete examples
```
"""

# Core components
from .models import (
    create_model,
    create_modular_model,
    create_classical_model,
    create_fhe_model,
    create_quantum_model,
    create_fhe_quantum_model
)

from .datasets import (
    create_data_loaders,
    create_federated_data_loaders,
    get_dataset_info,
    MultimodalDataset,
    get_transforms
)

from .client import (
    FedMLClient,
    MultimodalFedMLClient,
    create_client
)

from .server import (
    FedCustom,
    weighted_average,
    create_server_evaluate_fn,
    get_on_fit_config_fn,
    get_on_evaluate_config_fn
)

from .engine import (
    run_central_training,
    train_one_epoch,
    evaluate_model
)

from .utils import (
    get_device,
    save_confusion_matrix,
    save_roc_curve,
    plot_training_curves,
    count_parameters,
    set_seed
)

from .config import (
    run_experiment,
    pill_cnn,
    dna_mlp, 
    mri_cnn,
    federated_pill,
    quantum_mri,
    fhe_dna,
    hybrid_pill
)

# Version info
__version__ = "1.0.0"
__author__ = "Fed-ML-Chem Team"
__description__ = "A complete federated learning library for classical and quantum machine learning"

# Define what gets imported with "from fed_ml_lib import *"
__all__ = [
    # Models
    "create_model",
    "create_modular_model",
    "create_classical_model",
    "create_fhe_model",
    "create_quantum_model",
    "create_fhe_quantum_model",
    
    # Datasets
    "create_data_loaders",
    "create_federated_data_loaders",
    "get_dataset_info",
    "MultimodalDataset",
    "get_transforms",
    
    # Client
    "FedMLClient",
    "MultimodalFedMLClient", 
    "create_client",
    
    # Server
    "FedCustom",
    "weighted_average",
    "create_server_evaluate_fn",
    "get_on_fit_config_fn",
    "get_on_evaluate_config_fn",
    
    # Engine
    "run_central_training",
    "train_one_epoch",
    "evaluate_model",
    
    # Utils
    "get_device",
    "save_confusion_matrix",
    "save_roc_curve", 
    "plot_training_curves",
    "count_parameters",
    "set_seed",
    
    # Config
    "run_experiment",
    "pill_cnn",
    "dna_mlp",
    "mri_cnn", 
    "federated_pill",
    "quantum_mri",
    "fhe_dna",
    "hybrid_pill",
] 