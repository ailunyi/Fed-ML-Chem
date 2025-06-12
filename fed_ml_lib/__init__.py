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
from fed_ml_lib import create_model, create_federated_data_loaders, FedCustom
from fed_ml_lib.client import create_client
from fed_ml_lib.config import ExperimentConfig

# Create model
model = create_model("vgg16", num_classes=10)

# Create federated data
train_loaders, val_loaders, test_loader = create_federated_data_loaders(
    dataset_name="PILL", num_clients=5
)

# Run federated learning
# See examples/federated_learning_example.py for complete example
```
"""

# Core components
from .models import (
    VGG16Classifier,
    QuantumNet,
    HybridCNN_QNN,
    VariationalQuantumClassifier,
    create_model
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
    ModelConfig,
    DataConfig,
    TrainingConfig,
    ExperimentConfig,
    ConfigManager,
    load_experiment_config,
    create_custom_config
)

# Version info
__version__ = "1.0.0"
__author__ = "Fed-ML-Chem Team"
__description__ = "A complete federated learning library for classical and quantum machine learning"

# Define what gets imported with "from fed_ml_lib import *"
__all__ = [
    # Models
    "VGG16Classifier",
    "QuantumNet", 
    "HybridCNN_QNN",
    "VariationalQuantumClassifier",
    "create_model",
    
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
    "ModelConfig",
    "DataConfig",
    "TrainingConfig", 
    "ExperimentConfig",
    "ConfigManager",
    "load_experiment_config",
    "create_custom_config",
] 