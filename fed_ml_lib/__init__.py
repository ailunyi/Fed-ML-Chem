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
from fed_ml_lib import create_config, run_centralized_simulation, run_federated_simulation
from fed_ml_lib.models import create_model

# Simple experiment configuration
config = create_config(
    name="my_experiment",
    dataset="PILL",
    model="cnn",
    epochs=25,
    use_quantum=True
)

# Run complete simulations
results = run_centralized_simulation(config=config, model_params={'base_architecture': 'cnn'})

# Or federated learning
run_federated_simulation(config=config, model_params={'base_architecture': 'cnn'}, 
                        num_clients=3, num_rounds=5)

# Create models directly  
model = create_model('cnn', input_shape=(3, 224, 224), num_classes=10)

# See examples/ for complete examples
```
"""

# Core components
from .models import (
    create_model
)

from .data import (
    MultimodalDataset,
    infer_dataset_properties,
)

from .federated.client import (
    FlowerClient as FedMLClient,
    MultimodalFlowerClient as MultimodalFedMLClient,
    FHEFlowerClient as FHEFedMLClient
)

from .federated.server import (
    FedCustom
)

from .federated.utils import (
    weighted_average,
    get_on_fit_config_fn,
    create_client_fn,
    create_evaluate_fn,
    run_federated_simulation
)

from .federated.strategies import (
    create_fedavg_strategy
)

from .centralized.utils import (
    run_centralized_simulation
)

from .core.training import (
    train as run_central_training,
    train_step as train_one_epoch
)

from .core.testing import (
    test as evaluate_model
)

from .core.utils import (
    choice_device as get_device,
    get_parameters2 as get_model_parameters,
    set_parameters as set_model_parameters
)

from .core.visualization import (
    save_matrix as save_confusion_matrix,
    save_roc as save_roc_curve,
    plot_graph as plot_training_curves,
    save_all_results
)

from .config import (
    create_config,
)

# Version info
__version__ = "1.0.0"
__author__ = "Fed-ML-Chem Team"
__description__ = "A complete federated learning library for classical and quantum machine learning"

# Define what gets imported with "from fed_ml_lib import *"
__all__ = [
    # Models
    "create_model",
    
    # Datasets
    "MultimodalDataset",
    "infer_dataset_properties",
    
    # Client
    "FedMLClient",
    "MultimodalFedMLClient", 
    "FHEFedMLClient",
    
    # Server
    "FedCustom",
    "weighted_average",
    "get_on_fit_config_fn",
    "create_fedavg_strategy",
    "create_client_fn",
    "create_evaluate_fn",
    "run_federated_simulation",
    "run_centralized_simulation",
    
    # Engine
    "run_central_training",
    "train_one_epoch",
    "evaluate_model",
    
    # Utils
    "get_device",
    "get_model_parameters",
    "set_model_parameters",
    "save_confusion_matrix",
    "save_roc_curve", 
    "plot_training_curves",
    "save_all_results",
    
    # Config
    "create_config",
] 