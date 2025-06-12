# Fed-ML-Lib: Complete Federated Learning Library

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-1.5+-green.svg)](https://flower.dev/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.28+-orange.svg)](https://pennylane.ai/)

A comprehensive federated learning library that supports both **classical** and **quantum** machine learning models with advanced federated strategies and multimodal capabilities.

## 🌟 Key Features

### 🤖 **Model Support**
- **Classical Models**: VGG16-based CNNs for image classification
- **Quantum Models**: PennyLane-based quantum neural networks
- **Hybrid Models**: Classical-quantum hybrid architectures
- **Multimodal Support**: Framework for multi-input models (images + sequences)

### 🌐 **Federated Learning**
- **Complete Flower Integration**: Full client-server implementation
- **Custom Strategies**: Advanced aggregation methods (FedCustom with dynamic learning rates)
- **Data Partitioning**: IID, non-IID, and Dirichlet distribution strategies
- **Server-side Evaluation**: Centralized model evaluation with visualization

### ⚙️ **Configuration Management**
- **YAML/JSON Configs**: Flexible experiment configuration
- **Predefined Templates**: Ready-to-use configurations for different scenarios
- **Type Safety**: Dataclass-based configuration with validation

### 📊 **Visualization & Analysis**
- **Training Curves**: Real-time training progress visualization
- **Confusion Matrices**: Per-client and server-side evaluation
- **ROC Curves**: Binary classification performance analysis
- **Federated Metrics**: Aggregated performance tracking

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install fed-ml-lib

# With GPU support
pip install fed-ml-lib[gpu]

# With quantum computing support
pip install fed-ml-lib[quantum]

# Complete installation (all features)
pip install fed-ml-lib[all]
```

### Simple Federated Learning Example

```python
import torch
from fed_ml_lib import create_model, create_federated_data_loaders, FedCustom
from fed_ml_lib.client import create_client
import flwr as fl

# 1. Create model
model = create_model("vgg16", num_classes=2)

# 2. Create federated data loaders
train_loaders, val_loaders, test_loader = create_federated_data_loaders(
    dataset_name="PILL",
    data_path="./data/",
    num_clients=5,
    partition_strategy="iid"
)

# 3. Define client function
def client_fn(cid: str):
    client_id = int(cid)
    model = create_model("vgg16", num_classes=2)
    return create_client(
        client_id=cid,
        model=model,
        train_loader=train_loaders[client_id],
        val_loader=val_loaders[client_id],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

# 4. Create strategy
strategy = FedCustom(
    fraction_fit=1.0,
    min_fit_clients=3,
    learning_rate_strategy="split"  # Different LRs for different clients
)

# 5. Run federated learning
history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=5,
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy
)
```

### Quantum Federated Learning

```python
from fed_ml_lib import create_model

# Create quantum model
quantum_model = create_model(
    model_type="hybrid_cnn_qnn",
    num_classes=2,
    n_qubits=4,
    n_layers=2
)

# Use smaller batches and images for quantum models
train_loaders, val_loaders, test_loader = create_federated_data_loaders(
    dataset_name="PILL",
    batch_size=8,  # Smaller batches
    resize=64,     # Smaller images
    num_clients=3
)
```

### Configuration-Based Experiments

```python
from fed_ml_lib.config import ExperimentConfig

# Load configuration
config = ExperimentConfig.from_yaml("experiments/quantum_federated.yaml")

# Or create programmatically
config = ExperimentConfig(
    model=ModelConfig(
        model_type="vgg16",
        model_params={}
    ),
    data=DataConfig(
        dataset_name="PILL",
        partition_strategy="non_iid"
    ),
    training=TrainingConfig(
        num_clients=5,
        num_rounds=10,
        learning_rate_strategy="adaptive"
    )
)

# Run experiment
from examples.federated_learning_example import run_federated_experiment
results = run_federated_experiment(config)
```

## 📚 Comprehensive Examples

### 1. **Classical Federated Learning**
```bash
python examples/central_training_example.py
python examples/federated_learning_example.py
```

### 2. **Quantum Federated Learning**
```bash
python examples/quantum_training_example.py
```

### 3. **Custom Strategies**
```python
from fed_ml_lib.server import FedCustom

strategy = FedCustom(
    learning_rate_strategy="split",  # Split clients into groups
    base_learning_rate=0.001,
    higher_learning_rate=0.003,
    fraction_fit=0.8,  # Sample 80% of clients per round
    min_fit_clients=3
)
```

### 4. **Data Partitioning Strategies**
```python
# IID partitioning (default)
train_loaders, _, _ = create_federated_data_loaders(
    dataset_name="PILL",
    partition_strategy="iid"
)

# Non-IID partitioning (limited classes per client)
train_loaders, _, _ = create_federated_data_loaders(
    dataset_name="PILL",
    partition_strategy="non_iid"
)

# Dirichlet partitioning (realistic non-IID)
train_loaders, _, _ = create_federated_data_loaders(
    dataset_name="PILL",
    partition_strategy="dirichlet"
)
```

## 🏗️ Architecture Overview

```
fed_ml_lib/
├── models.py          # Classical & quantum model definitions
├── datasets.py        # Data loading & federated partitioning
├── client.py          # Flower client implementations
├── server.py          # Custom federated strategies
├── engine.py          # Training & evaluation loops
├── utils.py           # Visualization & utility functions
├── config.py          # Configuration management
└── __init__.py        # Main library interface
```

## 🔧 Advanced Features

### Custom Federated Strategies

The library includes `FedCustom`, an advanced federated strategy with:

- **Dynamic Learning Rates**: Different learning rates for different client groups
- **Flexible Client Sampling**: Configurable client participation rates
- **Server-side Evaluation**: Centralized model evaluation with visualization
- **Failure Handling**: Robust handling of client failures

### Quantum Computing Integration

Full integration with PennyLane for quantum machine learning:

- **Quantum Circuits**: Parameterized quantum circuits with angle embedding
- **Hybrid Models**: Classical preprocessing + quantum processing
- **Variational Algorithms**: VQC (Variational Quantum Classifier) support
- **Hardware Compatibility**: Support for various quantum backends

### Multimodal Learning Framework

Extensible framework for multimodal federated learning:

- **Multiple Input Types**: Images, sequences, graphs, etc.
- **Attention Mechanisms**: Cross-modal attention for data fusion
- **Flexible Architecture**: Easy extension to new modalities

## 📊 Supported Datasets

- **PILL**: Pharmaceutical pill quality classification
- **DNA**: DNA sequence classification
- **Wafer**: Semiconductor wafer defect detection
- **HIV**: HIV drug activity prediction
- **CIFAR-10**: Standard computer vision benchmark
- **Custom Datasets**: Easy integration of new datasets

## 🎯 Performance Considerations

### Classical Models
- **Batch Size**: 16-32 for optimal performance
- **Image Size**: 224x224 for VGG16-based models
- **Clients**: 5-10 clients for simulation

### Quantum Models
- **Batch Size**: 4-8 (quantum circuits are computationally intensive)
- **Image Size**: 32x32 or 64x64 (smaller for faster quantum processing)
- **Qubits**: 4-8 qubits for practical simulation
- **Layers**: 2-4 variational layers

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/fed-ml-chem/fed-ml-lib.git
cd fed-ml-lib

# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/

# Format code
black fed_ml_lib/
isort fed_ml_lib/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Flower**: For the excellent federated learning framework
- **PennyLane**: For quantum computing capabilities
- **PyTorch**: For the deep learning foundation
- **PURDUE SURF Program**: For supporting this research

## 📞 Support

- **Documentation**: [https://fed-ml-lib.readthedocs.io](https://fed-ml-lib.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/fed-ml-chem/fed-ml-lib/issues)
- **Discussions**: [GitHub Discussions](https://github.com/fed-ml-chem/fed-ml-lib/discussions)

---

**Fed-ML-Lib** - Empowering federated learning research with classical and quantum machine learning capabilities. 🚀🔬 