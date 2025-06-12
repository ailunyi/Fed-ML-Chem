# Fed ML Library

A Python library for federated learning and centralized machine learning with support for quantum neural networks.

## Features

- **Centralized Training**: Train models on a single machine
- **Federated Learning**: Distributed training across multiple clients (coming soon)
- **Quantum Neural Networks**: Support for quantum models using PennyLane
- **Hybrid Models**: Combine classical and quantum components
- **Multiple Datasets**: Easy integration with various datasets
- **Visualization**: Automatic plotting of training curves, confusion matrices, and ROC curves

## Installation

### From Source

1. Clone the repository:
```bash
git clone <repository-url>
cd Fed-ML-Chem
```

2. Install dependencies:
```bash
# For CPU version
pip install .[cpu]

# For GPU version  
pip install .[gpu]

# For all dependencies
pip install .[all]
```

## Quick Start

### Centralized Training (Classical)

```python
import torch
import torch.nn as nn
import torch.optim as optim

from fed_ml_lib.models import VGG16Classifier
from fed_ml_lib.engine import run_central_training
from fed_ml_lib.datasets import create_data_loaders, get_dataset_info

# Configuration
dataset_name = 'PILL'  # or 'Wafer', 'CIFAR', etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get dataset info and create data loaders
dataset_info = get_dataset_info(dataset_name, './data/')
train_loader, val_loader, test_loader = create_data_loaders(
    dataset_name=dataset_name,
    data_path='./data/',
    batch_size=32,
    resize=224,
    val_split=0.1,
    seed=42
)

# Create model, optimizer, and loss function
model = VGG16Classifier(num_classes=dataset_info['num_classes'])
optimizer = optim.Adam(model.parameters(), lr=2e-4)
loss_fn = nn.CrossEntropyLoss()

# Train the model
results = run_central_training(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device,
    epochs=25,
    results_dir='./results/',
    plot_results=True
)
```

### Quantum Training

```python
from fed_ml_lib.models import HybridCNN_QNN, QuantumNet, create_model

# Option 1: Hybrid CNN-Quantum model
model = HybridCNN_QNN(
    num_classes=dataset_info['num_classes'],
    n_qubits=4,
    n_layers=2
)

# Option 2: Pure quantum model
model = QuantumNet(
    input_size=784,  # For flattened 28x28 images
    num_classes=dataset_info['num_classes'],
    n_qubits=6,
    n_layers=3
)

# Option 3: Using factory function
model = create_model(
    model_type='hybrid_cnn_qnn',
    num_classes=dataset_info['num_classes'],
    n_qubits=4,
    n_layers=2
)

# Train as usual
optimizer = optim.Adam(model.parameters(), lr=1e-3)
results = run_central_training(model, train_loader, val_loader, optimizer, loss_fn, device)
```

## Library Structure

```
fed_ml_lib/
├── __init__.py          # Package initialization
├── models.py            # Model architectures (Classical, Quantum, Hybrid)
├── engine.py            # Training and evaluation loops
├── datasets.py          # Dataset loading and preprocessing
├── utils.py             # Utility functions (plotting, metrics)
├── client.py            # Federated learning client logic
├── server.py            # Federated learning server logic
└── config.py            # Configuration management
```

## Supported Datasets

The library supports various dataset formats:

- **Image Datasets**: PILL, Wafer, and any dataset with Training/Testing folder structure
- **Standard Datasets**: CIFAR-10 (automatically downloaded)
- **Custom Datasets**: Easy integration with your own datasets

### Dataset Structure

For custom image datasets, organize your data as follows:

```
data/
└── YourDataset/
    ├── Training/
    │   ├── class1/
    │   ├── class2/
    │   └── ...
    └── Testing/
        ├── class1/
        ├── class2/
        └── ...
```

## Models

### Classical Models

#### VGG16Classifier

A CNN model based on VGG16 pretrained on ImageNet:

```python
from fed_ml_lib.models import VGG16Classifier

model = VGG16Classifier(num_classes=2)  # Binary classification
model = VGG16Classifier(num_classes=10) # Multi-class classification
```

Features:
- Pretrained VGG16 feature extractor
- Custom classification head
- Frozen early layers for transfer learning

### Quantum Models

#### HybridCNN_QNN

A hybrid CNN-Quantum neural network that combines classical convolutional layers with quantum processing:

```python
from fed_ml_lib.models import HybridCNN_QNN

model = HybridCNN_QNN(
    num_classes=2,
    n_qubits=4,      # Number of qubits in quantum circuit
    n_layers=2       # Number of variational layers
)
```

Features:
- CNN feature extraction
- Quantum variational layers
- Suitable for image classification

#### QuantumNet

A hybrid model with classical preprocessing and quantum processing:

```python
from fed_ml_lib.models import QuantumNet

model = QuantumNet(
    input_size=784,   # Input feature size
    num_classes=10,
    n_qubits=6,
    n_layers=3
)
```

Features:
- Classical preprocessing layers
- Quantum circuit with angle embedding
- Classical post-processing

#### VariationalQuantumClassifier

A pure variational quantum classifier:

```python
from fed_ml_lib.models import VariationalQuantumClassifier

model = VariationalQuantumClassifier(
    input_size=4,
    num_classes=2,
    n_qubits=4,
    n_layers=3
)
```

Features:
- Pure quantum processing
- Variational quantum circuits
- Parameterized quantum gates

### Model Factory

Use the factory function for easy model creation:

```python
from fed_ml_lib.models import create_model

# Classical model
model = create_model('vgg16', num_classes=10)

# Quantum models
model = create_model('hybrid_cnn_qnn', num_classes=2, n_qubits=4, n_layers=2)
model = create_model('quantum', num_classes=10, input_size=784, n_qubits=6)
model = create_model('vqc', num_classes=2, input_size=4, n_qubits=4)
```

## Examples

### Running Example Scripts

```bash
# Classical training
cd examples
python central_training_example.py

# Quantum training
python quantum_training_example.py
```

### Customizing for Your Dataset

1. Change the `dataset_name` in the example script
2. Ensure your dataset follows the expected folder structure
3. Adjust hyperparameters as needed

```python
config = {
    'dataset_name': 'YourDataset',  # Change this
    'data_path': './data/',
    'batch_size': 32,              # Use smaller batches for quantum models
    'resize': 224,                 # Use smaller images for quantum models
    'epochs': 25,
    'learning_rate': 2e-4,         # Use smaller LR for quantum models
    'model_type': 'hybrid_cnn_qnn', # Choose model type
    'n_qubits': 4,                 # Quantum parameters
    'n_layers': 2,
}
```

## Quantum Computing Notes

### Performance Considerations

- **Quantum models are slower**: Due to quantum circuit simulation
- **Use smaller batch sizes**: Recommended 8-16 for quantum models
- **Use smaller images**: 32x32 or 64x64 for quantum processing
- **Fewer epochs**: Quantum models may converge faster

### Hardware Requirements

- **CPU**: Works on any CPU (quantum simulation)
- **GPU**: Limited benefit for quantum circuits (classical parts only)
- **Memory**: Quantum simulation can be memory-intensive

### Quantum Circuit Design

- **Angle Embedding**: Encodes classical data into quantum states
- **Variational Layers**: Parameterized quantum gates for learning
- **Entanglement**: Creates quantum correlations between qubits
- **Measurements**: Extract classical information from quantum states

## API Reference

### Engine Functions

- `run_central_training()`: Main function for centralized training
- `train()`: Core training loop
- `test()`: Evaluation function

### Dataset Functions

- `create_data_loaders()`: Create train/val/test data loaders
- `get_dataset_info()`: Get dataset metadata
- `get_transforms()`: Get appropriate data transforms

### Model Functions

- `create_model()`: Factory function for model creation
- `VGG16Classifier()`: Classical CNN model
- `HybridCNN_QNN()`: Hybrid CNN-Quantum model
- `QuantumNet()`: Hybrid classical-quantum model
- `VariationalQuantumClassifier()`: Pure quantum model

### Utility Functions

- `save_graphs()`: Save training curves
- `save_matrix()`: Save confusion matrix
- `save_roc()`: Save ROC curves
- `get_parameters()`: Extract model parameters
- `set_parameters()`: Set model parameters

## Results and Visualization

The library automatically generates:

- **Training Curves**: Loss and accuracy over epochs
- **Confusion Matrix**: Classification performance visualization
- **ROC Curves**: For binary classification tasks

Results are saved to the specified `results_dir`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Citation

If you use this library in your research, please cite:

```bibtex
[Add citation information here]
``` 