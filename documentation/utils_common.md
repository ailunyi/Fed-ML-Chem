# utils/common.py

## Overview
This utility file provides common functions used across the various notebooks in the project. It includes functionality for file operations, device selection, dataset utilities, visualization functions, and model parameter handling. These functions serve as the foundation for the different neural network approaches (centralized, federated, and quantum federated) implemented in the project.

## Key Components

### File Operations
- `write_yaml` and `read_yaml`: Functions for writing and reading YAML configuration files
- `supp_ds_store`: Function to delete hidden macOS .DS_Store files
- `create_files_train_test`: Function to split datasets into training and testing sets

### Device Selection
- `choice_device`: Function to select the appropriate device (CPU, GPU, or MPS on Mac) for model training and inference

### Dataset Utilities
- `classes_string`: Function that returns class labels for different datasets (DNA, PILL, HIV, Wafer, etc.)
- Dataset-specific normalization values for different datasets (MNIST, CIFAR, DNA, PILL, HIV, Wafer, etc.)

### Visualization Functions
- `save_matrix`: Function to save confusion matrices as images
- `save_roc`: Function to save ROC curves as images
- `save_graphs` and `save_graphs_multimodal`: Functions to save training/validation curves
- `plot_graph`: General function for plotting graphs

### Model Parameter Handling
- `get_parameters2`: Function to extract model parameters from PyTorch models
- `set_parameters`: Function to update model parameters in PyTorch models

## Usage
This utility file is imported in all the notebooks in the project and provides the common functionality needed for data handling, model training, evaluation, and visualization. The functions are designed to be reusable across different neural network approaches and datasets.

Example usage:
```python
from utils import common

# Select device
device = common.choice_device('gpu')

# Get class labels for a dataset
classes = common.classes_string('DNA')

# Extract model parameters
parameters = common.get_parameters2(model)

# Update model parameters
common.set_parameters(model, parameters)

# Save visualization
common.save_matrix(y_true, y_pred, 'confusion_matrix.png', classes)
common.save_roc(y_true, y_proba, 'roc_curve.png', len(classes))
common.save_graphs('results/', epochs, results)
```

## Dependencies
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- PyYAML