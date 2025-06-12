# utils/engine.py

## Overview
This utility file provides functions for training and testing PyTorch models. It includes specialized functions for different model types, including standard models, graph models, and multimodal models. These functions form the core training and evaluation engine for all the neural network approaches (centralized, federated, and quantum federated) implemented in the project.

## Key Components

### Core Training and Testing Functions
- `train_step`: Function to train a model for a single epoch
  - Handles batching, forward and backward passes, optimization, and metric calculation
  - Returns training loss and accuracy
- `test`: Function to evaluate a model on a test dataset
  - Performs forward passes without gradient calculation
  - Calculates loss, accuracy, and generates predictions and probabilities
  - Returns evaluation metrics and predictions for visualization

### Specialized Functions for Different Model Types
- **Graph Models**:
  - `train_step_graph`: Specialized training function for graph neural networks
  - `test_graph`: Specialized testing function for graph neural networks
  - Handles the unique input format of graph data (node features, edge indices, batch indices)

- **Multimodal Models**:
  - `train_step_multimodal`: Training function for models that process multiple data types
  - `test_multimodal`: Testing function for multimodal models
  - Handles separate inputs and outputs for different modalities

- **Multimodal Health Models**:
  - `train_step_multimodal_health`: Specialized training function for health-related multimodal data
  - `test_multimodal_health`: Specialized testing function for health-related multimodal data
  - Includes additional metrics and processing specific to health applications

### Main Training Function
- `train`: Main function that orchestrates the training and evaluation process
  - Takes a model, training and validation data, optimizer, loss function, and other parameters
  - Selects the appropriate training and testing functions based on the task type
  - Runs the training loop for the specified number of epochs
  - Tracks and returns metrics (loss and accuracy) for both training and validation
  - Supports early stopping and learning rate scheduling

## Usage
This utility file is imported in all the notebooks in the project and provides the training and evaluation functionality needed for the different neural network models. The functions are designed to be flexible and support various model types and learning approaches.

Example usage:
```python
from utils import engine

# Train a standard model
results = engine.train(
    model=net,
    trainloader=trainloader,
    valloader=valloader,
    optimizer=optimizer,
    loss_fn=criterion,
    epochs=25,
    device=device
)

# Test a model
loss, accuracy, y_pred, y_true, y_proba = engine.test(
    model=net,
    testloader=testloader,
    loss_fn=criterion,
    device=device
)
```

## Dependencies
- PyTorch
- NumPy
- tqdm (for progress bars)