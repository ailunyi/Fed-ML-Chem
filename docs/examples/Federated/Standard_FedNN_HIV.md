# Standard Federated Neural Network for HIV Dataset

## Overview
This notebook implements a federated learning approach for HIV molecular data classification using a Graph Convolutional Network (GCN). It distributes the training across multiple clients, each with their own subset of the data, and aggregates the model updates using a custom federated learning strategy. The model is designed to classify HIV molecules as either active or inactive against HIV.

## Model Architecture
The model uses a Graph Convolutional Network (GCN) architecture:
- Initial GCNConv layer that takes 9-dimensional node features as input
- Three additional GCNConv layers with tanh activations
- Global pooling that combines max and mean pooling
- A final linear classifier that outputs predictions for binary classification

```python
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.initial_conv = GCNConv(9, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)
        self.out = nn.Linear(embedding_size * 2, num_classes)
```

## Dataset
- HIV dataset with binary classification: 'confirmed inactive (CI)' and 'confirmed active (CA)/confirmed moderately active (CM)'
- Data is represented as molecular graphs using PyTorch Geometric
- The dataset is split across 10 clients for federated learning

## Training Approach
- Federated learning with 10 clients
- Custom FedCustom strategy that extends Flower's base Strategy class
- Differential learning rates: half the clients use a standard learning rate (1e-3) and the other half use a higher learning rate (0.003)
- Each client trains for 10 epochs per round
- Training runs for 20 rounds of federated learning
- Uses a custom RMSELoss (Root Mean Square Error) loss function
- Adam optimizer with learning rate of 1e-3
- Specialized training and evaluation functions for graph data:
  - engine.train with task="Graph"
  - engine.test_graph

## Key Parameters
- `number_clients = 10`: Number of federated learning clients
- `rounds = 20`: Number of federated learning rounds
- `max_epochs = 10`: Number of local epochs per round
- `embedding_size = batch_size = 64`: Both set to the same value
- `frac_fit = 1.0`: Fraction of clients used for training
- `frac_eval = 0.5`: Fraction of clients used for evaluation
- `lr = 1e-3`: Base learning rate (some clients use 0.003)

## Results and Visualization
- Saves confusion matrices and ROC curves for each client
- Aggregates and saves global model checkpoints
- Results are stored in 'results/FL_HIV/' directory
- Generates:
  - Training/validation loss and accuracy curves
  - Confusion matrices
  - ROC curves