# Standard Federated Quantum Neural Network for HIV Dataset

## Overview
This notebook implements a federated quantum neural network approach for HIV molecular data classification. It combines a Graph Convolutional Network (GCN) with a quantum neural network layer using PennyLane, and distributes the training across multiple clients. The model is designed to classify HIV molecules as either active or inactive against HIV, leveraging both classical graph neural networks and quantum computing.

## Model Architecture
The model uses a hybrid classical-quantum architecture:
- **Classical Component**: Graph Convolutional Network (GCN)
  - Initial GCNConv layer that takes 9-dimensional node features as input
  - Three additional GCNConv layers with tanh activations
  - Global pooling that combines max and mean pooling
  - A final linear classifier that outputs features for the quantum layer

- **Quantum Component**:
  - 2 qubits and 2 layers quantum circuit
  - AngleEmbedding for encoding classical data into quantum states
  - BasicEntanglerLayers for quantum entanglement operations
  - PauliZ measurements for converting quantum states back to classical outputs

```python
n_qubits = 2
n_layers = 2
weight_shapes = {"weights": (n_layers, n_qubits)}
dev = qml.device("default.qubit", wires=n_qubits)
    
@qml.qnode(dev, interface='torch')
def quantum_net(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits)) 
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
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
- `n_qubits = 2`: Number of qubits in the quantum circuit
- `n_layers = 2`: Number of layers in the quantum circuit
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
- Results are stored in 'results/QFL_HIV/' directory
- Generates:
  - Training/validation loss and accuracy curves
  - Confusion matrices
  - ROC curves