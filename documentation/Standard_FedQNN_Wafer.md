# Standard Federated Quantum Neural Network for Wafer Dataset

## Overview
This notebook implements a federated quantum neural network approach for semiconductor wafer classification. It combines a Convolutional Neural Network (CNN) with a quantum neural network layer using PennyLane, and distributes the training across multiple clients. The model is designed to classify wafer images into different categories based on their manufacturing quality or characteristics, leveraging both classical deep learning and quantum computing.

## Model Architecture
The model uses a hybrid classical-quantum architecture:
- **Classical Component**: Standard CNN architecture
  - **Features Component**:
    - First convolutional layer: 3 input channels → 16 output channels, 3×3 kernel, padding=1
    - ReLU activation and 2×2 max pooling
    - Second convolutional layer: 16 input channels → 32 output channels, 3×3 kernel, padding=1
    - ReLU activation and 2×2 max pooling
  - **Classifier Component**:
    - Flattening layer to convert 3D feature maps to 1D vector
    - First fully connected layer: 32 * 16 * 16 → 128 neurons
    - ReLU activation
    - Output layer: 128 → num_classes neurons

- **Quantum Component**:
  - 9 qubits and 9 layers quantum circuit
  - AngleEmbedding for encoding classical data into quantum states
  - BasicEntanglerLayers for quantum entanglement operations
  - PauliZ measurements for converting quantum states back to classical outputs

```python
n_qubits = 9
n_layers = 9
weight_shapes = {"weights": (n_layers, n_qubits)}

dev = qml.device("default.qubit", wires=n_qubits)
    
@qml.qnode(dev, interface='torch')
def quantum_net(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits)) 
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
```

## Dataset
- Wafer dataset containing semiconductor manufacturing wafer images
- Images are resized to 64×64 pixels
- The dataset is split across 10 clients for federated learning
- Uses standard data augmentation techniques for training

## Training Approach
- Federated learning with 10 clients
- Custom FedCustom strategy that extends Flower's base Strategy class
- Differential learning rates: half the clients use a standard learning rate (1e-3) and the other half use a higher learning rate (0.003)
- Each client trains for 10 epochs per round
- Training runs for 20 rounds of federated learning
- Standard CrossEntropyLoss function
- Adam optimizer with learning rate of 1e-3
- Uses standard training and evaluation functions:
  - engine.train for training
  - engine.test for evaluation

## Key Parameters
- `n_qubits = 9`: Number of qubits in the quantum circuit
- `n_layers = 9`: Number of layers in the quantum circuit
- `number_clients = 10`: Number of federated learning clients
- `rounds = 20`: Number of federated learning rounds
- `max_epochs = 10`: Number of local epochs per round
- `batch_size = 64`: Batch size for training and evaluation
- `resize = 64`: Image resize dimensions
- `frac_fit = 1.0`: Fraction of clients used for training
- `frac_eval = 0.5`: Fraction of clients used for evaluation
- `lr = 1e-3`: Base learning rate (some clients use 0.003)

## Results and Visualization
- Saves confusion matrices and ROC curves for each client
- Aggregates and saves global model checkpoints
- Results are stored in 'results/QFL_Wafer/' directory
- Generates:
  - Training/validation loss and accuracy curves
  - Confusion matrices
  - ROC curves