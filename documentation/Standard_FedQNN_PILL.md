# Standard Federated Quantum Neural Network for PILL Dataset

## Overview
This notebook implements a federated quantum neural network approach for the PILL dataset using the Flower framework and PennyLane quantum computing library. It combines federated learning with quantum computing, distributing the training across multiple clients and using quantum circuits as part of the model architecture.

## Model Architecture
- Hybrid classical-quantum architecture
- Classical component:
  - Uses VGG16 pretrained on ImageNet as a feature extractor
  - Custom classification head with MaxPool2d, AvgPool2d, Flatten, and Linear layers
  - The first 23 layers of the feature extractor are frozen
- Quantum component:
  - Uses PennyLane for quantum computing
  - 2 qubits quantum system
  - 2 layers of quantum gates
  - AngleEmbedding for encoding classical data into quantum states
  - BasicEntanglerLayers for entangling qubits
  - Measurement using PauliZ expectation values
  - Integrated with PyTorch using qml.qnn.TorchLayer

## Dataset
- PILL dataset with binary classification ('bad', 'good')
- Images are resized to 224x224 pixels
- Data is normalized using ImageNet normalization values
- Data is split across 10 clients for federated learning
- Each client's data is further split between training and validation sets with a 90/10 ratio

## Training Approach
- Uses a federated learning approach with 10 clients
- Adam optimizer with learning rate of 1e-3
- CrossEntropyLoss as the loss function
- Each client trains for 10 epochs per round
- Runs for 20 rounds of federated learning
- Implements a custom federated learning strategy (FedCustom) that:
  - Aggregates model parameters using weighted averaging
  - Saves model checkpoints after each round
  - Evaluates the model on a test set
  - Uses different learning rates for different clients (standard lr for first half, higher lr for second half)

## Key Parameters
- `number_clients`: 10 (true federated approach)
- `max_epochs`: 10 (epochs per round)
- `batch_size`: 32
- `lr`: 1e-3 (standard learning rate)
- `rounds`: 20 (rounds of federated learning)
- `frac_fit`: 1.0 (fraction of clients used for training)
- `frac_eval`: 0.5 (fraction of clients used for evaluation)
- `n_qubits`: 2 (number of qubits in the quantum circuit)
- `n_layers`: 2 (number of layers in the quantum circuit)

## Results and Visualization
The notebook generates and saves:
- Confusion matrix for model evaluation
- ROC curves for performance analysis
- Training and validation accuracy/loss curves for each client
- Results are saved in the 'results/QFL_PILL/' directory