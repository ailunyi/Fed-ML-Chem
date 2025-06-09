# Standard Federated Neural Network for DNA Dataset

## Overview
This notebook implements a federated learning approach for the DNA dataset using the Flower framework. It distributes the training across multiple clients, with each client training on a subset of the data, and then aggregates the model parameters.

## Model Architecture
- Uses a simple fully connected neural network (MLP)
- 5 linear layers with ReLU activations:
  - fc1: input_sp → 64 neurons
  - fc2: 64 → 32 neurons
  - fc3: 32 → 16 neurons
  - fc4: 16 → 8 neurons
  - fc5: 8 → num_classes (output layer)
- The input size (input_sp) is determined dynamically from the data

## Dataset
- DNA dataset with 7 classes ('0', '1', '2', '3', '4', '5', '6')
- Uses DNA sequence data rather than images
- Data is processed using text/sequence processing techniques
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
- `batch_size`: 16
- `lr`: 1e-3 (standard learning rate)
- `rounds`: 20 (rounds of federated learning)
- `frac_fit`: 1.0 (fraction of clients used for training)
- `frac_eval`: 0.5 (fraction of clients used for evaluation)

## Results and Visualization
The notebook generates and saves:
- Confusion matrix for model evaluation
- ROC curves for performance analysis
- Training and validation accuracy/loss curves for each client
- Results are saved in the 'results/FL_DNA/' directory