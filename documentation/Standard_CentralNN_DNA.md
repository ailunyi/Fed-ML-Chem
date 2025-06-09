# Standard Centralized Neural Network for DNA Dataset

## Overview
This notebook implements a centralized neural network approach for the DNA dataset.

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
- Data is split between training and validation sets with a 90/10 ratio

## Training Approach
- Uses a centralized approach (1 client) with the Flower framework
- Adam optimizer with learning rate of 1e-3
- CrossEntropyLoss as the loss function
- Trains for 25 epochs
- Implements a custom federated learning strategy (FedCustom) that:
  - Aggregates model parameters using weighted averaging
  - Saves model checkpoints after each round
  - Evaluates the model on a test set

## Key Parameters
- `number_clients`: 1 (centralized approach)
- `max_epochs`: 25
- `batch_size`: 16
- `lr`: 1e-3
- `rounds`: 1 (only one round of federated learning)
- `frac_fit`: 1.0 (fraction of clients used for training)
- `frac_eval`: 1.0 (fraction of clients used for evaluation)

## Results and Visualization
The notebook generates and saves:
- Confusion matrix for model evaluation
- ROC curves for performance analysis
- Training and validation accuracy/loss curves
- Results are saved in the 'results/CL_DNA/' directory