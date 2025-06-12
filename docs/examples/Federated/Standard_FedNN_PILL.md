# Standard Federated Neural Network for PILL Dataset

## Overview
This notebook implements a federated learning approach for the PILL dataset using the Flower framework. It distributes the training across multiple clients, with each client training on a subset of the data, and then aggregates the model parameters.

## Model Architecture
- Uses a CNN model based on VGG16 pretrained on ImageNet
- The feature extractor is from VGG16 with the last layer removed
- A custom classification head is added with:
  - MaxPool2d
  - AvgPool2d
  - Flatten
  - Linear layer for classification
- The first 23 layers of the feature extractor are frozen (weights not updated during training)

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

## Results and Visualization
The notebook generates and saves:
- Confusion matrix for model evaluation
- ROC curves for performance analysis
- Training and validation accuracy/loss curves for each client
- Results are saved in the 'results/FL_PILL/' directory