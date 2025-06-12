# Standard Centralized Neural Network for PILL Dataset

## Overview
This notebook implements a centralized neural network approach for the PILL dataset. Although it uses the Flower federated learning framework, it's configured with only one client, effectively making it a centralized learning approach.

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
- Images are resized to 256x256 pixels
- Data is normalized using ImageNet normalization values
- Data is split between training and validation sets with a 90/10 ratio

## Training Approach
- Uses a centralized approach (1 client) with the Flower framework
- Adam optimizer with learning rate of 2e-4
- CrossEntropyLoss as the loss function
- Trains for 25 epochs
- Implements a custom federated learning strategy (FedCustom) that:
  - Aggregates model parameters using weighted averaging
  - Saves model checkpoints after each round
  - Evaluates the model on a test set

## Key Parameters
- `number_clients`: 1 (centralized approach)
- `max_epochs`: 25
- `batch_size`: 32
- `lr`: 2e-4
- `rounds`: 1 (only one round of federated learning)
- `frac_fit`: 1.0 (fraction of clients used for training)
- `frac_eval`: 1.0 (fraction of clients used for evaluation)

## Results and Visualization
The notebook generates and saves:
- Confusion matrix for model evaluation
- ROC curves for performance analysis
- Training and validation accuracy/loss curves
- Results are saved in the 'results/CL_PILL/' directory