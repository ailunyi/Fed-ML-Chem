# Standard Centralized Neural Network for HIV Dataset

## Overview
This notebook implements a centralized neural network approach for the HIV dataset using graph neural networks. Although it uses the Flower federated learning framework, it's configured with only one client, effectively making it a centralized learning approach. The model processes molecular graph data to classify HIV inhibition activity.

## Model Architecture
- Uses a Graph Convolutional Network (GCN) architecture
- Network structure:
  - Initial GCNConv layer: 9 → embedding_size features
  - Three additional GCNConv layers with tanh activations
  - Global pooling (combining max and mean pooling)
  - Final linear classifier: embedding_size*2 → num_classes
- The model processes molecular graphs where:
  - Nodes represent atoms
  - Edges represent bonds
  - Node features include atomic properties
  - The graph structure captures molecular topology

## Dataset
- HIV dataset with binary classification:
  - 'confirmed inactive (CI)'
  - 'confirmed active (CA)/confirmed moderately active (CM)'
- Uses molecular graph data from the MoleculeNet collection
- Data is processed using PyTorch Geometric
- Data is split between training and validation sets with a 90/10 ratio

## Training Approach
- Uses a centralized approach (1 client) with the Flower framework
- Custom RMSELoss (Root Mean Square Error) as the loss function
- Adam optimizer with learning rate of 1e-3
- Trains for 25 epochs
- Uses specialized training and evaluation functions for graph data:
  - engine.train with task="Graph"
  - engine.test_graph
- Implements a custom federated learning strategy (FedCustom) that:
  - Aggregates model parameters using weighted averaging
  - Saves model checkpoints after each round
  - Evaluates the model on a test set

## Key Parameters
- `number_clients`: 1 (centralized approach)
- `max_epochs`: 25
- `embedding_size`: 64 (dimension of node embeddings)
- `batch_size`: 64
- `lr`: 1e-3
- `rounds`: 1 (only one round of federated learning)
- `frac_fit`: 1.0 (fraction of clients used for training)
- `frac_eval`: 0.5 (fraction of clients used for evaluation)

## Results and Visualization
The notebook generates and saves:
- Confusion matrix for model evaluation
- ROC curves for performance analysis
- Training and validation accuracy/loss curves
- Results are saved in the 'results/CL_HIV/' directory