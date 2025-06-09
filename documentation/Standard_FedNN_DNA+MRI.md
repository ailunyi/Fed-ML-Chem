# Standard Federated Neural Network for DNA+MRI Multimodal Dataset

## Overview
This notebook implements a federated learning approach for a multimodal dataset combining DNA sequences and MRI images using the Flower framework. It distributes the training across multiple clients, with each client training on a subset of the data, and then aggregates the model parameters. The model performs two classification tasks simultaneously, one for each modality.

## Model Architecture
- Multimodal architecture with separate networks for each data type:
  - **MRINet**: A CNN for processing MRI images
    - Convolutional layers with ReLU activations and max pooling
    - Fully connected layers that reduce to a shared expert vector
  - **DNANet**: A fully connected network for processing DNA sequences
    - 5 linear layers with ReLU activations
    - Final layer outputs to the shared expert vector
  - **Fusion Mechanism**:
    - MultiheadAttention for combining features from both modalities
    - Gating mechanism to weight the contribution of each modality
    - Separate output heads for MRI and DNA classification tasks

## Dataset
- Combined DNA+MRI dataset with:
  - MRI data with 4 classes ('glioma', 'meningioma', 'notumor', 'pituitary')
  - DNA data with 7 classes ('0', '1', '2', '3', '4', '5', '6')
- MRI images are resized to 224x224 pixels
- DNA sequences are processed using text/sequence processing techniques
- Uses a custom MultimodalDataset class to handle the two data types
- Data is split across 10 clients for federated learning
- Each client's data is further split between training and validation sets with a 90/10 ratio

## Training Approach
- Uses a federated learning approach with 10 clients
- Adam optimizer with learning rate of 1e-3
- Two separate CrossEntropyLoss functions (one for each modality)
- Each client trains for 10 epochs per round
- Runs for 20 rounds of federated learning
- Uses specialized training and evaluation functions for multimodal data
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
- `mri_n_qubits`: 4 (used for determining expert vector size)
- `dna_n_qubits`: 7 (used for determining expert vector size)
- `expert_vector`: 6 (shared representation size)
- `num_of_expert`: 2 (number of expert networks)

## Results and Visualization
The notebook generates and saves:
- Separate confusion matrices for MRI and DNA classification
- Separate ROC curves for MRI and DNA classification
- Training and validation accuracy/loss curves for each modality and each client
- Results are saved in the 'results/FL_DNA+MRI/' directory