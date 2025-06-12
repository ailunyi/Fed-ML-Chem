# utils/data_setup.py

## Overview
This utility file provides functionality for creating PyTorch DataLoaders for various datasets used in the project. It handles data preprocessing, normalization, splitting, and loading for different types of data, including images, DNA sequences, molecular graphs, and multimodal data. These functions are essential for preparing the datasets for the different neural network approaches (centralized, federated, and quantum federated) implemented in the project.

## Key Components

### Dataset Normalization
- Defines normalization values (mean and standard deviation) for different datasets:
  - MNIST, CIFAR, DNA, PILL, HIV, Wafer, and others
  - These values are used to standardize the input data for better model training

### Custom Dataset Classes
- `MultimodalDataset`: A custom PyTorch Dataset class for handling datasets with multiple modalities (like DNA+MRI)
  - Supports loading and processing data from different sources simultaneously
  - Enables training models on combined data types (e.g., DNA sequences and MRI images)

### Data Preprocessing Functions
- `read_and_prepare_data`: Function for processing DNA sequence data
  - Converts DNA sequences into numerical representations suitable for neural networks
- `preprocess_graph`: Function for processing HIV dataset from MoleculeNet
  - Converts molecular structures into graph representations for graph neural networks
- `preprocess_and_split_data`: Function for processing audio/visual data
  - Handles data augmentation, normalization, and splitting

### Data Splitting and Loading
- `split_data_client`: Function to split datasets across multiple clients for federated learning
  - Ensures each client has a balanced subset of the data
  - Supports different splitting strategies based on the dataset type
- `load_datasets`: Main function to load and prepare datasets for training and testing
  - Handles different dataset types (images, sequences, graphs, multimodal)
  - Creates appropriate DataLoaders with the specified batch size, normalization, etc.
  - Supports both centralized and federated learning approaches

## Usage
This utility file is imported in all the notebooks in the project and provides the data handling functionality needed for training and evaluating the different neural network models. The functions are designed to be flexible and support various dataset types and learning approaches.

Example usage:
```python
from utils import data_setup

# Load datasets for federated learning
trainloaders, valloaders, testloader = data_setup.load_datasets(
    num_clients=10,
    batch_size=64,
    resize=224,
    seed=0,
    num_workers=0,
    splitter=10,
    dataset='PILL',
    data_path='data/',
    data_path_val=None
)
```

## Dependencies
- PyTorch
- torchvision
- torch_geometric (for graph data)
- NumPy
- scikit-learn
- PIL (Python Imaging Library)