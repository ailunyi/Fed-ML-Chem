# Standard Federated Neural Network for Wafer Dataset

## Overview
This notebook implements a federated learning approach for semiconductor wafer classification using a Convolutional Neural Network (CNN). It distributes the training across multiple clients, each with their own subset of the data, and aggregates the model updates using a custom federated learning strategy. The model is designed to classify wafer images into different categories based on their manufacturing quality or characteristics.

## Model Architecture
The model uses a standard CNN architecture with two main components:
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

```python
class Net(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
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
- Results are stored in 'results/FL_Wafer/' directory
- Generates:
  - Training/validation loss and accuracy curves
  - Confusion matrices
  - ROC curves