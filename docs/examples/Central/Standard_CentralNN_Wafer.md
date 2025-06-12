# Standard Centralized Neural Network for Wafer Dataset

## Overview
This notebook implements a centralized neural network approach for semiconductor wafer classification using a Convolutional Neural Network (CNN). It uses the Flower framework with a single client to effectively create a centralized learning setup. The model is designed to classify wafer images into different categories based on their manufacturing quality or characteristics.

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
- Uses standard data augmentation techniques for training
- The dataset is processed using a single client (centralized approach)

## Training Approach
- Centralized learning with 1 client using the Flower framework
- Standard CrossEntropyLoss function
- Adam optimizer with learning rate of 1e-3
- Trains for 25 epochs in a single round
- Uses standard training and evaluation functions:
  - engine.train for training
  - engine.test for evaluation

## Key Parameters
- `number_clients = 1`: Single client for centralized learning
- `rounds = 1`: Single round of training
- `max_epochs = 25`: Number of epochs for training
- `batch_size = 64`: Batch size for training and evaluation
- `resize = 64`: Image resize dimensions
- `lr = 1e-3`: Learning rate for the Adam optimizer

## Results and Visualization
- Saves confusion matrices and ROC curves
- Saves model checkpoints
- Results are stored in 'results/CL_Wafer/' directory
- Generates:
  - Training/validation loss and accuracy curves
  - Confusion matrices
  - ROC curves