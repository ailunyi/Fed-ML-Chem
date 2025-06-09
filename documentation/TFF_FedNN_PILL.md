# TensorFlow Federated Neural Network for PILL Dataset

## Overview
This notebook implements a federated learning approach for pill image classification using TensorFlow Federated (TFF) instead of the Flower framework used in other notebooks. It distributes the training across multiple clients, each with their own subset of the data, and aggregates the model updates using the Federated Averaging (FedAvg) algorithm. The model is designed to classify pill images as either faulty or normal.

## Model Architecture
The model uses a transfer learning approach with VGG16 as the base model:
- VGG16 pretrained on ImageNet as a feature extractor
- First 23 layers of VGG16 are frozen (non-trainable)
- AveragePooling2D layer to reduce spatial dimensions
- Flatten layer to convert 3D feature maps to 1D vector
- Final Dense layer with softmax activation for classification

```python
class PillModel:
    def __init__(self, input_shape, num_classes):
        base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        for layer in base_model.layers[:23]:  # Freeze first 23 layers
            layer.trainable = False
            
        self.model = tf.keras.Sequential([
            base_model,
            layers.AveragePooling2D(pool_size=(224 // 2 ** 5, 224 // 2 ** 5)),
            layers.Flatten(),
            
            # Output layer for classification
            layers.Dense(num_classes, activation='softmax')
        ])
```

## Dataset
- PILL dataset with binary classification (faulty vs. normal pills)
- Images are resized to 256×256 pixels
- The dataset is split across 10 clients for federated learning
- Uses TensorFlow's image_dataset_from_directory for loading data
- Normalizes pixel values by dividing by 255.0

## Training Approach
- Federated learning with 10 clients using TensorFlow Federated
- Implements the Federated Averaging (FedAvg) algorithm
- Each client trains for 10 epochs per round
- Training runs for 20 rounds of federated learning
- Uses CategoricalCrossentropy loss function
- Adam optimizer with learning rate of 0.003
- Evaluates the model after each round using all clients

## Key Parameters
- `num_clients = 10`: Number of federated learning clients
- `image_shape = (256, 256, 3)`: Input image dimensions(256 height, 256 width, 3 colors red blue green)
- `num_categories = 2`: Number of classes (binary classification)
- `num_rounds = 20`: Number of federated learning rounds
- `num_epochs = 10`: Number of local epochs per round
- `batch_size = 32`: Batch size for training and evaluation
- Learning rate = 0.003: For Adam optimizer

## Results and Visualization
- Tracks and reports training and evaluation metrics after each round
- Calculates final averaged accuracy over all rounds
- Provides a comprehensive federated learning implementation using TensorFlow Federated

## Key Differences from Flower-based Implementations
- Uses TensorFlow Federated (TFF) instead of Flower for the federated learning framework
- Implements Federated Averaging using TFF's built-in functions
- Uses a slightly different model architecture and training approach
- Provides an alternative implementation for comparison with the Flower-based approaches