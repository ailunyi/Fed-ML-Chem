"""
Central Training Example
========================

This example demonstrates how to use the fed_ml_lib for centralized training
on any dataset. This example can be adapted for different datasets by changing
the dataset_name parameter.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add the library to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fed_ml_lib.models import VGG16Classifier
from fed_ml_lib.engine import run_central_training
from fed_ml_lib.datasets import create_data_loaders, get_dataset_info


def main():
    # Configuration
    config = {
        'dataset_name': 'PILL',  # Change this to your dataset name
        'data_path': './data/',
        'batch_size': 32,
        'resize': 224,  # Resize images to 224x224 for VGG16
        'val_split': 0.1,
        'epochs': 25,
        'learning_rate': 2e-4,
        'seed': 42,
        'results_dir': './results/central_training/',
    }
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get dataset information
    dataset_info = get_dataset_info(config['dataset_name'], config['data_path'])
    print(f"Dataset info: {dataset_info}")
    
    # Create data loaders
    print("Loading dataset...")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_name=config['dataset_name'],
        data_path=config['data_path'],
        batch_size=config['batch_size'],
        resize=config['resize'],
        val_split=config['val_split'],
        seed=config['seed']
    )
    
    # Create model
    print("Creating model...")
    model = VGG16Classifier(num_classes=dataset_info['num_classes'])
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_fn = nn.CrossEntropyLoss()
    
    # Run training
    print("Starting training...")
    results = run_central_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=config['epochs'],
        results_dir=config['results_dir'],
        plot_results=True
    )
    
    # Print final results
    print("\nTraining completed!")
    print(f"Final training accuracy: {results['train_acc'][-1]:.2f}%")
    print(f"Final validation accuracy: {results['val_acc'][-1]:.2f}%")
    print(f"Final training loss: {results['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {results['val_loss'][-1]:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    from fed_ml_lib.engine import test
    test_loss, test_acc, y_pred, y_true, y_proba = test(
        model=model,
        dataloader=test_loader,
        loss_fn=loss_fn,
        device=device
    )
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    
    # Save confusion matrix and ROC curve
    try:
        from fed_ml_lib.utils import save_matrix, save_roc
        os.makedirs(config['results_dir'], exist_ok=True)
        
        save_matrix(
            y_true=y_true,
            y_pred=y_pred,
            path=os.path.join(config['results_dir'], 'confusion_matrix_test.png'),
            classes=dataset_info['classes']
        )
        
        save_roc(
            y_true=y_true,
            y_proba=y_proba,
            path=os.path.join(config['results_dir'], 'roc_curve_test.png'),
            num_classes=dataset_info['num_classes']
        )
        
        print(f"Results saved to: {config['results_dir']}")
        
    except Exception as e:
        print(f"Warning: Could not save evaluation plots: {e}")


if __name__ == "__main__":
    main() 