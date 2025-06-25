"""
Simple Centralized DNA Training Example using Fed-ML-Lib data loading
"""
import torch
import torch.nn as nn
import os
import time

from fed_ml_lib.config.python_config import run_experiment
from fed_ml_lib.models.modular import create_model
from fed_ml_lib.data.loaders import load_datasets
from fed_ml_lib.core.visualization import save_graphs, save_matrix
from fed_ml_lib.core.testing import test
from fed_ml_lib.core.training import train

def main():
    """Main training function."""
    # Configuration using library's config system
    config = run_experiment(
        name="dna_centralized",
        dataset="DNA",
        model="mlp",
        epochs=25,
        learning_rate=0.001,
        batch_size=32,
        hidden_dims=[64, 32, 16, 8],
        dropout_rate=0.0,
        gpu=torch.cuda.is_available()
    )
    
    # Setup device
    device = torch.device('cuda' if config['gpu'] else 'cpu')
    print(f"Using device: {device}")
    
    # Load data using library's data loading system
    print("Loading DNA dataset using library...")
    trainloaders, valloaders, testloader = load_datasets(
        num_clients=1,
        batch_size=config['batch_size'],
        resize=None,  # Not needed for DNA data
        seed=42,
        num_workers=0,
        splitter=10,
        dataset='DNA',
        data_path="data/"
    )
    
    # Get the single client's data loaders
    train_loader = trainloaders[0]
    val_loader = valloaders[0]
    
    # Get input shape and number of classes from the data
    sample_input, sample_label = next(iter(testloader))
    input_shape = (sample_input.shape[1],)  # Feature dimension
    
    # Get number of classes from the data
    all_labels = set()
    for _, labels in testloader:
        all_labels.update(labels.tolist())
    for _, labels in train_loader:
        all_labels.update(labels.tolist())
    num_classes = len(all_labels)
    
    print(f"Dataset loaded: {len(train_loader.dataset)} train samples, {len(testloader.dataset)} test samples")
    print(f"Number of classes: {num_classes}")
    print(f"Input shape: {input_shape}")
    
    # Create model using library's model creation system
    model = create_model(
        base_architecture="mlp",
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_dims=config['hidden_dims'],
        dropout_rate=config['dropout_rate'],
        domain='sequence'
    )
    
    # Move model to device
    model = model.to(device)
    print(f"Model moved to: {device}")
    
    # Setup optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_fn = nn.CrossEntropyLoss()
    
    # Create results directory
    os.makedirs("results/centralized_dna_example", exist_ok=True)
    
    # Train model using library's training function
    print("Starting training...")
    start_time = time.time()
    
    results = train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=config['epochs'],
        device=device
    )
    
    # Test on training data
    train_loss, train_accuracy, train_pred, train_true, train_proba = test(
        model=model,
        dataloader=train_loader,
        loss_fn=loss_fn,
        device=device
    )
    
    # Test on test data
    test_loss, test_accuracy, test_pred, test_true, test_proba = test(
        model=model,
        dataloader=testloader,
        loss_fn=loss_fn,
        device=device
    )
    
    # Save confusion matrices
    class_names = [str(i) for i in range(num_classes)]
    save_matrix(
        y_true=train_true,
        y_pred=train_pred,
        classes=class_names,
        path="results/centralized_dna_example/confusion_matrix_train.png"
    )
    
    save_matrix(
        y_true=test_true,
        y_pred=test_pred,
        classes=class_names,
        path="results/centralized_dna_example/confusion_matrix_test.png"
    )
    
    # Save training curves
    save_graphs(
        path_save="results/centralized_dna_example/",
        local_epoch=config['epochs'],
        results=results,
        end_file="_DNA_centralized"
    )
    
    print(f"Training accuracy: {train_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")
    print("Training plots and confusion matrices saved to results/centralized_dna_example/ folder")

if __name__ == "__main__":
    main() 