"""
Simple Centralized PILL Training Example using Fed-ML-Lib's built-in functionality
"""
import torch
import torch.nn as nn
import os

from fed_ml_lib.config import pill_cnn
from fed_ml_lib.models.modular import create_model
from fed_ml_lib.data.loaders import load_datasets
from fed_ml_lib.core.training import train
from fed_ml_lib.core.testing import test
from fed_ml_lib.core.visualization import save_matrix, save_graphs, save_roc

def main():
    config = pill_cnn("pill_centralized", epochs=25, learning_rate=0.0002, batch_size=32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data using Fed-ML-Lib's data loader
    trainloaders, valloaders, test_loader = load_datasets(
        num_clients=1,
        batch_size=config['batch_size'],
        resize=224,  # Changed to match legacy size
        seed=42,
        num_workers=0,
        splitter=10,
        dataset='PILL',
        data_path='data/'
    )
    train_loader = trainloaders[0]
    val_loader = valloaders[0]
    
    # Create model using library's built-in functionality
    model_config = {
        'base_architecture': 'pretrained_cnn',
        'num_classes': 2,
        'freeze_layers': 23,  # Freeze first 23 layers like in original
        'input_shape': (3, 224, 224),  # RGB images resized to 224x224
        'use_fhe': False,
        'use_quantum': False
    }
    
    model = create_model(**model_config).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_fn = nn.CrossEntropyLoss()
    
    os.makedirs("results/centralized_pill_example", exist_ok=True)
    
    # Train model using Fed-ML-Lib's training function
    training_history = train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=config['epochs'],
        device=device
    )
    
    # Test on training data for training confusion matrix
    train_loss, train_acc, train_pred, train_true, train_proba = test(
        model=model,
        dataloader=train_loader,
        loss_fn=loss_fn,
        device=device
    )
    
    # Test on test data for test confusion matrix
    test_loss, test_acc, test_pred, test_true, test_proba = test(
        model=model,
        dataloader=test_loader,
        loss_fn=loss_fn,
        device=device
    )
    
    # Save confusion matrices and training curves
    save_matrix(
        y_true=train_true,
        y_pred=train_pred,
        classes=['bad', 'good'],
        path="results/centralized_pill_example/confusion_matrix_train.png"
    )
    
    save_matrix(
        y_true=test_true,
        y_pred=test_pred,
        classes=['bad', 'good'],
        path="results/centralized_pill_example/confusion_matrix_test.png"
    )
    
    # Save ROC curves
    save_roc(
        targets=train_true,
        y_proba=train_proba,
        path="results/centralized_pill_example/roc_train.png",
        nbr_classes=2
    )
    
    save_roc(
        targets=test_true,
        y_proba=test_proba,
        path="results/centralized_pill_example/roc_test.png",
        nbr_classes=2
    )
    
    # Save training curves
    save_graphs(
        path_save="results/centralized_pill_example/",
        local_epoch=config['epochs'],
        results=training_history,
        end_file=""
    )
    
    print(f"Training accuracy: {train_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main() 