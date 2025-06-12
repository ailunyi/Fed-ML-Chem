"""
Quantum Training Example
========================

This example demonstrates how to use the fed_ml_lib quantum models for training
on any dataset. This includes hybrid quantum-classical models and pure quantum models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add the library to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fed_ml_lib.models import create_model, HybridCNN_QNN, QuantumNet, VariationalQuantumClassifier
from fed_ml_lib.engine import run_central_training
from fed_ml_lib.datasets import create_data_loaders, get_dataset_info


def main():
    # Configuration
    config = {
        'dataset_name': 'PILL',  # Change this to your dataset name
        'data_path': './data/',
        'batch_size': 16,  # Smaller batch size for quantum models
        'resize': 64,  # Smaller images for quantum processing
        'val_split': 0.1,
        'epochs': 10,  # Fewer epochs for demonstration
        'learning_rate': 1e-3,
        'seed': 42,
        'results_dir': './results/quantum_training/',
        
        # Quantum model parameters
        'model_type': 'hybrid_cnn_qnn',  # Options: 'quantum', 'hybrid_cnn_qnn', 'vqc'
        'n_qubits': 4,
        'n_layers': 2,
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
    
    # Create quantum model
    print(f"Creating {config['model_type']} model...")
    
    if config['model_type'] == 'hybrid_cnn_qnn':
        model = HybridCNN_QNN(
            num_classes=dataset_info['num_classes'],
            n_qubits=config['n_qubits'],
            n_layers=config['n_layers']
        )
    elif config['model_type'] == 'quantum':
        # For flattened image input
        input_size = 3 * config['resize'] * config['resize']  # RGB image flattened
        model = QuantumNet(
            input_size=input_size,
            num_classes=dataset_info['num_classes'],
            n_qubits=config['n_qubits'],
            n_layers=config['n_layers']
        )
    elif config['model_type'] == 'vqc':
        model = VariationalQuantumClassifier(
            input_size=config['n_qubits'],  # Will be preprocessed to match
            num_classes=dataset_info['num_classes'],
            n_qubits=config['n_qubits'],
            n_layers=config['n_layers']
        )
    else:
        # Use factory function
        model = create_model(
            model_type=config['model_type'],
            num_classes=dataset_info['num_classes'],
            n_qubits=config['n_qubits'],
            n_layers=config['n_layers']
        )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer and loss function
    # Use smaller learning rate for quantum models
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_fn = nn.CrossEntropyLoss()
    
    # Run training
    print("Starting quantum training...")
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
    print("\nQuantum training completed!")
    print(f"Final training accuracy: {results['train_acc'][-1]:.2f}%")
    print(f"Final validation accuracy: {results['val_acc'][-1]:.2f}%")
    print(f"Final training loss: {results['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {results['val_loss'][-1]:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating quantum model on test set...")
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
            path=os.path.join(config['results_dir'], f'confusion_matrix_{config["model_type"]}.png'),
            classes=dataset_info['classes']
        )
        
        save_roc(
            y_true=y_true,
            y_proba=y_proba,
            path=os.path.join(config['results_dir'], f'roc_curve_{config["model_type"]}.png'),
            num_classes=dataset_info['num_classes']
        )
        
        print(f"Results saved to: {config['results_dir']}")
        
    except Exception as e:
        print(f"Warning: Could not save evaluation plots: {e}")


def compare_models():
    """
    Compare different quantum model architectures on the same dataset.
    """
    print("\n" + "="*50)
    print("COMPARING QUANTUM MODEL ARCHITECTURES")
    print("="*50)
    
    model_configs = [
        {'type': 'hybrid_cnn_qnn', 'n_qubits': 4, 'n_layers': 2},
        {'type': 'quantum', 'n_qubits': 6, 'n_layers': 3},
        {'type': 'vqc', 'n_qubits': 4, 'n_layers': 2},
    ]
    
    dataset_name = 'PILL'
    results_comparison = {}
    
    for model_config in model_configs:
        print(f"\nTraining {model_config['type']} model...")
        
        # Quick training configuration
        config = {
            'dataset_name': dataset_name,
            'data_path': './data/',
            'batch_size': 8,
            'resize': 32,  # Very small for quick comparison
            'epochs': 3,   # Very few epochs for quick comparison
            'learning_rate': 1e-3,
            'seed': 42,
        }
        
        try:
            # Get data
            dataset_info = get_dataset_info(config['dataset_name'], config['data_path'])
            train_loader, val_loader, _ = create_data_loaders(
                dataset_name=config['dataset_name'],
                data_path=config['data_path'],
                batch_size=config['batch_size'],
                resize=config['resize'],
                val_split=0.2,  # Larger validation set for quick evaluation
                seed=config['seed']
            )
            
            # Create model
            if model_config['type'] == 'hybrid_cnn_qnn':
                model = HybridCNN_QNN(
                    num_classes=dataset_info['num_classes'],
                    n_qubits=model_config['n_qubits'],
                    n_layers=model_config['n_layers']
                )
            elif model_config['type'] == 'quantum':
                input_size = 3 * config['resize'] * config['resize']
                model = QuantumNet(
                    input_size=input_size,
                    num_classes=dataset_info['num_classes'],
                    n_qubits=model_config['n_qubits'],
                    n_layers=model_config['n_layers']
                )
            elif model_config['type'] == 'vqc':
                model = VariationalQuantumClassifier(
                    input_size=model_config['n_qubits'],
                    num_classes=dataset_info['num_classes'],
                    n_qubits=model_config['n_qubits'],
                    n_layers=model_config['n_layers']
                )
            
            # Train
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
            loss_fn = nn.CrossEntropyLoss()
            
            results = run_central_training(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                epochs=config['epochs'],
                results_dir=None,  # Don't save plots for comparison
                plot_results=False
            )
            
            results_comparison[model_config['type']] = {
                'final_val_acc': results['val_acc'][-1],
                'final_val_loss': results['val_loss'][-1],
                'parameters': sum(p.numel() for p in model.parameters())
            }
            
            print(f"{model_config['type']}: Val Acc = {results['val_acc'][-1]:.2f}%, "
                  f"Val Loss = {results['val_loss'][-1]:.4f}, "
                  f"Params = {results_comparison[model_config['type']]['parameters']}")
            
        except Exception as e:
            print(f"Error training {model_config['type']}: {e}")
            results_comparison[model_config['type']] = {'error': str(e)}
    
    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    for model_type, results in results_comparison.items():
        if 'error' in results:
            print(f"{model_type}: ERROR - {results['error']}")
        else:
            print(f"{model_type}: {results['final_val_acc']:.2f}% accuracy, "
                  f"{results['parameters']} parameters")


if __name__ == "__main__":
    # Run main quantum training example
    main()
    
    # Optionally run model comparison
    print("\nWould you like to compare different quantum models? (This will take additional time)")
    # Uncomment the line below to run comparison
    # compare_models() 