import os
import time
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable, List, Tuple


def run_centralized_simulation(
    config: Dict[str, Any],
    model_params: Dict[str, Any] = None,
    result_base_path: str = None,
    custom_data_loader: Optional[Callable] = None,
    dataset_params: Optional[Dict[str, Any]] = None,
    class_names: Optional[List[str]] = None,
    custom_model: Optional[torch.nn.Module] = None
) -> Dict[str, float]:
    """
    Run complete centralized training simulation with all boilerplate handled.
    
    This eliminates ~90 lines of repetitive code from every centralized example.
    Supports both built-in and custom dataset loading functions, and custom models.
    
    Args:
        config: Configuration dictionary from create_config()
        model_params: Parameters for create_model() (architecture, quantum settings, etc.)
                     Ignored if custom_model is provided.
        result_base_path: Base path for saving results
        custom_data_loader: Optional custom function that returns (train_loader, val_loader, test_loader)
        dataset_params: Optional dataset loading parameters
        class_names: Optional custom class names list
        custom_model: Optional custom PyTorch model. If provided, model_params is ignored.
        
    Returns:
        Dictionary with training results (train_acc, test_acc, training_time, etc.)
        
    Example:
        ```python
        # Simple usage with built-in datasets
        results = run_centralized_simulation(
            config=config,
            model_params={'base_architecture': 'mlp', 'hidden_dims': [64, 32]}
        )
        
        # Advanced usage with custom dataset
        def my_custom_loader(config, dataset_params):
            # Your custom loading logic
            return train_loader, val_loader, test_loader
            
        results = run_centralized_simulation(
            config=config,
            model_params=model_params,
            custom_data_loader=my_custom_loader,
            class_names=["healthy", "disease", "critical"]
        )
        ```
    """
    from ..data.loaders import load_datasets, infer_dataset_properties
    from ..models.modular import create_model
    from ..core.training import train
    from ..core.testing import test
    from ..core.visualization import save_all_results
    
    # Auto-generate result path if not provided
    if result_base_path is None:
        result_base_path = f"results/centralized_{config['dataset'].lower()}_example"
    
    # Set default parameters
    if dataset_params is None:
        dataset_params = {}
    if model_params is None:
        model_params = {}
    
    # Device setup
    device = torch.device('cuda' if config['gpu'] else 'cpu')
    print(f"Using device: {device}")
    
    # Load data using library's data loading system or custom loader
    if custom_data_loader is not None:
        print("Using custom dataset loader...")
        train_loader, val_loader, test_loader = custom_data_loader(config, dataset_params)
    else:
        print(f"Loading {config['dataset']} dataset using library...")
        trainloaders, valloaders, testloader = load_datasets(
            num_clients=1,
            batch_size=config['batch_size'],
            resize=dataset_params.get('resize', 224) if dataset_params.get('resize') is not None else 224,
            seed=dataset_params.get('seed', 42),
            num_workers=dataset_params.get('num_workers', 0),
            splitter=dataset_params.get('splitter', 10),
            dataset=config['dataset'],
            data_path=dataset_params.get('data_path', "data/"),
            custom_normalizations=dataset_params.get('custom_normalizations')
        )
        
        # Get the single client's data loaders
        train_loader = trainloaders[0]
        val_loader = valloaders[0]
        test_loader = testloader
    
    # Get input shape and number of classes from the data
    input_shape, num_classes = infer_dataset_properties(test_loader)
    
    print(f"Dataset loaded: {len(train_loader.dataset)} train samples, {len(test_loader.dataset)} test samples")
    print(f"Number of classes: {num_classes}")
    print(f"Input shape: {input_shape}")
    
    # Create model using either custom model or library's model creation system
    if custom_model is not None:
        model = custom_model.to(device)
        print("Using custom model")
    else:
        model = create_model(
            input_shape=input_shape,
            num_classes=num_classes,
            **model_params
        ).to(device)
        print("Using library model creation system")
    
    print(f"Model moved to: {device}")
    
    # Setup optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_fn = nn.CrossEntropyLoss()
    
    # Create results directory
    os.makedirs(result_base_path, exist_ok=True)
    
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
        dataloader=test_loader,
        loss_fn=loss_fn,
        device=device
    )
    
    # Auto-generate class names if not provided
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    
    # Save all visualization results using convenience function
    save_all_results(
        train_true=train_true,
        train_pred=train_pred,
        train_proba=train_proba,
        test_true=test_true,
        test_pred=test_pred,
        test_proba=test_proba,
        training_history=results,
        classes=class_names,
        results_path=result_base_path,
        config=config,
        file_suffix=f"_{config['dataset']}_centralized"
    )
    
    training_time = time.time() - start_time
    
    print(f"Training accuracy: {train_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Training plots and confusion matrices saved to {result_base_path}/ folder")
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'training_time': training_time,
        'num_classes': num_classes,
        'input_shape': input_shape
    } 