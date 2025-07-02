from typing import Dict, Any, Optional, List

def create_config(
    name: str,
    dataset: str,
    model: str = "cnn",
    
    # Model architecture
    conv_channels: Optional[List[int]] = None,
    hidden_dims: Optional[List[int]] = None,
    dropout_rate: float = 0.2,
    
    # Training
    epochs: int = 25,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    optimizer: str = "adam",
    
    # FHE options
    use_fhe: bool = False,
    fhe_scheme: str = "CKKS",
    fhe_layers: Optional[List[str]] = None,
    
    # Quantum options
    use_quantum: bool = False,
    n_qubits: int = 4,
    quantum_layers: Optional[List[str]] = None,
    quantum_circuit: str = "basic_entangler",
    
    # Federated learning
    federated: bool = False,
    num_clients: int = 10,
    num_rounds: int = 20,
    local_epochs: int = 5,
    
    # Environment
    gpu: bool = True,
    gpu_ids: Optional[List[int]] = None,
    debug: bool = False,
    
    # Paths
    data_path: str = "./data",
    output_path: str = "./outputs",
    
    # Custom parameters
    **kwargs
) -> Dict[str, Any]:
    """
    Create an experiment configuration dictionary.
    
    This is the main entry point - just call this function with your parameters!
    
    Args:
        name: Experiment name
        dataset: Dataset name (PILL, DNA, MRI, HIV, CIFAR10)
        model: Model architecture (cnn, mlp, gcn, pretrained_cnn)
        
        # Model parameters
        conv_channels: Convolutional channels for CNN [32, 64, 128]
        hidden_dims: Hidden layer dimensions [256, 128]
        dropout_rate: Dropout rate (0.0-1.0)
        
        # Training parameters
        epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        optimizer: Optimizer (adam, sgd, adamw)
        
        # FHE parameters
        use_fhe: Enable FHE encryption
        fhe_scheme: FHE scheme (CKKS, BFV, TFHE)
        fhe_layers: Which layers to encrypt ['classifier']
        
        # Quantum parameters
        use_quantum: Enable quantum processing
        n_qubits: Number of qubits
        quantum_layers: Which layers to enhance ['classifier']
        quantum_circuit: Quantum circuit type
        
        # Federated learning
        federated: Enable federated learning
        num_clients: Number of federated clients
        num_rounds: Number of federated rounds
        local_epochs: Local epochs per round
        
        # Environment
        gpu: Use GPU
        gpu_ids: GPU device IDs [0]
        debug: Debug mode
        
        # Paths
        data_path: Path to data
        output_path: Path for outputs
        
        **kwargs: Any additional parameters
    
    Returns:
        Dictionary with all configuration parameters
    """
    
    # Dataset info
    dataset_info = {
        'PILL': {'input_shape': (3, 224, 224), 'num_classes': 2},  # Binary classification: bad/good pills
        'DNA': {'input_shape': (180,), 'num_classes': 7},
        'MRI': {'input_shape': (3, 224, 224), 'num_classes': 4},
        'HIV': {'input_shape': (9,), 'num_classes': 2},
        'CIFAR10': {'input_shape': (3, 32, 32), 'num_classes': 10}
    }
    
    # Set defaults based on model type
    if conv_channels is None:
        conv_channels = [32, 64, 128] if model == "cnn" else []
    
    if hidden_dims is None:
        if model == "mlp":
            hidden_dims = [128, 64]
        elif model == "gcn":
            hidden_dims = [64, 64]
        else:
            hidden_dims = [256]
    
    if fhe_layers is None:
        fhe_layers = ['classifier'] if use_fhe else []
    
    if quantum_layers is None:
        quantum_layers = ['classifier'] if use_quantum else []
    
    if gpu_ids is None:
        gpu_ids = [0] if gpu else []
    
    # Build configuration
    config = {
        # Experiment info
        'experiment_name': name,
        'dataset': dataset,
        
        # Add dataset-specific info
        **dataset_info.get(dataset, {}),
        
        # Model
        'model': model,
        'conv_channels': conv_channels,
        'hidden_dims': hidden_dims,
        'dropout_rate': dropout_rate,
        
        # Training
        'epochs': epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'optimizer': optimizer,
        
        # FHE
        'use_fhe': use_fhe,
        'fhe_scheme': fhe_scheme,
        'fhe_layers': fhe_layers,
        
        # Quantum
        'use_quantum': use_quantum,
        'n_qubits': n_qubits,
        'quantum_layers': quantum_layers,
        'quantum_circuit': quantum_circuit,
        
        # Federated
        'federated': federated,
        'num_clients': num_clients,
        'num_rounds': num_rounds,
        'local_epochs': local_epochs,
        
        # Environment
        'gpu': gpu,
        'gpu_ids': gpu_ids,
        'debug': debug,
        
        # Paths
        'data_path': data_path,
        'output_path': output_path,
        
        # Custom parameters
        **kwargs
    }
    
    # Print configuration
    print(f"  Creating config: {name}")
    print(f"   Dataset: {dataset}")
    print(f"   Model: {model}")
    
    if use_fhe:
        print(f"   FHE: {fhe_scheme} on {fhe_layers}")
    
    if use_quantum:
        print(f"   Quantum: {n_qubits} qubits on {quantum_layers}")
    
    if federated:
        print(f"   Federated: {num_clients} clients, {num_rounds} rounds")
    
    print(f"   Training: {epochs} epochs, lr={learning_rate}")
    
    # Return the configuration dictionary
    return config



