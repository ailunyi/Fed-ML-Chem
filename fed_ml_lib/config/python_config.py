from typing import Dict, Any, Optional, List, Callable
import inspect
from pathlib import Path

def run_experiment(
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
    Run an experiment with the given configuration.
    
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
    print(f"  Running experiment: {name}")
    print(f"   Dataset: {dataset}")
    print(f"   Model: {model}")
    
    if use_fhe:
        print(f"   FHE: {fhe_scheme} on {fhe_layers}")
    
    if use_quantum:
        print(f"   Quantum: {n_qubits} qubits on {quantum_layers}")
    
    if federated:
        print(f"   Federated: {num_clients} clients, {num_rounds} rounds")
    
    print(f"   Training: {epochs} epochs, lr={learning_rate}")
    
    # Here you would call your actual training code
    # For now, just return the config
    return config

# Convenience functions for common scenarios
def pill_cnn(name: str, **kwargs):
    """Quick PILL CNN experiment."""
    return run_experiment(name, "PILL", "cnn", **kwargs)

def dna_mlp(name: str, **kwargs):
    """Quick DNA MLP experiment."""
    return run_experiment(name, "DNA", "mlp", **kwargs)

def mri_cnn(name: str, **kwargs):
    """Quick MRI CNN experiment."""
    return run_experiment(name, "MRI", "cnn", **kwargs)

def federated_pill(name: str, **kwargs):
    """Quick federated PILL experiment."""
    return run_experiment(name, "PILL", "cnn", federated=True, **kwargs)

def quantum_mri(name: str, **kwargs):
    """Quick quantum MRI experiment."""
    return run_experiment(name, "MRI", "cnn", use_quantum=True, **kwargs)

def fhe_dna(name: str, **kwargs):
    """Quick FHE DNA experiment."""
    return run_experiment(name, "DNA", "mlp", use_fhe=True, **kwargs)

def hybrid_pill(name: str, **kwargs):
    """Quick hybrid FHE+Quantum PILL experiment."""
    return run_experiment(name, "PILL", "cnn", use_fhe=True, use_quantum=True, **kwargs)

# Configuration templates as Python functions
def basic_cnn_config(name: str, dataset: str, **kwargs):
    """Basic CNN configuration template."""
    defaults = {
        'model': 'cnn',
        'conv_channels': [32, 64, 128],
        'hidden_dims': [256],
        'dropout_rate': 0.2,
        'epochs': 25,
        'learning_rate': 0.001,
        'batch_size': 32
    }
    defaults.update(kwargs)
    return run_experiment(name, dataset, **defaults)

def federated_config(name: str, dataset: str, **kwargs):
    """Federated learning configuration template."""
    defaults = {
        'model': 'cnn',
        'conv_channels': [16, 32, 64],
        'epochs': 10,
        'learning_rate': 0.01,
        'batch_size': 16,
        'federated': True,
        'num_clients': 10,
        'num_rounds': 20,
        'local_epochs': 5
    }
    defaults.update(kwargs)
    return run_experiment(name, dataset, **defaults)

def quantum_config(name: str, dataset: str, **kwargs):
    """Quantum ML configuration template."""
    defaults = {
        'model': 'cnn',
        'conv_channels': [32, 64],
        'use_quantum': True,
        'n_qubits': 4,
        'quantum_layers': ['classifier'],
        'epochs': 20,
        'learning_rate': 0.001
    }
    defaults.update(kwargs)
    return run_experiment(name, dataset, **defaults)

def fhe_config(name: str, dataset: str, **kwargs):
    """FHE configuration template."""
    defaults = {
        'model': 'cnn',
        'conv_channels': [16, 32],
        'use_fhe': True,
        'fhe_scheme': 'CKKS',
        'fhe_layers': ['classifier'],
        'epochs': 15,
        'learning_rate': 0.0005
    }
    defaults.update(kwargs)
    return run_experiment(name, dataset, **defaults)

def hybrid_config(name: str, dataset: str, **kwargs):
    """Hybrid FHE+Quantum configuration template."""
    defaults = {
        'model': 'cnn',
        'conv_channels': [32, 64],
        'use_fhe': True,
        'fhe_scheme': 'CKKS',
        'fhe_layers': ['classifier'],
        'use_quantum': True,
        'n_qubits': 4,
        'quantum_layers': ['features'],
        'epochs': 20,
        'learning_rate': 0.001
    }
    defaults.update(kwargs)
    return run_experiment(name, dataset, **defaults)

# Batch experiment runner
def run_experiments(experiments: List[Callable], **common_kwargs):
    """
    Run multiple experiments with common parameters.
    
    Args:
        experiments: List of experiment functions to run
        **common_kwargs: Common parameters for all experiments
    
    Example:
        run_experiments([
            lambda: pill_cnn("exp1", epochs=50),
            lambda: dna_mlp("exp2", epochs=40),
            lambda: quantum_mri("exp3", n_qubits=6)
        ], debug=True, gpu=False)
    """
    results = []
    
    for i, experiment_func in enumerate(experiments):
        print(f"\n--- Running experiment {i+1}/{len(experiments)} ---")
        
        # Get the experiment function signature
        if hasattr(experiment_func, '__call__'):
            try:
                result = experiment_func()
                # Apply common kwargs
                result.update(common_kwargs)
                results.append(result)
            except Exception as e:
                print(f"Experiment {i+1} failed: {e}")
                results.append(None)
    
    return results

# Parameter sweep helper
def parameter_sweep(base_func: Callable, param_name: str, values: List, **base_kwargs):
    """
    Run parameter sweep over a list of values.
    
    Args:
        base_func: Base experiment function
        param_name: Parameter name to sweep
        values: List of values to try
        **base_kwargs: Base parameters
    
    Example:
        parameter_sweep(
            lambda name, **kwargs: pill_cnn(name, **kwargs),
            'learning_rate',
            [0.001, 0.01, 0.1],
            epochs=20
        )
    """
    results = []
    
    for i, value in enumerate(values):
        experiment_name = f"sweep_{param_name}_{i:03d}"
        print(f"\n--- Sweep {i+1}/{len(values)}: {param_name}={value} ---")
        
        kwargs = base_kwargs.copy()
        kwargs[param_name] = value
        
        try:
            result = base_func(experiment_name, **kwargs)
            results.append(result)
        except Exception as e:
            print(f"Sweep {i+1} failed: {e}")
            results.append(None)
    
    return results

    
# Custom experiment with lots of parameters
# run_experiment(
#     name="custom_experiment",
#     dataset="DNA",
#     model="mlp",
#     hidden_dims=[512, 256, 128, 64],
#     dropout_rate=0.3,
#     epochs=100,
#     learning_rate=0.0005,
#     batch_size=16,
#     use_quantum=True,
#     n_qubits=10,
#     quantum_layers=['features'],
#     federated=True,
#     num_clients=20,
#     debug=True
# ) 