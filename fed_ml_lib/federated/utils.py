import os
import torch
import numpy as np
import time
from typing import List, Tuple, Dict, Optional, Callable, Any
from collections import OrderedDict
from flwr.common import Metrics, NDArrays, Scalar, parameters_to_ndarrays
from ..core.utils import *

# These will need to be passed as parameters or defined elsewhere
central = None
testloader = None
DEVICE = None
save_results = ""
CLASSES = []

# DEVICE = torch.device(choice_device(device))
# CLASSES = classes_string(dataset)
# trainloaders, valloaders, testloader = data_setup.load_datasets(num_clients=number_clients, batch_size=batch_size, resize=None, seed=seed, num_workers=num_workers, splitter=splitter, dataset=dataset, data_path=data_path, data_path_val=None)
# _, input_sp = next(iter(testloader))[0].shape
# central = Net(num_classes=len(CLASSES)).to(DEVICE)

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Filter out clients with no validation examples to prevent dimension mismatch
    valid_metrics = [(num_examples, m) for num_examples, m in metrics if num_examples > 0]
    
    if not valid_metrics:
        # If no clients have validation data, return default accuracy
        return {"accuracy": 0.0}
    
    accuracies = [num_examples * m["accuracy"] for num_examples, m in valid_metrics]
    examples = [num_examples for num_examples, _ in valid_metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def evaluate2(server_round: int, parameters: NDArrays,
              config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    set_parameters(central, parameters)
    loss, accuracy, y_pred, y_true, y_proba = engine.test(central, testloader, loss_fn=torch.nn.CrossEntropyLoss(),
                                                          device=DEVICE)
    os.makedirs(save_results, exist_ok=True)
    # Note: This old function should use save_all_results for consistency
    # but keeping minimal change to preserve existing functionality
    from ..core.visualization import save_matrix, save_roc
    save_matrix(y_true, y_pred, save_results + "confusion_matrix_test.png", CLASSES)
    save_roc(y_true, y_proba, save_results + "roc_test.png", len(CLASSES))
    
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}

def get_on_fit_config_fn(epoch=2, lr=0.001, batch_size=32) -> Callable[[int], Dict[str, str]]:
    def fit_config(server_round: int) -> Dict[str, str]:
        config = {
            "learning_rate": str(lr),
            "batch_size": str(batch_size),
            "server_round": server_round,
            "local_epochs": epoch
        }
        return config
    return fit_config

def aggreg_fit_checkpoint(server_round, aggregated_parameters, central_model, path_checkpoint):
    if aggregated_parameters is not None:
        print(f"Saving round {server_round} aggregated_parameters...")
        aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
        
        params_dict = zip(central_model.state_dict().keys(), aggregated_ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        central_model.load_state_dict(state_dict, strict=True)
        if path_checkpoint:
            torch.save({
                'model_state_dict': central_model.state_dict(),
            }, path_checkpoint)


# New convenience functions to eliminate redundancy across federated examples
def create_client_fn(
    config: Dict[str, Any],
    num_clients: int,
    model_params: Dict[str, Any] = None,
    result_base_path: str = "results",
    dataset_params: Optional[Dict[str, Any]] = None,
    custom_model: Optional[torch.nn.Module] = None
) -> Callable:
    """
    Generate a client_fn for federated learning.
    
    This convenience function creates a standardized client_fn that follows
    the common pattern used across all Fed-ML-Chem federated examples.
    
    Args:
        config: Configuration dictionary from create_config()
        num_clients: Total number of federated clients
        model_params: Parameters for create_model() (architecture, quantum settings, etc.)
                     Ignored if custom_model is provided.
        result_base_path: Base path for saving client results
        dataset_params: Optional dataset loading parameters (splitter, etc.)
        custom_model: Optional custom PyTorch model template. If provided, model_params is ignored.
        
    Returns:
        A client_fn function ready for use in fl.simulation.start_simulation
        
    Example:
        ```python
        model_params = {
            'base_architecture': 'mlp',
            'hidden_dims': [64, 32, 16, 8],
            'dropout_rate': 0.0,
            'domain': 'sequence'
        }
        
        client_fn = create_client_fn(
            config=config,
            num_clients=3,
            model_params=model_params,
            result_base_path="results/federated_dna_example"
        )
        ```
    """
    from ..data.loaders import load_datasets, infer_dataset_properties
    from ..models.modular import create_model
    from .client import FlowerClient
    
    # Set default parameters
    if dataset_params is None:
        dataset_params = {}
    if model_params is None:
        model_params = {}
    
    def client_fn(cid: str):
        device = torch.device('cuda' if config.get('gpu', False) else 'cpu')
        
        # Load federated datasets - data is automatically split across clients
        trainloaders, valloaders, testloader = load_datasets(
            num_clients=num_clients,
            batch_size=config.get('batch_size', 32),
            resize=dataset_params.get('resize', 224) if dataset_params.get('resize') is not None else 224,
            seed=dataset_params.get('seed', 42),
            num_workers=dataset_params.get('num_workers', 0),
            splitter=dataset_params.get('splitter', 10),
            dataset=config.get('dataset', 'DNA'),
            data_path=dataset_params.get('data_path', "data/"),
            custom_normalizations=dataset_params.get('custom_normalizations')
        )
        
        # Get dataset properties for model creation
        input_shape, num_classes = infer_dataset_properties(testloader)
        
        # Create client model using either custom model or library's modular system
        if custom_model is not None:
            import copy
            model = copy.deepcopy(custom_model).to(device)
        else:
            model = create_model(
                input_shape=input_shape,
                num_classes=num_classes,
                **model_params
            ).to(device)
        
        class_names = [str(i) for i in range(num_classes)]
        
        # Return configured Flower client
        return FlowerClient(
            cid=cid,
            net=model,
            trainloader=trainloaders[int(cid)],  # Client-specific training data
            valloader=valloaders[int(cid)],      # Client-specific validation data
            device=device,
            batch_size=config.get('batch_size', 32),
            save_results=f"{result_base_path}/client_{cid}/",
            matrix_path="confusion_matrix.png",
            roc_path="roc.png",
            yaml_path=None,
            classes=class_names
        )
    
    return client_fn


def create_evaluate_fn(
    config: Dict[str, Any],
    model_params: Dict[str, Any] = None,
    result_base_path: str = "results",
    dataset_params: Optional[Dict[str, Any]] = None,
    custom_model: Optional[torch.nn.Module] = None
) -> Callable:
    """
    Generate an evaluate_fn for federated learning server-side evaluation.
    
    This convenience function creates a standardized evaluate_fn that follows
    the common pattern used across all Fed-ML-Chem federated examples.
    
    Args:
        config: Configuration dictionary from create_config()
        model_params: Parameters for create_model() (architecture, quantum settings, etc.)
        result_base_path: Base path for saving server results
        dataset_params: Optional dataset loading parameters (splitter, etc.)
        
    Returns:
        An evaluate_fn function ready for use in FedAvg strategy
        
    Example:
        ```python
        model_params = {
            'base_architecture': 'mlp',
            'use_quantum': True,
            'n_qubits': 7,
            'quantum_circuit': 'basic_entangler'
        }
        
        evaluate_fn = create_evaluate_fn(
            config=config,
            model_params=model_params,
            result_base_path="results/quantum_federated_dna"
        )
        ```
    """
    from ..data.loaders import load_datasets, infer_dataset_properties
    from ..models.modular import create_model
    from ..core.testing import test
    from ..core.visualization import save_all_results

    
    # Set default parameters
    if dataset_params is None:
        dataset_params = {}
    if model_params is None:
        model_params = {}
    
    def evaluate_fn(server_round, parameters, config_round):
        device = torch.device('cuda' if config.get('gpu', False) else 'cpu')
        
        # Load test dataset for server evaluation
        _, _, testloader = load_datasets(
            num_clients=1,
            batch_size=config.get('batch_size', 32),
            resize=dataset_params.get('resize', 224) if dataset_params.get('resize') is not None else 224,
            seed=dataset_params.get('seed', 42),
            num_workers=dataset_params.get('num_workers', 0),
            splitter=dataset_params.get('splitter', 10),
            dataset=config.get('dataset', 'DNA'),
            data_path=dataset_params.get('data_path', "data/"),
            custom_normalizations=dataset_params.get('custom_normalizations')
        )
        
        # Get dataset properties
        input_shape, num_classes = infer_dataset_properties(testloader)
        
        # Create model for evaluation using either custom model or library's modular system
        if custom_model is not None:
            import copy
            model = copy.deepcopy(custom_model).to(device)
        else:
            model = create_model(
                input_shape=input_shape,
                num_classes=num_classes,
                **model_params
            ).to(device)
        
        # Load aggregated parameters from federated training
        set_parameters(model, parameters)
        
        # Evaluate global model
        loss, accuracy, y_pred, y_true, y_proba = test(
            model=model,
            dataloader=testloader,
            loss_fn=torch.nn.CrossEntropyLoss(),
            device=device
        )
        
        # Save server evaluation results using save_all_results
        os.makedirs(f"{result_base_path}/server/", exist_ok=True)
        class_names = [str(i) for i in range(num_classes)]
        
        # For server evaluation, we only have test data, so we use test data for both train and test
        # and provide empty training history since this is just evaluation
        save_all_results(
            train_true=y_true,  # Use test data as placeholder for train data
            train_pred=y_pred, 
            train_proba=y_proba,
            test_true=y_true,
            test_pred=y_pred,
            test_proba=y_proba,
            training_history={'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []},  # Empty history for server eval
            classes=class_names,
            results_path=f"{result_base_path}/server",
            config=config,
            file_suffix="_server_eval"
        )
        
        # Create appropriate prefix for output
        prefix = "Quantum " if model_params.get('use_quantum', False) else ""
        print(f"{prefix}Server evaluation - Round {server_round}: Loss {loss:.4f}, Accuracy {accuracy:.2f}%")
        
        return loss, {"accuracy": accuracy}
    
    return evaluate_fn


def run_federated_simulation(
    config: Dict[str, Any],
    model_params: Dict[str, Any] = None,
    num_clients: int = 3,
    num_rounds: int = 3,
    result_base_path: str = None,
    custom_data_loader: Optional[Callable] = None,
    dataset_params: Optional[Dict[str, Any]] = None,
    custom_model: Optional[torch.nn.Module] = None
) -> None:
    """
    Run complete federated learning simulation with all boilerplate handled.
    
    This eliminates ~85 lines of repetitive code from every federated example.
    Supports both built-in and custom dataset loading functions, and custom models.
    
    Args:
        config: Configuration dictionary from create_config()
        model_params: Parameters for create_model() (architecture, quantum settings, etc.)
                     Ignored if custom_model is provided.
        num_clients: Number of federated clients
        num_rounds: Number of federated training rounds
        result_base_path: Base path for saving results
        custom_data_loader: Optional custom function that returns (trainloaders, valloaders, testloader)
        dataset_params: Optional dataset loading parameters
        custom_model: Optional custom PyTorch model template. If provided, model_params is ignored.
        
    Example:
        ```python
        # Simple usage with built-in datasets
        run_federated_simulation(
            config=config,
            model_params={'base_architecture': 'mlp', 'hidden_dims': [64, 32]},
            num_clients=3,
            num_rounds=5
        )
        
        # Advanced usage with custom dataset
        def my_custom_loader(config, dataset_params):
            # Your custom loading logic
            return trainloaders, valloaders, testloader
            
        run_federated_simulation(
            config=config,
            model_params=model_params,
            custom_data_loader=my_custom_loader
        )
        ```
    """
    import flwr as fl
    from .strategies import create_fedavg_strategy
    from ..data.loaders import load_datasets, infer_dataset_properties
    from ..models.modular import create_model
    
    # Auto-generate result path if not provided
    if result_base_path is None:
        result_base_path = f"results/federated_{config['dataset'].lower()}_example"
    
    # Set default dataset parameters
    if dataset_params is None:
        dataset_params = {}
    
    # Device setup
    device = torch.device('cuda' if config['gpu'] else 'cpu')
    
    # Load data for initial model creation
    if custom_data_loader is not None:
        print("Using custom dataset loader for initial model...")
        trainloaders, valloaders, testloader = custom_data_loader(config, dataset_params)
        # For custom loaders, use the testloader directly
        initial_testloader = testloader
    else:
        print("Loading dataset for initial model creation...")
        _, _, initial_testloader = load_datasets(
            num_clients=1,
            batch_size=32,
            resize=dataset_params.get('resize', 224) if dataset_params.get('resize') is not None else 224,
            seed=dataset_params.get('seed', 42),
            num_workers=dataset_params.get('num_workers', 0),
            splitter=dataset_params.get('splitter', 10),
            dataset=config['dataset'],
            data_path=dataset_params.get('data_path', "data/"),
            custom_normalizations=dataset_params.get('custom_normalizations')
        )
    
    # Get dataset properties for initial model
    input_shape, num_classes = infer_dataset_properties(initial_testloader)
    
    # Create initial global model for parameter initialization
    initial_model = create_model(
        input_shape=input_shape,
        num_classes=num_classes,
        **model_params
    ).to(device)
    
    # Set default parameters
    if model_params is None:
        model_params = {}
    
    # Create client and evaluation functions
    client_fn = create_client_fn(config, num_clients, model_params, result_base_path, dataset_params, custom_model)
    evaluate_fn = create_evaluate_fn(config, model_params, result_base_path, dataset_params, custom_model)
    
    # Create federated learning strategy
    strategy = create_fedavg_strategy(
        initial_model=initial_model,
        num_clients=num_clients,
        config=config,
        evaluate_fn=evaluate_fn,
        fraction_evaluate=1.0
    )
    
    # Configure client resources
    client_resources = {"num_cpus": 1}
    if device.type == "cuda":
        client_resources["num_gpus"] = 0.33  # Share GPU among clients
    
    print(f"Starting federated learning with {device}")
    print(f"Clients: {num_clients}, Rounds: {num_rounds}")
    start_time = time.time()
    
    # Start federated learning simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources
    )
    
    training_time = time.time() - start_time
    print(f"Federated training completed in {training_time:.2f} seconds")
    print(f"Results saved to: {result_base_path}")