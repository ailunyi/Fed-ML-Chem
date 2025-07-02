from typing import Optional, Callable, Dict, Any
import flwr as fl
from flwr.server.strategy import FedAvg
from fed_ml_lib.federated.utils import get_parameters2


def create_fedavg_strategy(
    initial_model,
    num_clients: int,
    config: Dict[str, Any],
    evaluate_fn: Optional[Callable] = None,
    fraction_evaluate: float = 1.0,
    num_rounds: int = 5
) -> FedAvg:
   
    # Calculate minimum clients based on fraction_evaluate
    min_evaluate_clients = max(1, int(num_clients * fraction_evaluate))
    
    # Create client configuration function
    def on_fit_config_fn(round_num: int) -> Dict[str, str]:
        """Send configuration to clients each round."""
        return {
            "learning_rate": str(config.get('learning_rate', 0.001)),
            "batch_size": str(config.get('batch_size', 32)),
            "server_round": str(round_num),
            "local_epochs": str(config.get('epochs', 5))
        }
    
    # Extract initial parameters from model
    initial_parameters = fl.common.ndarrays_to_parameters(get_parameters2(initial_model))
    
    # Create FedAvg strategy with common configuration
    strategy = FedAvg(
        fraction_fit=1.0,                           # Use all available clients for training
        fraction_evaluate=fraction_evaluate,        # Configurable evaluation fraction
        min_fit_clients=num_clients,                # Minimum clients needed for training round
        min_evaluate_clients=min_evaluate_clients,  # Minimum clients needed for evaluation round
        min_available_clients=num_clients,          # Minimum clients that must be available
        initial_parameters=initial_parameters,      # Initial model parameters
        evaluate_fn=evaluate_fn,                    # Server-side evaluation function
        on_fit_config_fn=on_fit_config_fn          # Configuration sent to clients each round
    )
    
    return strategy


 