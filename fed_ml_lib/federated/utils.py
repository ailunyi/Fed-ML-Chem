import os
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
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
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def evaluate2(server_round: int, parameters: NDArrays,
              config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    set_parameters(central, parameters)
    loss, accuracy, y_pred, y_true, y_proba = engine.test(central, testloader, loss_fn=torch.nn.CrossEntropyLoss(),
                                                          device=DEVICE)
    os.makedirs(save_results, exist_ok=True)
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