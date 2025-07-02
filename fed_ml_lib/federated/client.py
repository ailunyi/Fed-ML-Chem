import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any, Union
import flwr as fl
from flwr.common import NDArrays, Scalar
import numpy as np
import tenseal as ts

from fed_ml_lib.federated.utils import *
from fed_ml_lib.core.training import train
from fed_ml_lib.core.testing import test, test_multimodal_health
from fed_ml_lib.core.visualization import save_matrix, save_roc, save_graphs, save_graphs_multimodal
from fed_ml_lib.core import security

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, device, batch_size, save_results, matrix_path, roc_path,
                 yaml_path, classes):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.cid = cid
        self.device = device
        self.batch_size = batch_size
        self.save_results = save_results
        self.matrix_path = matrix_path
        self.roc_path = roc_path
        self.yaml_path = yaml_path
        self.classes = classes

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters2(self.net)

    def fit(self, parameters, config):
        server_round = config['server_round']
        local_epochs = config['local_epochs']
        lr = float(config["learning_rate"])

        print(f'[Client {self.cid}, round {server_round}] fit, config: {config}')

        set_parameters(self.net, parameters)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        results = train(self.net, self.trainloader, self.valloader, optimizer=optimizer, loss_fn=criterion,
                        epochs=local_epochs, device=self.device)

        if self.save_results:
            save_graphs(self.save_results, local_epochs, results, f"_Client {self.cid}")

        return get_parameters2(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)

        loss, accuracy, y_pred, y_true, y_proba = test(self.net, self.valloader,
                                                        loss_fn=torch.nn.CrossEntropyLoss(), device=self.device)

        if self.save_results:
            os.makedirs(self.save_results, exist_ok=True)
            if self.matrix_path:
                save_matrix(y_true, y_pred, self.save_results + self.matrix_path, self.classes)
            if self.roc_path:
                save_roc(y_true, y_proba, self.save_results + self.roc_path, len(self.classes))

        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
    
class MultimodalFlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, device, batch_size, save_results, matrix_path, roc_path,
                 yaml_path, classes):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.cid = cid
        self.device = device
        self.batch_size = batch_size
        self.save_results = save_results
        self.matrix_path = matrix_path
        self.roc_path = roc_path
        self.yaml_path = yaml_path
        self.classes = classes

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters2(self.net)

    def fit(self, parameters, config):
        server_round = config['server_round']
        local_epochs = config['local_epochs']
        lr = float(config["learning_rate"])

        print(f'[Client {self.cid}, round {server_round}] fit, config: {config}')

        set_parameters(self.net, parameters)

        criterion_mri = torch.nn.CrossEntropyLoss()
        criterion_dna = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        results = train(self.net, self.trainloader, self.valloader, optimizer=optimizer, loss_fn=(criterion_mri, criterion_dna),
                        epochs=local_epochs, device=self.device, task="Multimodal")

        if self.save_results:
            save_graphs_multimodal(self.save_results, local_epochs, results, f"_Client {self.cid}")

        return get_parameters2(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)

        loss, accuracy, y_pred, y_true, y_proba = test_multimodal_health(self.net, self.valloader,
                                                                      loss_fn=(torch.nn.CrossEntropyLoss(),torch.nn.CrossEntropyLoss()), device=self.device)

        loss_mri, loss_dna = loss
        accuracy_mri, accuracy_dna = accuracy
        y_pred_mri, y_pred_dna = y_pred
        y_true_mri, y_true_dna = y_true
        y_proba_mri, y_proba_dna = y_proba

        if self.save_results:
            os.makedirs(self.save_results, exist_ok=True)
            if self.matrix_path:
                save_matrix(y_true_mri, y_pred_mri, self.save_results + "MRI_" + self.matrix_path, self.classes[0])
                save_matrix(y_true_dna, y_pred_dna, self.save_results + "DNA_" + self.matrix_path, self.classes[1])
            if self.roc_path:
                save_roc(y_true_mri, y_proba_mri, self.save_results + "MRI_" + self.roc_path, len(self.classes[0]))
                save_roc(y_true_dna, y_proba_dna, self.save_results + "DNA_" + self.roc_path, len(self.classes[1]))

        return float(loss_mri), len(self.valloader), {"accuracy": float(accuracy_mri)}

class FHEFlowerClient(FlowerClient):
    """
    FHE-aware Flower client with built-in homomorphic encryption support.
    
    This client automatically handles encryption/decryption of model parameters
    during federated learning while maintaining compatibility with standard
    federated learning workflows.
    """
    
    def __init__(self, cid, net, trainloader, valloader, device, batch_size, 
                 save_results, matrix_path, roc_path, yaml_path, classes,
                 fhe_scheme: str = "CKKS", fhe_layers: Optional[List[str]] = None,
                 context_path: Optional[str] = None):
        """
        Initialize FHE-aware federated client.
        
        Args:
            fhe_scheme: FHE scheme to use ("CKKS", "BFV", "TFHE")
            fhe_layers: List of layer names to encrypt (default: ["classifier"])
            context_path: Path to load/save FHE context
        """
        super().__init__(cid, net, trainloader, valloader, device, batch_size,
                        save_results, matrix_path, roc_path, yaml_path, classes)
        
        self.fhe_scheme = fhe_scheme
        self.fhe_layers = fhe_layers if fhe_layers is not None else ["classifier"]
        self.context_path = context_path
        self.context_client = None
        
        self._setup_fhe_context()
    
    def _setup_fhe_context(self):
        """Setup FHE context for encryption/decryption operations."""
        try:
            if self.context_path and os.path.exists(self.context_path):
                # Load existing context
                _, context_data = security.read_query(self.context_path)
                self.context_client = ts.context_from(context_data)
                print(f"[Client {self.cid}] Loaded FHE context from {self.context_path}")
            else:
                # Create new context
                self.context_client = security.context()
                print(f"[Client {self.cid}] Created new FHE context")
                
                # Save context if path provided
                if self.context_path:
                    security.write_query(self.context_path, {
                        "contexte": self.context_client.serialize(save_secret_key=True)
                    })
                    
        except Exception as e:
            print(f"[Client {self.cid}] FHE setup failed: {e}")
            raise e
    
    def _should_encrypt_layer(self, layer_idx: int, layer_name: Optional[str] = None) -> bool:
        """Determine if a layer should be encrypted based on configuration."""
        if not self.context_client:
            return False
            
        # Check if all layers should be encrypted
        if "all" in self.fhe_layers:
            return True
            
        # Check if specific layer name matches
        if layer_name and layer_name in self.fhe_layers:
            return True
            
        # Check if it's the last layer (classifier) and classifier is in fhe_layers
        if "classifier" in self.fhe_layers:
            # Assume last layer is classifier
            total_params = len(list(self.net.parameters()))
            return layer_idx >= total_params - 2  # Last weight and bias
            
        return False
    
    def get_parameters(self, config):
        """Get parameters with selective FHE encryption."""
        print(f"[Client {self.cid}] get_parameters (FHE enabled)")
        
        params = get_parameters2(self.net)
        
        if not self.context_client:
            return params
            
        try:
            # Encrypt specified layers
            encrypted_params = []
            for i, param in enumerate(params):
                if self._should_encrypt_layer(i):
                    # Encrypt this parameter
                    encrypted_tensor = ts.ckks_tensor(self.context_client, param)
                    encrypted_params.append(encrypted_tensor.serialize())
                    print(f"[Client {self.cid}] Encrypted parameter {i}")
                else:
                    # Keep parameter unencrypted
                    encrypted_params.append(param)
                    
            return encrypted_params
            
        except Exception as e:
            print(f"[Client {self.cid}] FHE encryption failed: {e}")
            raise e
    
    def fit(self, parameters, config):
        """Training with FHE parameter decryption."""
        server_round = config['server_round']
        print(f'[Client {self.cid}, round {server_round}] fit (FHE enabled)')
        
        if self.context_client:
            try:
                # Decrypt parameters
                decrypted_params = []
                for i, param in enumerate(parameters):
                    if isinstance(param, bytes) and self._should_encrypt_layer(i):
                        # Decrypt encrypted parameter
                        encrypted_tensor = ts.ckks_tensor_from(self.context_client, param)
                        decrypted_param = encrypted_tensor.decrypt()
                        decrypted_params.append(decrypted_param)
                        print(f"[Client {self.cid}] Decrypted parameter {i}")
                    else:
                        # Use parameter as-is
                        decrypted_params.append(param)
                        
                set_parameters(self.net, decrypted_params)
                
            except Exception as e:
                print(f"[Client {self.cid}] FHE decryption failed: {e}")
                raise e
        else:
            set_parameters(self.net, parameters)

        # Continue with normal training
        return super().fit(parameters, config)
    
    def evaluate(self, parameters, config):
        """Evaluation with FHE parameter decryption."""
        print(f"[Client {self.cid}] evaluate (FHE enabled)")
        
        if self.context_client:
            try:
                # Decrypt parameters for evaluation
                decrypted_params = []
                for i, param in enumerate(parameters):
                    if isinstance(param, bytes) and self._should_encrypt_layer(i):
                        encrypted_tensor = ts.ckks_tensor_from(self.context_client, param)
                        decrypted_param = encrypted_tensor.decrypt()
                        decrypted_params.append(decrypted_param)
                    else:
                        decrypted_params.append(param)
                        
                set_parameters(self.net, decrypted_params)
                
            except Exception as e:
                print(f"[Client {self.cid}] FHE decryption failed: {e}")
                raise e
        else:
            set_parameters(self.net, parameters)
            
        # Continue with normal evaluation
        return super().evaluate(parameters, config)