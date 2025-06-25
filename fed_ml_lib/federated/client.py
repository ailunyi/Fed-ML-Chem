import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any, Union
import flwr as fl
from flwr.common import NDArrays, Scalar
import numpy as np

from fed_ml_lib.federated.utils import *
from fed_ml_lib.core.training import train
from fed_ml_lib.core.testing import test, test_multimodal_health
from fed_ml_lib.core.visualization import save_matrix, save_roc, save_graphs, save_graphs_multimodal

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