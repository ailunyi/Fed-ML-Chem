"""
Centralized DNA+MRI Training Example using Fed-ML-Lib's modular architecture
"""
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import time

from fed_ml_lib.models.modular import create_model
from fed_ml_lib.data.loaders import load_datasets
from fed_ml_lib.core.training import train
from fed_ml_lib.core.testing import test_multimodal_health
from fed_ml_lib.core.visualization import save_matrix, save_graphs_multimodal, save_roc
from fed_ml_lib.config.python_config import run_experiment

class MultimodalNet(nn.Module):
    """Multimodal network combining MRI and DNA models with attention."""
    
    def __init__(self, mri_model, dna_model, expert_vector=6):
        super(MultimodalNet, self).__init__()
        self.mri_net = mri_model
        self.dna_net = dna_model
        
        self.feature_dim = expert_vector
        self.num_heads = expert_vector
        self.num_of_expert = 2
        
        # Attention and fusion layers
        self.attention = nn.MultiheadAttention(
            embed_dim=self.num_of_expert*self.feature_dim, 
            num_heads=self.num_heads,
            batch_first=True  # Modern PyTorch convention
        )
        self.fc_gate = nn.Linear(self.num_of_expert*self.feature_dim, 2)
        self.fc2_mri = nn.Linear(self.feature_dim, 4)  # 4 MRI classes
        self.fc2_dna = nn.Linear(self.feature_dim, 7)  # 7 DNA classes
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc_gate.weight)
        nn.init.xavier_uniform_(self.fc2_mri.weight)
        nn.init.xavier_uniform_(self.fc2_dna.weight)

    def forward(self, mri_input, dna_input):
        """Forward pass handling both modalities.
        
        Args:
            mri_input: MRI data tensor
            dna_input: DNA data tensor
        Returns:
            Tuple of (mri_output, dna_output)
        """
        # Get features from individual networks
        mri_features = self.mri_net(mri_input)
        dna_features = self.dna_net(dna_input)
        
        # Combine features
        combined_features = torch.cat((mri_features, dna_features), dim=1)
        combined_features = combined_features.unsqueeze(1)  # Add sequence length dimension
        
        # Apply attention
        attn_output, _ = self.attention(
            combined_features, combined_features, combined_features
        )
        attn_output = attn_output.squeeze(1)
        
        # Apply gating mechanism
        gates = F.softmax(self.fc_gate(attn_output), dim=1)
        combined_output = (
            gates[:, 0].unsqueeze(1) * mri_features + 
            gates[:, 1].unsqueeze(1) * dna_features
        )
        
        # Get final outputs
        mri_output = self.fc2_mri(combined_output)
        dna_output = self.fc2_dna(combined_output)
        
        return mri_output, dna_output

def main():
    # Configuration
    config = run_experiment(
        name="dna_mri_centralized",
        dataset="DNA+MRI",
        model="multimodal",
        epochs=25,
        learning_rate=0.0002,
        batch_size=32,
        hidden_dims=[64, 32, 16, 8],  # For DNA model
        dropout_rate=0.2,
        gpu=torch.cuda.is_available()
    )
    
    device = torch.device('cuda' if config['gpu'] else 'cpu')
    
    print(f"Training on {device}")
    start_time = time.time()
    
    # Load data
    trainloaders, valloaders, testloader = load_datasets(
        num_clients=1,
        batch_size=config['batch_size'],
        resize=224,
        seed=42,
        num_workers=0,
        splitter=10,
        dataset='DNA+MRI',
        data_path="data/"
    )
    train_loader = trainloaders[0]
    val_loader = valloaders[0]
    
    # Get DNA input size from the data
    _, input_sp = next(iter(testloader))[0][1].shape
    
    # Create MRI model
    mri_model = create_model(
        base_architecture='pretrained_cnn',
        input_shape=(3, 224, 224),
        num_classes=6,  # Expert vector size
        freeze_layers=23,
        use_pretrained=True,
        domain='vision'
    )
    
    # Create DNA model with correct input size
    dna_model = create_model(
        base_architecture='mlp',
        input_shape=(input_sp,),  # Use actual DNA input size
        num_classes=6,  # Expert vector size
        hidden_dims=config['hidden_dims'],
        dropout_rate=config['dropout_rate'],
        domain='sequence'
    )
    
    # Create multimodal model
    model = MultimodalNet(mri_model, dna_model).to(device)
    
    # Loss and optimizer
    criterion_mri = nn.CrossEntropyLoss()
    criterion_dna = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Create results directory
    os.makedirs("results/centralized_dna_mri_example", exist_ok=True)
    
    # Training loop
    training_history = train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=val_loader,
        optimizer=optimizer,
        loss_fn=(criterion_mri, criterion_dna),
        epochs=config['epochs'],
        device=device,
        task="multimodal_health"
    )
    
    # Test on training data
    train_loss, train_acc, train_pred, train_true, train_proba = test_multimodal_health(
        model=model,
        dataloader=train_loader,
        loss_fn=(criterion_mri, criterion_dna),
        device=device
    )
    
    # Test on test data
    test_loss, test_acc, test_pred, test_true, test_proba = test_multimodal_health(
        model=model,
        dataloader=testloader,
        loss_fn=(criterion_mri, criterion_dna),
        device=device
    )
    
    # Unpack results
    (train_loss_mri, train_loss_dna) = train_loss
    (train_acc_mri, train_acc_dna) = train_acc
    (train_pred_mri, train_pred_dna) = train_pred
    (train_true_mri, train_true_dna) = train_true
    (train_proba_mri, train_proba_dna) = train_proba
    
    (test_loss_mri, test_loss_dna) = test_loss
    (test_acc_mri, test_acc_dna) = test_acc
    (test_pred_mri, test_pred_dna) = test_pred
    (test_true_mri, test_true_dna) = test_true
    (test_proba_mri, test_proba_dna) = test_proba
    
    # Save MRI results
    save_matrix(
        y_true=train_true_mri,
        y_pred=train_pred_mri,
        classes=['glioma', 'meningioma', 'notumor', 'pituitary'],
        path="results/centralized_dna_mri_example/confusion_matrix_mri_train.png"
    )
    
    save_matrix(
        y_true=test_true_mri,
        y_pred=test_pred_mri,
        classes=['glioma', 'meningioma', 'notumor', 'pituitary'],
        path="results/centralized_dna_mri_example/confusion_matrix_mri_test.png"
    )
    
    save_roc(
        targets=train_true_mri,
        y_proba=train_proba_mri,
        path="results/centralized_dna_mri_example/roc_mri_train.png",
        nbr_classes=4
    )
    
    save_roc(
        targets=test_true_mri,
        y_proba=test_proba_mri,
        path="results/centralized_dna_mri_example/roc_mri_test.png",
        nbr_classes=4
    )
    
    # Save DNA results
    save_matrix(
        y_true=train_true_dna,
        y_pred=train_pred_dna,
        classes=['0', '1', '2', '3', '4', '5', '6'],
        path="results/centralized_dna_mri_example/confusion_matrix_dna_train.png"
    )
    
    save_matrix(
        y_true=test_true_dna,
        y_pred=test_pred_dna,
        classes=['0', '1', '2', '3', '4', '5', '6'],
        path="results/centralized_dna_mri_example/confusion_matrix_dna_test.png"
    )
    
    save_roc(
        targets=train_true_dna,
        y_proba=train_proba_dna,
        path="results/centralized_dna_mri_example/roc_dna_train.png",
        nbr_classes=7
    )
    
    save_roc(
        targets=test_true_dna,
        y_proba=test_proba_dna,
        path="results/centralized_dna_mri_example/roc_dna_test.png",
        nbr_classes=7
    )
    
    # Save training curves
    save_graphs_multimodal(
        path_save="results/centralized_dna_mri_example/",
        local_epoch=config['epochs'],
        results=training_history,
        end_file=""
    )
    
    print(f"MRI Training accuracy: {train_acc_mri:.2f}%")
    print(f"MRI Test accuracy: {test_acc_mri:.2f}%")
    print(f"DNA Training accuracy: {train_acc_dna:.2f}%")
    print(f"DNA Test accuracy: {test_acc_dna:.2f}%")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 