"""
Simple Centralized DNA Training Example
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import os
import json

from fed_ml_lib.config import dna_mlp
from fed_ml_lib.models import create_model
from fed_ml_lib.data.loaders import infer_dataset_properties
from fed_ml_lib.core.visualization import save_all_results
from fed_ml_lib.core.testing import test
from fed_ml_lib.core.training import train

def read_and_prepare_data(file_path, seed, size=6, max_samples=5000):
    """
    Reads DNA sequence data from a text file and prepares it for modeling using k-mers and TF-IDF.
    """
    # Read data from file with limited samples to prevent memory issues
    data = pd.read_table(file_path, nrows=max_samples)
    print(f"Loaded {len(data)} DNA sequences from {file_path}")

    # Function to extract k-mers from a sequence
    def getKmers(sequence):
        return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

    # Function to preprocess data
    def preprocess_data(data):
        data['words'] = data['sequence'].apply(lambda x: getKmers(x))
        data = data.drop('sequence', axis=1)
        return data

    # Preprocess data
    data = preprocess_data(data)

    def kmer_lists_to_texts(kmer_lists):
        return [' '.join(map(str, l)) for l in kmer_lists]

    data['texts'] = kmer_lists_to_texts(data['words'])

    # Prepare data for modeling
    texts = data['texts'].tolist()
    y_data = data.iloc[:, 0].values
    model = TfidfVectorizer(ngram_range=(5,5))
    embeddings = model.fit_transform(texts).toarray()
    X_train, X_test, y_train, y_test = train_test_split(embeddings, y_data, test_size=0.2, random_state=seed)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    trainset = TensorDataset(X_train, y_train)
    testset = TensorDataset(X_test, y_test)
    return trainset, testset

def load_dna_data(data_path='data/', seed=42):
    """Load and preprocess DNA data."""
    try:
        trainset, testset = read_and_prepare_data(data_path + 'DNA/human.txt', seed)
        
        # Get number of classes from the data
        train_labels = [trainset[i][1].item() for i in range(len(trainset))]
        test_labels = [testset[i][1].item() for i in range(len(testset))]
        all_labels = set(train_labels + test_labels)
        num_classes = len(all_labels)
        
        return trainset, testset, num_classes
    
    except FileNotFoundError:
        print("DNA data not found at data/DNA/human.txt")
        print("Please download the DNA dataset and place human.txt in data/DNA/ folder")
        return None, None, None

def create_data_loaders(trainset, testset, batch_size=32):
    """Create train and test data loaders."""
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def test_and_return_results(model, data_loader, device):
    """Test the model and return results."""
    loss, accuracy, y_pred, y_true, y_proba = test(
        model=model,
        dataloader=data_loader,
        loss_fn=nn.CrossEntropyLoss(),
        device=device
    )
    
    return loss, accuracy, y_pred, y_true, y_proba

def main():
    """Main training function."""
    # Configuration
    config = dna_mlp("dna_centralized", epochs=25, learning_rate=0.001, batch_size=32)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    trainset, testset, num_classes = load_dna_data()
    if trainset is None:
        return
        
    train_loader, test_loader = create_data_loaders(trainset, testset, batch_size=config['batch_size'])
    
    # Get input shape from data
    input_shape, _ = infer_dataset_properties(train_loader)
    
    print(f"Dataset loaded: {len(trainset)} train samples, {len(testset)} test samples")
    print(f"Number of classes: {num_classes}")
    print(f"Input shape: {input_shape}")
    
    # Create model
    model = create_model(
        base_architecture="mlp",
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_dims=[64, 32, 16, 8],  # Match the legacy architecture
        dropout_rate=0.0,
        use_fhe=False,
        use_quantum=False
    )
    
    # Move model to device
    model = model.to(device)
    print(f"Model moved to: {device}")
    
    # Setup optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_fn = nn.CrossEntropyLoss()
    
    # Create results directory
    os.makedirs("results/centralized_dna_example", exist_ok=True)
    
    # Train model using library's training function
    print("Starting training...")
    results = train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=config['epochs'],
        device=device
    )
    
    # Test on training data 
    train_loss, train_accuracy, train_pred, train_true, train_proba = test_and_return_results(model, train_loader, device)
    
    # Test on test data
    test_loss, test_accuracy, test_pred, test_true, test_proba = test_and_return_results(model, test_loader, device)
    
    # Save all visualization results using convenience function
    class_names = [str(i) for i in range(num_classes)]
    
    save_all_results(
        train_true=train_true,
        train_pred=train_pred,
        train_proba=train_proba,
        test_true=test_true,
        test_pred=test_pred,
        test_proba=test_proba,
        training_history=results,
        classes=class_names,
        results_path="results/centralized_dna_example",
        config=config,
        file_suffix="_DNA_centralized"
    )
    
    print(f"Training accuracy: {train_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    print("Training plots and confusion matrices saved to results/centralized_dna_example/ folder")

if __name__ == "__main__":
    main() 