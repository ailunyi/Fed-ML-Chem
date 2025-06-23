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
from fed_ml_lib.models import create_classical_model
from fed_ml_lib.core.visualization import save_graphs, save_matrix
from fed_ml_lib.core.testing import test

def read_and_prepare_data(file_path, seed, size=6):
    """
    Reads DNA sequence data from a text file and prepares it for modeling using k-mers and TF-IDF.
    """
    # Read data from file
    data = pd.read_table(file_path)

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

def train_model(model, train_loader, test_loader, device, epochs=25, lr=0.001):
    """Train the model and collect metrics for plotting."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Collect metrics for plotting
    results = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        train_accuracy = 100. * correct / total
        avg_train_loss = total_loss / len(train_loader)
        results["train_loss"].append(avg_train_loss)
        results["train_acc"].append(train_accuracy)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        val_accuracy = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(test_loader)
        results["val_loss"].append(avg_val_loss)
        results["val_acc"].append(val_accuracy)
        
        if epoch % 5 == 0:
            print(f'Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={train_accuracy:.2f}%, Val Loss={avg_val_loss:.4f}, Val Acc={val_accuracy:.2f}%')
    
    return results

def test_model(model, test_loader, device, num_classes):
    """Test the model using the built-in test function."""
    loss, accuracy, y_pred, y_true, y_proba = test(
        model=model,
        dataloader=test_loader,
        loss_fn=nn.CrossEntropyLoss(),
        device=device
    )
    
    print(f'Test Results: Loss={loss:.4f}, Accuracy={accuracy:.2f}%')
    
    # Save confusion matrix using the collected predictions
    class_names = [str(i) for i in range(num_classes)]
    save_matrix(y_true, y_pred, "results/confusion_matrix.png", class_names)
    
    return accuracy

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
    sample_input, _ = next(iter(train_loader))
    input_shape = (sample_input.shape[1],)  # TF-IDF feature dimension
    
    print(f"Dataset loaded: {len(trainset)} train samples, {len(testset)} test samples")
    print(f"Number of classes: {num_classes}")
    print(f"Input shape: {input_shape}")
    
    # Create model
    model = create_classical_model(
        architecture="mlp",
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_dims=[64, 32, 16, 8],  # Match the legacy architecture
        dropout_rate=0.0
    )
    
    # Train model
    results = train_model(model, train_loader, test_loader, device, epochs=config['epochs'], lr=config['learning_rate'])
    
    # Test model
    test_accuracy = test_model(model, test_loader, device, num_classes)
    
    # Save training curves to results folder
    save_graphs("results/", config['epochs'], results, "_DNA_centralized")
    
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print("Training plots and confusion matrix saved to results/ folder")

if __name__ == "__main__":
    main() 