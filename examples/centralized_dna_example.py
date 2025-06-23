"""
Simple Centralized DNA Training Example
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

from fed_ml_lib.config import dna_mlp
from fed_ml_lib.models import create_classical_model

def load_dna_data():
    """Load and preprocess DNA data."""
    # Load DNA dataset (assuming it's in data/DNA/)
    try:
        df = pd.read_csv('data/DNA/human_data.txt', sep='\t')
        
        # Encode DNA sequences to numerical format
        sequences = df['sequence'].values
        labels = df['class'].values
        
        # Simple DNA encoding (A=0, T=1, G=2, C=3)
        def encode_sequence(seq):
            mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
            return [mapping.get(base, 0) for base in seq]
        
        # Encode all sequences
        encoded_sequences = []
        for seq in sequences:
            encoded = encode_sequence(seq)
            # Pad or truncate to fixed length (180)
            if len(encoded) < 180:
                encoded.extend([0] * (180 - len(encoded)))
            else:
                encoded = encoded[:180]
            encoded_sequences.append(encoded)
        
        # Encode labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        X = np.array(encoded_sequences, dtype=np.float32)
        y = np.array(encoded_labels, dtype=np.int64)
        
        return X, y, len(label_encoder.classes_)
    
    except FileNotFoundError:
        # Generate synthetic DNA data for demo
        np.random.seed(42)
        X = np.random.randint(0, 4, size=(1000, 180)).astype(np.float32)
        y = np.random.randint(0, 7, size=1000).astype(np.int64)
        return X, y, 7

def create_data_loaders(X, y, batch_size=32, test_size=0.2):
    """Create train and test data loaders."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, device, epochs=25, lr=0.001):
    """Train the model."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
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
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        
        if epoch % 5 == 0:
            print(f'Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%')

def test_model(model, test_loader, device):
    """Test the model."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    
    print(f'Test Results: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%')
    return accuracy

def main():
    """Main training function."""
    # Configuration
    config = dna_mlp("dna_centralized", epochs=25, learning_rate=0.001, batch_size=32)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    X, y, num_classes = load_dna_data()
    train_loader, test_loader = create_data_loaders(X, y, batch_size=config['batch_size'])
    
    # Create model
    model = create_classical_model(
        architecture="mlp",
        input_shape=(180,),
        num_classes=num_classes,
        hidden_dims=[128, 64, 32],
        dropout_rate=0.2
    )
    
    # Train model
    train_model(model, train_loader, device, epochs=config['epochs'], lr=config['learning_rate'])
    
    # Test model
    test_accuracy = test_model(model, test_loader, device)
    
    print(f"Final test accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main() 