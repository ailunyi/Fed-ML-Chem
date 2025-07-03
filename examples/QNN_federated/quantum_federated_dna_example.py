"""
Quantum Federated DNA Training Example using Fed-ML-Lib and PennyLane
"""
from fed_ml_lib import create_config, run_federated_simulation

def main():
    """Main quantum federated learning orchestration"""
    config = create_config(
        name="quantum_dna_federated",
        dataset="DNA",
        model="quantum_mlp",
        epochs=2,
        learning_rate=0.001,
        batch_size=32,
        gpu=True
    )
    
    # Define quantum model parameters - quantum processing enabled
    model_params = {
        'base_architecture': 'mlp',
        'use_quantum': True,                    # Enable quantum processing
        'hidden_dims': [1024, 512, 256, 128],   # Classical hidden layers  
        'dropout_rate': 0.0,
        'domain': 'sequence',
        'n_qubits': 7,                          # Quantum circuit parameters
        'n_layers': 7,
        'quantum_circuit': "basic_entangler",
        'quantum_layers': ['classifier']        # Apply quantum to classifier
    }
    
    # Dataset parameters for quantum example
    dataset_params = {
        'splitter': 20,
        'resize': None,  # Not needed for DNA
        'seed': 42
    }
    
    print("="*60)
    print("QUANTUM FEDERATED LEARNING - DNA CLASSIFICATION")
    print("="*60)
    print(f"Qubits: {model_params['n_qubits']}")
    print(f"Quantum Circuit: {model_params['quantum_circuit']}")
    print(f"Clients: 4")
    print("="*60)
    
    # Run complete quantum federated learning simulation
    run_federated_simulation(
        config=config,
        model_params=model_params,
        num_clients=4,
        num_rounds=3,
        result_base_path="results/quantum_federated_dna2",
        dataset_params=dataset_params
    )

if __name__ == "__main__":
    main() 