"""
Federated DNA Training Example using Fed-ML-Lib
"""
from fed_ml_lib import create_config, run_federated_simulation

def main():
    """Main federated learning orchestration."""
    # Global configuration
    config = create_config(
        name="dna_federated",
        dataset="DNA",
        model="mlp",
        epochs=5,
        learning_rate=0.001,
        batch_size=32,
        hidden_dims=[64, 32, 16, 8],
        dropout_rate=0.0,
        gpu=True
    )
    
    # Model parameters
    model_params = {
        'base_architecture': 'mlp',
        'hidden_dims': config['hidden_dims'],
        'dropout_rate': config['dropout_rate'],
        'domain': 'sequence'
    }
    
    # Run complete federated learning simulation
    run_federated_simulation(
        config=config,
        model_params=model_params,
        num_clients=3,
        num_rounds=3,
        result_base_path="results/federated_dna_example",
        dataset_params={'resize': None}  # Not needed for DNA data
    )

if __name__ == "__main__":
    main() 