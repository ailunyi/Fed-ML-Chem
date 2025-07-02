"""
Simple Centralized DNA Training Example using Fed-ML-Lib
"""
from fed_ml_lib import create_config, run_centralized_simulation

def main():
    """Main training function."""
    # Configuration using library's config system
    config = create_config(
        name="dna_centralized",
        dataset="DNA",
        model="mlp",
        epochs=25,
        learning_rate=0.001,
        batch_size=32,
        hidden_dims=[64, 32, 16, 8],
        dropout_rate=0.0,
        gpu=True,
        data_path="data/"
    )
    
    # Model parameters
    model_params = {
        'base_architecture': 'mlp',
        'hidden_dims': config['hidden_dims'],
        'dropout_rate': config['dropout_rate'],
        'domain': 'sequence'
    }
    
    # Run complete centralized training simulation
    results = run_centralized_simulation(
        config=config,
        model_params=model_params,
        result_base_path="results/centralized_dna_example",
        dataset_params={'resize': None}  # Not needed for DNA data
    )
    
    print(f"Final training accuracy: {results['train_accuracy']:.2f}%")
    print(f"Final test accuracy: {results['test_accuracy']:.2f}%")
    print(f"Training completed in {results['training_time']:.2f} seconds")

if __name__ == "__main__":
    main() 