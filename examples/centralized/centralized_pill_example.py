"""
Simple Centralized PILL Training Example using Fed-ML-Lib
"""
from fed_ml_lib import create_config, run_centralized_simulation

def main():
    config = create_config(
        name="pill_centralized",
        dataset="PILL",
        model="cnn",
        epochs=25,
        learning_rate=0.0002,
        batch_size=32
    )
    
    # Model parameters
    model_params = {
        'base_architecture': 'pretrained_cnn',
        'freeze_layers': 23,  # Freeze first 23 layers like in original
        'use_fhe': False,
        'use_quantum': False
    }
    
    # Run complete centralized training simulation
    results = run_centralized_simulation(
        config=config,
        model_params=model_params,
        result_base_path="results/centralized_pill_example",
        dataset_params={'resize': 224},  # Match legacy size
        class_names=['bad', 'good']
    )
    
    print(f"Training accuracy: {results['train_accuracy']:.2f}%")
    print(f"Test accuracy: {results['test_accuracy']:.2f}%")
    print(f"Training completed in {results['training_time']:.2f} seconds")

if __name__ == "__main__":
    main() 