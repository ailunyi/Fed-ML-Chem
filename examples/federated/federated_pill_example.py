"""
Federated PILL Training Example using Fed-ML-Lib
"""
from fed_ml_lib import create_config, run_federated_simulation

def main():
    """Main federated learning orchestration using working architecture."""
    # Use same config as working centralized example
    config = create_config(
        name="pill_federated",
        dataset="PILL",
        model="cnn",
        epochs=10,
        learning_rate=0.001,
        batch_size=32,
        data_path = "../data"
    )
    
    # Model parameters - same as centralized
    model_params = {
        'base_architecture': 'pretrained_cnn',
        'freeze_layers': 23,  # Freeze first 23 layers like in original
        'use_fhe': False,
        'use_quantum': False
    }
    
    # Dataset parameters for PILL
    dataset_params = {
        'resize': 224,  # Match centralized example
        'splitter': 20
    }
    
    print("="*50)
    print("FEDERATED LEARNING - PILL CLASSIFICATION")
    print("="*50)
    print(f"Clients: 4")
    print(f"Rounds: 20")
    print(f"Dataset: PILL (Binary Classification)")
    print(f"Architecture: Pretrained CNN (Library Modular)")
    print(f"Matching centralized_pill_example.py architecture")
    
    # Run complete federated learning simulation
    run_federated_simulation(
        config=config,
        model_params=model_params,
        num_clients=4,
        num_rounds=20,
        result_base_path="../results/federated_pill",
        dataset_params=dataset_params
    )
    
    print("Results saved to results/federated_pill/")

if __name__ == "__main__":
    main() 