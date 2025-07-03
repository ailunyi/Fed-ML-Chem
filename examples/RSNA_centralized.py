"""
Centralized RSNA Chest X-ray Training Example using Fed-ML-Lib
Following the research methodology: 70% train, 10% val, 20% test split
Images downscaled from 512x512 to 32x32 for efficient processing
Dataset: 14863 samples (normal=8851, lung opacity=6012)
"""
import wandb
from fed_ml_lib import create_config, run_centralized_simulation
import GPUtil
import psutil

def main():
    """Main training function for RSNA chest X-ray classification."""
    
    # Initialize wandb
    wandb.init(
        project="rsna-centralized-training",
        name="rsna_cnn_centralized",
        tags=["rsna", "chest-xray", "cnn", "centralized", "medical-imaging"],
        notes="Centralized CNN training on RSNA chest X-ray dataset with 32x32 resolution"
    )
    
    # Configuration following research paper methodology
    config = create_config(
        name="rsna_centralized_cnn",
        dataset="rsna_chestxray",
        model="cnn",
        epochs=25,  # More epochs for complex medical imaging
        learning_rate=0.001,
        batch_size=32,
        dropout_rate=0.1,
        gpu=True,
        data_path="../../../../anvil/projects/x-chm250024/data/"
    )
    
    # Model parameters for CNN architecture
    model_params = {
        'base_architecture': 'cnn',
        'conv_channels': [32],  # Single conv layer with 32 channels to match description
        'hidden_dims': [64],    # Minimal hidden layer to approximate simple architecture
        'dropout_rate': config['dropout_rate'],
        'domain': 'vision',
        'kernel_size': 2,       # 2×2 filter size as specified
        'pooling_type': 'max',  # Max pooling layer as specified
        'output_activation': 'sigmoid'  # Sigmoid output layer as specified
    }
    
    # Dataset parameters following research methodology
    dataset_params = {
        'resize': 32,  # Downscale from 512x512 to 32x32
        'train_split': 0.7,  # 70% for training
        'val_split': 0.1,    # 10% for validation  
        'test_split': 0.2,   # 20% for testing
        'seed': 42,
        'num_workers': 4,
        'data_path': config['data_path'],  # Use the Anvil path from config
        'custom_normalizations': {
            'rsna_chestxray': {
                'mean': (0.485, 0.456, 0.406),  # ImageNet values for medical imaging
                'std': (0.229, 0.224, 0.225)
            }
        }
    }
    
    # Log configuration to wandb
    wandb.config.update({
        # Training config
        "epochs": config['epochs'],
        "learning_rate": config['learning_rate'],
        "batch_size": config['batch_size'],
        "dropout_rate": config['dropout_rate'],
        "model_type": config['model'],
        "dataset": config['dataset'],
        "gpu_enabled": config['gpu'],
        
        # Model architecture
        "base_architecture": model_params['base_architecture'],
        "conv_channels": model_params['conv_channels'],
        "hidden_dims": model_params['hidden_dims'],
        "dropout_rate": model_params['dropout_rate'],
        "domain": model_params['domain'],
        "kernel_size": model_params['kernel_size'],
        "pooling_type": model_params['pooling_type'], 
        "output_activation": model_params['output_activation'],
        
        # Dataset config
        "image_resize": dataset_params['resize'],
        "train_split": dataset_params['train_split'],
        "val_split": dataset_params['val_split'],
        "test_split": dataset_params['test_split'],
        "random_seed": dataset_params['seed'],
        "num_workers": dataset_params['num_workers'],
        "normalization_mean": dataset_params['custom_normalizations']['rsna_chestxray']['mean'],
        "normalization_std": dataset_params['custom_normalizations']['rsna_chestxray']['std'],
        
        # Dataset statistics
        "total_samples": 14863,
        "normal_samples": 8851,
        "lung_opacity_samples": 6012,
        "original_image_size": "512x512",
        "processed_image_size": "32x32"
    })
    
    print(f"Loading dataset from: {config['data_path']}")
    print("Using custom RSNA normalization: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)")
    print("Starting centralized CNN training...")
    print(f"Tracking experiment on wandb: {wandb.run.url}")
    
    # This will show what wandb can actually monitor
    print("Available GPUs to wandb:", GPUtil.getGPUs())
    print("GPU count visible:", len(GPUtil.getGPUs()))
    
    # Run complete centralized training simulation
    results = run_centralized_simulation(
        config=config,
        model_params=model_params,
        result_base_path="results/centralized_rsna_cnn2",
        dataset_params=dataset_params
    )
    
    # Log final results to wandb
    wandb.log({
        "final_train_accuracy": results['train_accuracy'],
        "final_test_accuracy": results['test_accuracy'],
        "training_time_seconds": results['training_time'],
        "training_time_minutes": results['training_time'] / 60
    })
    
    # Log summary metrics
    wandb.summary["best_train_accuracy"] = results['train_accuracy']
    wandb.summary["best_test_accuracy"] = results['test_accuracy']
    wandb.summary["total_training_time"] = results['training_time']
    
    print("=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"Final training accuracy: {results['train_accuracy']:.2f}%")
    print(f"Final test accuracy: {results['test_accuracy']:.2f}%")
    print(f"Training completed in {results['training_time']:.2f} seconds")
    print(f"View results at: {wandb.run.url}")
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main() 