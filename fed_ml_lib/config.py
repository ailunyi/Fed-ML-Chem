"""
config.py
---------
This module contains configuration parsing and management utilities for the fed_ml_lib library.
"""

import yaml
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import os


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    model_type: str = 'vgg16'  # 'vgg16', 'quantum', 'hybrid_cnn_qnn', 'vqc'
    num_classes: int = 2
    n_qubits: Optional[int] = 4
    n_layers: Optional[int] = 2
    input_size: Optional[int] = None


@dataclass
class DataConfig:
    """Configuration for dataset parameters."""
    dataset_name: str = 'PILL'
    data_path: str = './data/'
    batch_size: int = 32
    resize: Optional[int] = 224
    val_split: float = 0.1
    seed: int = 42
    num_workers: int = 0


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    epochs: int = 25
    learning_rate: float = 2e-4
    optimizer: str = 'adam'  # 'adam', 'sgd', 'adamw'
    loss_function: str = 'crossentropy'
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    results_dir: str = './results/'
    plot_results: bool = True
    save_model: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    experiment_name: str = 'default_experiment'
    description: str = ''


class ConfigManager:
    """Manages configuration loading, saving, and validation."""
    
    @staticmethod
    def create_default_configs() -> Dict[str, ExperimentConfig]:
        """Create default configurations for different scenarios."""
        
        configs = {}
        
        # Classical VGG16 configuration
        configs['classical_vgg16'] = ExperimentConfig(
            model=ModelConfig(
                model_type='vgg16',
                num_classes=2
            ),
            data=DataConfig(
                dataset_name='PILL',
                batch_size=32,
                resize=224
            ),
            training=TrainingConfig(
                epochs=25,
                learning_rate=2e-4
            ),
            experiment_name='classical_vgg16',
            description='Classical VGG16 model for image classification'
        )
        
        # Hybrid CNN-Quantum configuration
        configs['hybrid_cnn_qnn'] = ExperimentConfig(
            model=ModelConfig(
                model_type='hybrid_cnn_qnn',
                num_classes=2,
                n_qubits=4,
                n_layers=2
            ),
            data=DataConfig(
                dataset_name='PILL',
                batch_size=16,  # Smaller batch for quantum
                resize=64       # Smaller images for quantum
            ),
            training=TrainingConfig(
                epochs=15,      # Fewer epochs for quantum
                learning_rate=1e-3
            ),
            experiment_name='hybrid_cnn_qnn',
            description='Hybrid CNN-Quantum model for image classification'
        )
        
        # Pure quantum configuration
        configs['quantum_net'] = ExperimentConfig(
            model=ModelConfig(
                model_type='quantum',
                num_classes=2,
                n_qubits=6,
                n_layers=3,
                input_size=784  # For flattened images
            ),
            data=DataConfig(
                dataset_name='PILL',
                batch_size=8,   # Very small batch for quantum
                resize=28       # Small images for quantum
            ),
            training=TrainingConfig(
                epochs=10,
                learning_rate=5e-4
            ),
            experiment_name='quantum_net',
            description='Pure quantum neural network'
        )
        
        # Variational Quantum Classifier
        configs['vqc'] = ExperimentConfig(
            model=ModelConfig(
                model_type='vqc',
                num_classes=2,
                n_qubits=4,
                n_layers=3,
                input_size=4
            ),
            data=DataConfig(
                dataset_name='PILL',
                batch_size=8,
                resize=32
            ),
            training=TrainingConfig(
                epochs=8,
                learning_rate=1e-3
            ),
            experiment_name='vqc',
            description='Variational Quantum Classifier'
        )
        
        return configs
    
    @staticmethod
    def save_config(config: ExperimentConfig, filepath: str) -> None:
        """Save configuration to file (YAML or JSON)."""
        config_dict = asdict(config)
        
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError("Unsupported file format. Use .yaml, .yml, or .json")
    
    @staticmethod
    def load_config(filepath: str) -> ExperimentConfig:
        """Load configuration from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError("Unsupported file format. Use .yaml, .yml, or .json")
        
        return ConfigManager._dict_to_config(config_dict)
    
    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Convert dictionary to ExperimentConfig."""
        model_config = ModelConfig(**config_dict['model'])
        data_config = DataConfig(**config_dict['data'])
        training_config = TrainingConfig(**config_dict['training'])
        
        return ExperimentConfig(
            model=model_config,
            data=data_config,
            training=training_config,
            experiment_name=config_dict.get('experiment_name', 'loaded_experiment'),
            description=config_dict.get('description', '')
        )
    
    @staticmethod
    def create_config_templates(output_dir: str = './configs/') -> None:
        """Create template configuration files."""
        os.makedirs(output_dir, exist_ok=True)
        
        default_configs = ConfigManager.create_default_configs()
        
        for name, config in default_configs.items():
            filepath = os.path.join(output_dir, f'{name}.yaml')
            ConfigManager.save_config(config, filepath)
            print(f"Created template: {filepath}")
    
    @staticmethod
    def validate_config(config: ExperimentConfig) -> bool:
        """Validate configuration parameters."""
        try:
            # Validate model config
            valid_model_types = ['vgg16', 'quantum', 'hybrid_cnn_qnn', 'vqc']
            if config.model.model_type not in valid_model_types:
                raise ValueError(f"Invalid model_type: {config.model.model_type}")
            
            if config.model.num_classes < 1:
                raise ValueError("num_classes must be >= 1")
            
            # Validate quantum parameters
            if config.model.model_type in ['quantum', 'hybrid_cnn_qnn', 'vqc']:
                if config.model.n_qubits is None or config.model.n_qubits < 1:
                    raise ValueError("n_qubits must be >= 1 for quantum models")
                if config.model.n_layers is None or config.model.n_layers < 1:
                    raise ValueError("n_layers must be >= 1 for quantum models")
            
            # Validate data config
            if config.data.batch_size < 1:
                raise ValueError("batch_size must be >= 1")
            if not 0 < config.data.val_split < 1:
                raise ValueError("val_split must be between 0 and 1")
            
            # Validate training config
            if config.training.epochs < 1:
                raise ValueError("epochs must be >= 1")
            if config.training.learning_rate <= 0:
                raise ValueError("learning_rate must be > 0")
            
            return True
            
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False


def load_experiment_config(config_name_or_path: str) -> ExperimentConfig:
    """
    Load experiment configuration by name or file path.
    
    Args:
        config_name_or_path: Either a predefined config name or path to config file
        
    Returns:
        ExperimentConfig object
    """
    # Check if it's a predefined config name
    default_configs = ConfigManager.create_default_configs()
    if config_name_or_path in default_configs:
        return default_configs[config_name_or_path]
    
    # Otherwise, try to load from file
    if os.path.exists(config_name_or_path):
        return ConfigManager.load_config(config_name_or_path)
    
    # If neither, raise error
    available_configs = list(default_configs.keys())
    raise ValueError(
        f"Configuration '{config_name_or_path}' not found. "
        f"Available predefined configs: {available_configs}"
    )


def create_custom_config(
    model_type: str = 'vgg16',
    dataset_name: str = 'PILL',
    num_classes: int = 2,
    batch_size: int = 32,
    epochs: int = 25,
    learning_rate: float = 2e-4,
    **kwargs
) -> ExperimentConfig:
    """
    Create a custom configuration with common parameters.
    
    Args:
        model_type: Type of model to use
        dataset_name: Name of the dataset
        num_classes: Number of output classes
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        **kwargs: Additional parameters
        
    Returns:
        ExperimentConfig object
    """
    # Adjust defaults for quantum models
    if model_type in ['quantum', 'hybrid_cnn_qnn', 'vqc']:
        batch_size = min(batch_size, 16)  # Smaller batches for quantum
        epochs = min(epochs, 15)          # Fewer epochs for quantum
        learning_rate = max(learning_rate, 1e-3)  # Higher LR for quantum
    
    model_config = ModelConfig(
        model_type=model_type,
        num_classes=num_classes,
        n_qubits=kwargs.get('n_qubits', 4),
        n_layers=kwargs.get('n_layers', 2),
        input_size=kwargs.get('input_size')
    )
    
    data_config = DataConfig(
        dataset_name=dataset_name,
        batch_size=batch_size,
        resize=kwargs.get('resize', 224 if model_type == 'vgg16' else 64),
        val_split=kwargs.get('val_split', 0.1),
        seed=kwargs.get('seed', 42)
    )
    
    training_config = TrainingConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        results_dir=kwargs.get('results_dir', f'./results/{model_type}_{dataset_name}/')
    )
    
    return ExperimentConfig(
        model=model_config,
        data=data_config,
        training=training_config,
        experiment_name=f'{model_type}_{dataset_name}',
        description=f'Custom {model_type} configuration for {dataset_name}'
    ) 