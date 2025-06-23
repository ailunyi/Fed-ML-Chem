# Configuration System Documentation

## Overview

Fed-ML-Lib features a comprehensive configuration management system designed to handle the complexity of federated learning, quantum computing, and FHE encryption experiments. The system provides a clean, type-safe, and extensible way to manage all aspects of your experiments.

## 🏗️ Architecture

The configuration system is built with several key components:

### 1. **Base Configuration Classes** (`fed_ml_lib.config.base`)
- **Hierarchical**: Nested configuration structure
- **Type-safe**: Automatic type validation
- **Serializable**: YAML/JSON export/import
- **Mergeable**: Combine configurations

### 2. **Fluent API Builder** (`fed_ml_lib.config.builder`)
- **Chainable**: Method chaining for easy configuration
- **Preset-aware**: Integrates with dataset presets
- **Validation**: Built-in validation during construction

### 3. **Enhanced Manager** (`fed_ml_lib.config.enhanced_manager`)
- **Versioning**: Automatic configuration versioning
- **Templates**: Reusable configuration templates
- **Environment**: Environment-specific settings

### 4. **Validation System** (`fed_ml_lib.config.validation`)
- **Comprehensive**: Validates all configuration aspects
- **Helpful**: Provides suggestions for fixes
- **Flexible**: Warning vs error levels

## 🚀 Quick Start

### Basic Usage

```python
from fed_ml_lib.config import ConfigBuilder

# Simple CNN experiment
config = (ConfigBuilder()
         .experiment("my_cnn_experiment")
         .dataset("PILL")
         .model("cnn", conv_channels=[32, 64, 128])
         .training(epochs=25, learning_rate=0.001)
         .build())
```

### Modular Architecture Integration

```python
# CNN with FHE encryption
config = (ConfigBuilder()
         .experiment("fhe_cnn")
         .dataset("PILL")
         .model("cnn")
         .with_fhe(scheme="CKKS", layers=['classifier'])
         .build())

# CNN with quantum enhancement
config = (ConfigBuilder()
         .experiment("quantum_cnn")
         .dataset("MRI")
         .model("cnn")
         .with_quantum(n_qubits=6, layers=['features'])
         .build())

# Hybrid: FHE + Quantum
config = (ConfigBuilder()
         .experiment("hybrid_cnn")
         .dataset("DNA")
         .model("mlp")
         .with_fhe(layers=['classifier'])
         .with_quantum(layers=['features'])
         .build())
```

### Federated Learning

```python
# Federated learning experiment
config = (ConfigBuilder()
         .experiment("federated_dna")
         .dataset("DNA", partition_strategy="dirichlet", alpha=0.3)
         .model("mlp", hidden_dims=[128, 64])
         .training(epochs=20)
         .federated(num_clients=10, num_rounds=25)
         .build())
```

## 📋 Configuration Components

### ExperimentConfig
The top-level configuration containing all experiment settings.

```python
@dataclass
class ExperimentConfig:
    name: str = "default_experiment"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    federated: Optional[FederatedConfig] = None
```

### ModelConfig
Configuration for the modular model system.

```python
@dataclass
class ModelConfig:
    # Base architecture
    architecture: str = "cnn"  # cnn, mlp, gcn, pretrained_cnn
    
    # Architecture parameters
    conv_channels: Optional[List[int]] = None
    hidden_dims: Optional[List[int]] = None
    dropout_rate: float = 0.1
    
    # Modular enhancements
    use_fhe: bool = False
    fhe_scheme: str = "CKKS"
    fhe_layers: List[str] = field(default_factory=list)
    
    use_quantum: bool = False
    n_qubits: int = 4
    quantum_layers: List[str] = field(default_factory=list)
```

### DatasetConfig
Dataset and data loading configuration.

```python
@dataclass
class DatasetConfig:
    name: str = "PILL"
    input_shape: tuple = (3, 224, 224)
    num_classes: int = 10
    
    # Data preprocessing
    batch_size: int = 32
    normalize: bool = True
    augmentation: bool = True
    
    # Federated settings
    partition_strategy: str = "iid"
    num_partitions: int = 10
```

### TrainingConfig
Training and optimization settings.

```python
@dataclass
class TrainingConfig:
    epochs: int = 25
    learning_rate: float = 0.001
    optimizer: str = "adam"
    
    # Regularization
    weight_decay: float = 1e-4
    gradient_clipping: Optional[float] = None
    early_stopping: bool = False
    
    # Logging
    log_frequency: int = 10
    save_checkpoints: bool = True
```

### FederatedConfig
Federated learning specific settings.

```python
@dataclass
class FederatedConfig:
    num_clients: int = 10
    num_rounds: int = 20
    clients_per_round: int = 5
    local_epochs: int = 5
    
    # Aggregation
    aggregation_strategy: str = "fedavg"
    
    # Privacy
    differential_privacy: bool = False
    dp_epsilon: float = 1.0
```

## 🔧 Fluent API Reference

### ConfigBuilder Methods

#### Experiment Metadata
```python
.experiment(name: str, description: str = "", tags: List[str] = None)
```

#### Dataset Configuration
```python
.dataset(name: str, **kwargs)
# Example: .dataset("PILL", batch_size=64, resize=(224, 224))
```

#### Model Architecture
```python
.model(architecture: str, **kwargs)
# Example: .model("cnn", conv_channels=[32, 64], dropout_rate=0.2)
```

#### FHE Enhancement
```python
.with_fhe(scheme: str = "CKKS", layers: List[str] = None, **kwargs)
# Example: .with_fhe("CKKS", layers=['classifier'])
```

#### Quantum Enhancement
```python
.with_quantum(n_qubits: int = 4, layers: List[str] = None, **kwargs)
# Example: .with_quantum(n_qubits=6, layers=['features', 'classifier'])
```

#### Training Configuration
```python
.training(**kwargs)
# Example: .training(epochs=50, learning_rate=0.001, optimizer="adamw")
```

#### Federated Learning
```python
.federated(num_clients: int = 10, num_rounds: int = 20, **kwargs)
# Example: .federated(num_clients=15, num_rounds=30, local_epochs=3)
```

#### Environment Settings
```python
.environment(env: Environment = Environment.DEVELOPMENT, **kwargs)
.gpu(use_gpu: bool = True, gpu_ids: List[int] = None)
.debug(debug: bool = True)
.paths(data_root: str = None, output_root: str = None, **kwargs)
```

## 🎯 Preset Builders

For common experiment types, use preset builders:

```python
from fed_ml_lib.config import PresetBuilder

# Classical ML
config = PresetBuilder.classical_experiment("PILL", "cnn").build()

# Federated learning
config = PresetBuilder.federated_experiment("DNA", "mlp", num_clients=10).build()

# Quantum ML
config = PresetBuilder.quantum_experiment("MRI", "cnn", n_qubits=4).build()

# FHE encryption
config = PresetBuilder.fhe_experiment("PILL", "cnn", scheme="CKKS").build()

# Hybrid (FHE + Quantum)
config = PresetBuilder.hybrid_experiment("DNA", "mlp").build()
```

## 📁 Configuration Management

### Saving and Loading

```python
from fed_ml_lib.config import config_manager

# Create and save
config = ConfigBuilder().experiment("my_exp").dataset("PILL").build()
config_manager.save_experiment(config)

# Load
loaded_config = config_manager.load_experiment("my_exp")

# List all experiments
experiments = config_manager.list_experiments()
```

### Versioning

```python
# Automatic versioning when saving
config_manager.save_experiment(config, version=True)

# Get version history
versions = config_manager.get_experiment_versions("my_exp")

# Load specific version
old_config = config_manager.load_experiment("my_exp", version="001")
```

### Cloning and Modification

```python
# Clone an experiment with modifications
new_config = config_manager.clone_experiment(
    "original_exp", 
    "modified_exp",
    modifications={
        'training': {'epochs': 100, 'learning_rate': 0.0001},
        'model': {'dropout_rate': 0.3}
    }
)
```

### Experiment Series

```python
# Create systematic variations
variations = [
    {'training': {'learning_rate': 0.001}},
    {'training': {'learning_rate': 0.01}},
    {'training': {'learning_rate': 0.1}},
]

base_config = ConfigBuilder().dataset("DNA").model("mlp").build()

series = config_manager.create_experiment_series(
    "lr_sweep",
    variations,
    base_config=base_config
)
```

## ✅ Validation System

### Automatic Validation

```python
from fed_ml_lib.config import validate_config

# Validate configuration
results = validate_config(config)

for result in results:
    print(f"{result.level}: {result.field} - {result.message}")
    if result.suggestion:
        print(f"  Suggestion: {result.suggestion}")
```

### Validation Levels

- **ERROR**: Critical issues that prevent execution
- **WARNING**: Potential problems or suboptimal settings
- **INFO**: Informational messages

### Common Validations

- **Type checking**: Ensures correct data types
- **Range validation**: Values within acceptable ranges
- **Compatibility**: Cross-component compatibility
- **Best practices**: Recommendations for optimal settings

## 🌍 Environment Management

### Environment Types

```python
from fed_ml_lib.config import Environment

Environment.DEVELOPMENT  # Development environment
Environment.TESTING      # Testing environment  
Environment.PRODUCTION    # Production environment
```

### Environment-Specific Settings

```python
# Set environment configurations
dev_config = {
    'debug': True,
    'log_level': 'DEBUG',
    'use_gpu': False,
    'data_root': './data_dev'
}

config_manager.set_environment_config('development', dev_config)

# Use environment in experiment
config = (ConfigBuilder()
         .experiment("my_exp")
         .environment(Environment.DEVELOPMENT)
         .build())
```

## 💾 Serialization

### YAML Export/Import

```python
# Export to YAML
yaml_str = config.to_yaml()
config.to_yaml("config.yaml")  # Save to file

# Import from YAML
config = ExperimentConfig.from_yaml("config.yaml")
```

### JSON Export/Import

```python
# Export to JSON
json_str = config.to_json()
config.to_json("config.json")  # Save to file

# Import from JSON
config = ExperimentConfig.from_json("config.json")
```

## 🔗 Integration with Modular System

The configuration system seamlessly integrates with the modular architecture:

```python
from fed_ml_lib.models import create_modular_model

# Create model from configuration
model = create_modular_model(
    config.model.architecture,
    use_fhe=config.model.use_fhe,
    use_quantum=config.model.use_quantum,
    **config.model.to_dict()
)
```

## 📖 Best Practices

### 1. Use Descriptive Names
```python
.experiment("pill_classification_vgg16_fhe", "PILL classification with VGG16 and FHE")
```

### 2. Leverage Presets
```python
# Start with preset, then customize
config = (PresetBuilder.classical_experiment("PILL", "cnn")
         .training(epochs=50)
         .with_fhe()
         .build())
```

### 3. Validate Early
```python
# Validate during development
try:
    validate_config(config, strict=True)
except ValidationError as e:
    print(f"Configuration issues: {e}")
```

### 4. Use Environment Configs
```python
# Different settings for different environments
.environment(Environment.PRODUCTION if production else Environment.DEVELOPMENT)
```

### 5. Version Your Experiments
```python
# Enable versioning for important experiments
config_manager.save_experiment(config, version=True)
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Validation Errors**: Check the validation messages for specific issues
3. **File Not Found**: Verify paths are correct and files exist
4. **Type Errors**: Ensure parameter types match expected types

### Debug Mode

```python
# Enable debug mode for detailed logging
config = (ConfigBuilder()
         .debug(True)
         .build())
```

## 📚 Examples

See `examples/config_system_example.py` for comprehensive usage examples.

## 🔄 Migration from Legacy System

The new system maintains backward compatibility:

```python
# Legacy system still works
from fed_ml_lib.config import LegacyConfigManager, get_dataset_preset

# New system provides enhanced features
from fed_ml_lib.config import ConfigBuilder, config_manager
```

## 🤝 Contributing

To extend the configuration system:

1. Add new configuration classes in `base.py`
2. Extend validation in `validation.py`
3. Add builder methods in `builder.py`
4. Update documentation and examples 