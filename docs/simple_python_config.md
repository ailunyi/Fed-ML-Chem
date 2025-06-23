# Simple Python Configuration System

## Overview

Fed-ML-Lib now includes a **super simple** Python function-based configuration system. No YAML files, no complex classes - just call Python functions with your parameters!

## Why Python Functions?

- ✅ **Familiar**: Everyone knows Python
- ✅ **Flexible**: Full Python capabilities  
- ✅ **IDE Support**: Autocomplete and type hints
- ✅ **No Files**: No separate config files to manage
- ✅ **Easy Debugging**: Just regular Python code

## Quick Start

### Basic Experiments

```python
from fed_ml_lib.config import pill_cnn, dna_mlp, quantum_mri

# PILL classification - just call the function!
pill_cnn("my_experiment", epochs=50, learning_rate=0.001)

# DNA classification  
dna_mlp("dna_exp", epochs=40, batch_size=64)

# Quantum MRI
quantum_mri("quantum_exp", n_qubits=6, epochs=30)
```

### Advanced Experiments

```python
from fed_ml_lib.config import run_experiment

# Full custom experiment with all options
run_experiment(
    name="my_custom_experiment",
    dataset="PILL", 
    model="cnn",
    
    # Model
    conv_channels=[32, 64, 128],
    dropout_rate=0.2,
    
    # Training
    epochs=50,
    learning_rate=0.001,
    batch_size=32,
    
    # Add FHE encryption
    use_fhe=True,
    fhe_scheme="CKKS",
    fhe_layers=['classifier'],
    
    # Add quantum processing
    use_quantum=True,
    n_qubits=4,
    quantum_layers=['features'],
    
    # Make it federated
    federated=True,
    num_clients=10,
    num_rounds=20
)
```

## Available Functions

### Quick Experiments
- `pill_cnn(name, **kwargs)` - PILL CNN classification
- `dna_mlp(name, **kwargs)` - DNA MLP classification  
- `mri_cnn(name, **kwargs)` - MRI CNN classification
- `federated_pill(name, **kwargs)` - Federated PILL experiment
- `quantum_mri(name, **kwargs)` - Quantum MRI experiment
- `fhe_dna(name, **kwargs)` - FHE DNA experiment
- `hybrid_pill(name, **kwargs)` - Hybrid FHE+Quantum PILL

### Main Function
- `run_experiment(name, dataset, model, **params)` - Full custom experiment

### Batch Operations
- `run_experiments(experiment_list)` - Run multiple experiments
- `parameter_sweep(func, param, values)` - Parameter sweeps

## Parameter Reference

### Core Parameters
- `name`: Experiment name
- `dataset`: "PILL", "DNA", "MRI", "HIV", "CIFAR10"
- `model`: "cnn", "mlp", "gcn", "pretrained_cnn"

### Model Architecture  
- `conv_channels`: [32, 64, 128] - CNN channels
- `hidden_dims`: [256, 128] - Hidden layer sizes
- `dropout_rate`: 0.2 - Dropout rate

### Training
- `epochs`: 25 - Number of epochs
- `learning_rate`: 0.001 - Learning rate
- `batch_size`: 32 - Batch size
- `optimizer`: "adam" - Optimizer type

### FHE Encryption
- `use_fhe`: False - Enable FHE
- `fhe_scheme`: "CKKS" - FHE scheme
- `fhe_layers`: ['classifier'] - Which layers to encrypt

### Quantum Processing
- `use_quantum`: False - Enable quantum
- `n_qubits`: 4 - Number of qubits
- `quantum_layers`: ['classifier'] - Which layers to enhance

### Federated Learning
- `federated`: False - Enable federated learning
- `num_clients`: 10 - Number of clients
- `num_rounds`: 20 - Number of rounds
- `local_epochs`: 5 - Local epochs per round

## Examples

### 1. Basic CNN Experiment
```python
pill_cnn("basic_cnn", epochs=30, learning_rate=0.001)
```

### 2. Quantum Experiment
```python
quantum_mri("quantum_test", n_qubits=8, epochs=25)
```

### 3. Federated Learning
```python
federated_pill("fed_test", num_clients=15, num_rounds=30)
```

### 4. Hybrid FHE + Quantum
```python
hybrid_pill("hybrid_test", n_qubits=6, fhe_scheme="BFV")
```

### 5. Custom Experiment
```python
run_experiment(
    "custom_dna",
    dataset="DNA",
    model="mlp",
    hidden_dims=[512, 256, 128],
    use_quantum=True,
    n_qubits=10,
    federated=True,
    num_clients=20
)
```

### 6. Parameter Sweep
```python
from fed_ml_lib.config import parameter_sweep

# Sweep learning rates
parameter_sweep(
    pill_cnn,
    'learning_rate', 
    [0.001, 0.01, 0.1],
    epochs=20
)
```

### 7. Batch Experiments
```python
from fed_ml_lib.config import run_experiments

experiments = [
    lambda: pill_cnn("exp1", epochs=25),
    lambda: dna_mlp("exp2", epochs=30), 
    lambda: quantum_mri("exp3", n_qubits=4)
]

run_experiments(experiments)
```

## Integration with Modular System

The configuration automatically works with your modular architecture:

```python
# This creates the right model based on your config
config = run_experiment("test", "PILL", "cnn", use_fhe=True, use_quantum=True)

# The config contains all the info to create your modular model
from fed_ml_lib.models import create_modular_model
model = create_modular_model(**config)
```

## Benefits

### ✨ **Super Simple**
- No YAML files to write or manage
- No complex configuration classes
- Just call functions with parameters

### 🔧 **Flexible**  
- Full Python capabilities
- Easy to add custom logic
- Simple parameter modifications

### 🎯 **Productive**
- IDE autocomplete works perfectly
- Type hints guide you
- Easy debugging and testing

### 🚀 **Powerful**
- Built-in parameter sweeps
- Batch experiment support
- All modular architecture features

## Comparison

| Feature | YAML Config | Python Functions |
|---------|-------------|------------------|
| Ease of use | ❌ Complex | ✅ Super simple |
| Flexibility | ❌ Limited | ✅ Full Python |
| IDE support | ❌ No autocomplete | ✅ Perfect support |
| File management | ❌ Separate files | ✅ No files needed |
| Debugging | ❌ Hard to debug | ✅ Easy debugging |
| Parameter sweeps | ❌ Manual | ✅ Built-in |

## Getting Started

1. Import the functions:
```python
from fed_ml_lib.config import pill_cnn, quantum_mri, run_experiment
```

2. Run your first experiment:
```python
pill_cnn("my_first_experiment", epochs=50)
```

3. Try quantum enhancement:
```python
quantum_mri("quantum_test", n_qubits=6)
```

4. Create a custom experiment:
```python
run_experiment("custom", "DNA", "mlp", use_fhe=True, epochs=100)
```

That's it! No configuration files, no complex setup - just Python functions that do what you want. 