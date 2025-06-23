## Federated Learning in Chemical Engineering: A Tutorial on a Framework for Privacy-Preserving Collaboration Across Distributed Data Sources

<p align="center">
  <img src="https://github.com/user-attachments/assets/30fe91f9-be9d-49c1-8a73-812134d36816" alt="Centered Image">
</p>

### Overview

Welcome to the Review on Federated Learning (FL) for Chemical Engineering repository. Federated Learning is where multiple decentralized devices collaboratively train a model without sharing their local data. Each device trains the model on its own data and only shares the model updates.

A Review paper is coming soon for the Industrial & Engineering Chemistry Research Journal! - Link - {WIP}

The comprehensive results section can be seen:

### Experiment Results

| **Experiment** | **Dataset** | **Train Accuracy** | **Test Accuracy** | **Train Loss** | **Test Loss** | **Time (sec)** |
|----------------|-------------|---------------|--------------|----------------|---------------|----------------|
| **Centralized Learning Experiments** (Num Epochs: 25) | | | | | | |
| PILL - [Standard_CentralNN_PILL.ipynb](https://github.com/elucidator8918/Fed-ML-Chem/blob/main/src/Standard_CentralNN_PILL.ipynb) | PILL: Pharmaceutical Dataset | 93.75% | 82.39% | 0.2600 | 0.5700 | 226.75 |
| DNA - [Standard_CentralNN_DNA.ipynb](https://github.com/elucidator8918/Fed-ML-Chem/blob/main/src/Standard_CentralNN_DNA.ipynb) | Human DNA Sequence | 100.00% | 94.50% | 0.0000 | 0.5080 | 97.49 |
| DNA (MMoE) - [Standard_CentralNN_DNA+MRI.ipynb](https://github.com/elucidator8918/Fed-ML-Chem/blob/main/src/Standard_CentralNN_DNA+MRI.ipynb) | Brain MRI Scan + Human DNA Sequence | 99.24% | 84.64% | 0.0274 | 0.7513 | 350.32 |
| MRI (MMoE) - [Standard_CentralNN_DNA+MRI.ipynb](https://github.com/elucidator8918/Fed-ML-Chem/blob/main/src/Standard_CentralNN_DNA+MRI.ipynb) | Brain MRI Scan + Human DNA Sequence | 99.72% | 90.63% | 0.0115 | 0.8555 | |
| HIV - [Standard_CentralNN_HIV.ipynb](https://github.com/elucidator8918/Fed-ML-Chem/blob/main/src/Standard_CentralNN_HIV.ipynb) | Human DNA Sequence | 96.77% | 95.45% | 0.1640 | 0.1770 | 100.79 |
| **Federated Learning Experiments** (Num Rounds: 20, Num Clients: 10, Num Epochs: 10) | | | | | | |
| PILL - [Standard_FedNN_PILL.ipynb](https://github.com/elucidator8918/Fed-ML-Chem/blob/main/src/Standard_FedNN_PILL.ipynb) | PILL: Pharmaceutical Dataset | 93.54% | 94.79% | 0.2800 | 0.2070 | 2215.40 |
| DNA - [Standard_FedNN_DNA.ipynb](https://github.com/elucidator8918/Fed-ML-Chem/blob/main/src/Standard_FedNN_DNA.ipynb) | Human DNA Sequence | 100.00% | 94.09% | 0.0000 | 1.2030 | 3921.43 |
| DNA (MMoE) - [Standard_FedNN_DNA+MRI.ipynb](https://github.com/elucidator8918/Fed-ML-Chem/blob/main/src/Standard_FedNN_DNA+MRI.ipynb) | Brain MRI Scan + Human DNA Sequence | 99.00% | 94.75% | 0.0776 | 0.4167 | 5543.29 |
| MRI (MMoE) - [Standard_FedNN_DNA+MRI.ipynb](https://github.com/elucidator8918/Fed-ML-Chem/blob/main/src/Standard_FedNN_DNA+MRI.ipynb) | Brain MRI Scan + Human DNA Sequence | 99.38% | 85.56% | 0.1997 | 1.0720 | |
| HIV - [Standard_FedNN_HIV.ipynb](https://github.com/elucidator8918/Fed-ML-Chem/blob/main/src/Standard_FedNN_HIV.ipynb) | Human DNA Sequence | 96.31% | 95.34% | 0.1790 | 0.1870 | 1042.82 |

*Results of Central and FL Experiments. In FL, both train loss and accuracy refer to a single client.*

### Repository Structure

```
.
├── LICENSE
├── README.md
├── run-cpu.sh
├── run-gpu.sh
├── run-tff-cpu.sh
├── run-tff-gpu.sh
└── src
    ├── README.md
    ├── Standard_CentralNN_DNA.ipynb
    ├── Standard_CentralNN_DNA+MRI.ipynb
    ├── Standard_CentralNN_HIV.ipynb
    ├── Standard_CentralNN_PILL.ipynb
    ├── Standard_CentralNN_Wafer.ipynb
    ├── Standard_FedNN_DNA.ipynb
    ├── Standard_FedNN_DNA+MRI.ipynb
    ├── Standard_FedNN_HIV.ipynb
    ├── Standard_FedNN_PILL.ipynb
    ├── Standard_FedNN_Wafer.ipynb
    ├── Standard_FedQNN_DNA.ipynb
    ├── Standard_FedQNN_DNA+MRI.ipynb
    ├── Standard_FedQNN_HIV.ipynb
    ├── Standard_FedQNN_PILL.ipynb
    ├── Standard_FedQNN_Wafer.ipynb
    ├── TFF_FedNN_PILL.ipynb
    └── utils
        ├── common.py
        ├── data_setup.py
        ├── engine.py
        └── __init__.py
```

## 🚀 NEW: Modular Architecture System

Fed-ML-Lib now features a **truly modular architecture** that allows you to combine any classical neural network with any combination of FHE encryption and quantum computing techniques!

### Key Features
- **🧱 Mix & Match**: Combine any base architecture (CNN, MLP, GCN, Pretrained) with any enhancements
- **🔐 Granular FHE**: Add encryption to specific layers only  
- **⚛️ Flexible Quantum**: Add quantum processing to any layers
- **🔄 Full Composability**: Use FHE + Quantum together in the same model
- **⚙️ Easy Configuration**: Simple API for complex combinations

### Quick Start with Modular System

```python
from fed_ml_lib.models import (
    create_modular_model, create_classical_model, 
    create_fhe_quantum_model
)

# Pure classical CNN
model = create_classical_model('cnn', input_shape=(3,32,32), num_classes=10)

# CNN with FHE encryption on classifier only
model = create_modular_model('cnn', use_fhe=True, fhe_layers=['classifier'])

# CNN with quantum enhancement on features only  
model = create_modular_model('cnn', use_quantum=True, quantum_layers=['features'])

# CNN with BOTH FHE and quantum (different layers)
model = create_fhe_quantum_model(
    'cnn', 
    fhe_layers=['classifier'],      # Encrypt classifier
    quantum_layers=['features']     # Quantum-enhance features
)
```

### Supported Combinations
| Base Architecture | + FHE Encryption | + Quantum Processing | + Both |
|-------------------|------------------|---------------------|---------|
| CNN | ✅ | ✅ | ✅ |
| MLP | ✅ | ✅ | ✅ |
| GCN | ✅ | ✅ | ✅ |
| Pretrained CNN | ✅ | ✅ | ✅ |

### Example: Run the Modular Demo

```bash
python examples/modular_training_example.py
```

### Installation

#### Clone the Repository

```bash
git clone https://github.com/elucidator8918/Fed-ML-Chem.git
cd Fed-ML-Chem
```

#### Install Dependencies

##### For CPU

```bash
conda create -n fed python=3.10.12 anaconda
conda init
conda activate fed
bash run-cpu.sh
```

##### For GPU

```bash
conda create -n fed python=3.10.12 anaconda
conda init
conda activate fed
bash run-gpu.sh
```

#### Running Experiments

Choose the appropriate notebook based on your dataset and encryption preference:

##### Standard Federated Learning

- **DNA Sequence Dataset:**
  - Notebook: `src/Standard_FedNN_DNA.ipynb`
  - Description: This dataset includes DNA sequences used for various biological and genetic studies, focusing on sequence classification and pattern recognition.

- **DNA+MRI Multimodal Dataset:**
  - **Notebook:** `src/Standard_FedNN_DNA+MRI.ipynb`
  - **Description:** It is used as a MoE with Multimodaility leveraging both DNA Sequence and MRI scans data to develop and evaluate models for detecting and interpreting tumors and dna classes.

- **PILL Dataset:**
  - **Notebook:** `src/Standard_FedNN_PILL.ipynb`
  - **Description:** This dataset includes images of pharmaceutical pills from the Pill Dataset used for various pharmaceutical studies, focusing on classification and pattern recognition.

##### Standard Quantum Federated Learning

- **DNA Sequence Dataset:**
  - Notebook: `src/Standard_FedQNN_DNA.ipynb`
  - Description: This dataset includes DNA sequences used for various biological and genetic studies, focusing on sequence classification and pattern recognition.

- **DNA+MRI Multimodal Dataset:**
  - **Notebook:** `src/Standard_FedQNN_DNA+MRI.ipynb`
  - **Description:** It is used as a MoE with Multimodaility leveraging both DNA Sequence and MRI scans data to develop and evaluate models for detecting and interpreting tumors and dna classes.

- **PILL Dataset:**
  - **Notebook:** `src/Standard_FedQNN_PILL.ipynb`
  - **Description:** This dataset includes images of pharmaceutical pills from the Pill Dataset used for various pharmaceutical studies, focusing on classification and pattern recognition.

### License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
