# KG-DFI: A prediction of drug-food interactions based on knowledge graph embedding

## Description
A deep learning framework for predicting food-drug interactions using a heterogeneous biomedical knowledge graph. The model integrates a Relational Graph Convolutional Network (RGCN) encoder with a multi-head cross-attention module and a residual MLP predictor to capture complex relationships between food and drug entities. The framework supports both binary classification (interaction / no interaction) and multi-class classification (Possible, Positive, Negative, Harmful).

## Requirements
### Environment Setup
```bash
conda env create -f environment.yaml
conda activate KG-DFI
```

### Dependencies
- Python: 3.9.18
- PyTorch: 2.2.1 (CUDA 11.8)
- torchvision: 0.17.1
- torchaudio: 2.2.1
- DGL: 2.4.0+cu118
- DGL-LifeSci: 0.3.2
- AmpliGraph: 2.1.0
- scikit-learn: 1.5.2
- NumPy: 1.26.4
- Pandas: 2.2.3
- SciPy: 1.10.0
- RDKit: 2024.3.5
- matplotlib: 3.9.2
- seaborn: 0.13.2
- tqdm: 4.66.5

## Data & Model Weights
Due to file size limitations, some files cannot be hosted on GitHub. Please download them manually from Google Drive.

| File | Download Link | Destination |
|---|---|---|
| `background_kg_graph.pt` | [Google Drive](https://drive.google.com/file/d/1s2qnmnTK8fsIRiCDSdGrCau-JRTZMm97/view?usp=sharing) | `data/knowledge-graph/background_kg_graph.pt` |

## Usage
### 1. Training and Evaluation (Notebook)
Run the unified pipeline for both binary and multi-class tasks:
```bash
jupyter notebook KG-DFI.ipynb
```

### 2. Training and Evaluation (Python Script)
```python
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'code'))

from train import quick_train

# Binary classification
binary_model, binary_history, binary_data = quick_train(task='binary', data_dir='data/')

# Multi-class classification
multi_model, multi_history, multi_data = quick_train(task='multi', data_dir='data/')
```

### 3. Custom Configuration
Modify hyperparameters (e.g., embedding dimension, number of RGCN layers, learning rate) by passing keyword arguments:
```python
binary_model, history, data = quick_train(
    task='binary',
    embedding_dim=256,
    batch_size=512,
    learning_rate=0.0005,
    num_epochs=300,
    kg_layers=3,
    use_attention=True,
)
```

## Contact
- Mingi Kang: kmg013858@gmail.com
- Sunyong Yoo: syyoo@jnu.ac.kr
