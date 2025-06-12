#!/bin/bash
pip install "flwr==1.5.0" "numpy==1.26.4" pennylane "ray>=2.3.0" matplotlib pillow scikit-learn seaborn pandas pyyaml kaggle sentence_transformers tqdm rdkit torch-geometric
pip install "torch==2.4.0+cu121" "torchvision==0.19.0+cu121" "torchaudio==2.4.0+cu121" --index-url https://download.pytorch.org/whl/cu121
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html