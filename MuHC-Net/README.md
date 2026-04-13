# MuHC-Net

Code repository for MuHC-Net: an interpretable deep learning framework with causal hypergraph neural network for cancer prognostication and risk stratification from multi-omics data

##  Project Layout

```
MuHC-Net/
  config/
    paths.py
  models/
    ae.py
    hgnn.py
  src/
    train_ae.py
    extract_representations.py
    train_hgnn.py
  data/
    miRNA_preprocess.csv
    meth_preprocess.csv
    mRNA_preprocess.csv
    clinical_delete_process.csv
  outputs/
    ae_model.pth
    model_v8_global/
  requirements.txt
  README.md
  .gitignore
```



## Setup
pip install -r requirements.txt



## Configure Paths
Edit config/paths.py to set your local data directory. 



### Data Preparation
Place preprocessed files in data/:

- miRNA_preprocess.csv (N×200)
- meth_preprocess.csv (N×600)  
- mRNA_preprocess.csv (N×600)
- clinical_delete_process.csv (must contain 'os_time' and 'os_status' columns)



## Run Training Pipeline

Step 1: Train Autoencoder
python src/train_ae.py

Step 2: Extract Representations
python src/extract_representations.py

Step 3: Train HGNN (Global Hyperparameter Search)
python src/train_hgnn.py

Notes
Code is split by responsibility: models (architectures), src (training scripts), config (paths).
train_hgnn.py performs 5-fold CV and selects optimal params based on cross-fold average CI.