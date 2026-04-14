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
    train_hgnn_auc.py
  data/
    miRNA_preprocess.csv
    meth_preprocess.csv
    mRNA_preprocess.csv
    clinical_delete_process.csv
    p_meth_64d.csv
    p_mir_64d.csv
    p_mrna_64d.csv
    h_shared_64d.csv
  outputs/
    auc_model/
    CI_model/
    ae_model.pth
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



### Representations

Shared and modality-private representations from mRNA, miRNA, and DNA methylation profiles

- h_shared_64d.csv (N×64)

- p_mrna_64d.csv (N×64)  

- p_mir_64d.csv (N×64)

- p_meth_64d.csv (N×64)

  

MuHC-Net\outputs\ae_model.pth

- `ae_model.pth` is the complete model checkpoint file for MSPAE , containing the following network parameters and training states:
  - **Encoder weights**: `enc_mir`, `enc_meth`, `enc_mrna` (shared and private branches for miRNA, DNA methylation, and mRNA modalities, respectively)
  - **Attention fusion layer weights**: `attn_fusion` 
  - **Shared decoder weights**: `dec_shared_mir`, `dec_shared_meth`, `dec_shared_mrna` 
  - **Private decoder weights**: `dec_priv_mir`, `dec_priv_meth`, `dec_priv_mrna` 
  - **Optimizer state**: `optimizer`
  - **Training metadata**: Final epoch number, total loss, and final attention weight distribution (α_mir, α_meth, α_mrna)

python src/extract_representations.py

- Shared representations: `h_shared_64d.csv`

- Private representations: `p_mir_64d.csv`, `p_meth_64d.csv`, `p_mrna_64d.csv`

  

## Run Training Pipeline

Train Autoencoder
python src/train_ae.py

Extract Representations
python src/extract_representations.py

Train HGNN 
python src/train_hgnn.py

Notes
Code is split by responsibility: models (architectures), src (training scripts), config (paths).
train_hgnn.py performs 5-fold CV and selects optimal params.