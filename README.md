# MuHC-Net

Code repository for MuHC-Net: an interpretable deep learning framework with causal hypergraph neural network for cancer prognostication and risk stratification from multi-omics data.



##  Project Layout

```
MuHC-Net5/
├── configs/
│   └── default.yaml          # Centralized configuration (paths + hyperparameters)
├── data/
│   └── Cancer/
│       ├── Cancer_miRNA_preprocess.csv
│       ├── Cancer_meth_preprocess.csv
│       ├── Cancer_mRNA_preprocess.csv
│       └── Cancer_clinical_delete_process.csv
├── src/
│   ├── models/
│   │   ├── ae.py             # Multi-omics Shared-Private Autoencoder (MSPAE)
│   │   ├── hgnn.py           # Causal Hypergraph Neural Network components
│   │   └── losses.py         # Cox, consistency, and ranking distillation losses
│   ├── utils/
│   │   ├── data_utils.py     # Data loading and patient alignment
│   │   ├── graph_utils.py    # Cosine-similarity hypergraph construction
│   │   ├── train_utils.py    # Random seeding and causal effect estimation
│   │   └── io_utils.py       # Directory and checkpoint utilities
│   └── scripts/
│       ├── train_ae.py       # Train AE representations
│       ├── infer_ae.py       # Infer AE representations from saved model
│       ├── train_ci.py       # Train CI prognosis model (Cox + HGNN)
│       ├── train_auc.py      # Train AUC survival classification model
│       └── eval_ci.py        # Evaluate CI (print only, no file output)
├── results/
│   └── Cancer/
│       ├── result_AE/        # AE model + 4 representation CSVs
│       ├── result_CI/        # CI best models + final_summary.json
│       └── result_AUC/       # AUC best models + final_summary.json
├── checkpoints/
│   ├── CI/                   # Grid search checkpoints (auto-created)
│   └── AUC/                  # Grid search checkpoints (auto-created)
├── requirements.txt
└── README.md
```



## Configuration

All paths and hyperparameters are centralized in `configs/default.yaml`.  
Modify the `paths.root` field to match your local environment before running any script.



## Main Modules

- `src/models/ae.py`: Multi-omics Shared-Private Autoencoder (MSPAE). Disentangles miRNA, methylation, and mRNA into shared and private representations.
- `src/models/hgnn.py`: Causal Hypergraph Neural Network. 
- `src/models/losses.py`: Survival analysis losses. Includes Cox partial log-likelihood, consistency loss for independent branch discrimination, and ranking distillation for label-free knowledge transfer.
- `src/utils/data_utils.py`: Multi-omics data loading and patient index alignment across miRNA, methylation, mRNA, and clinical tables.
- `src/utils/graph_utils.py`: Hypergraph adjacency construction via cosine similarity top-k neighbor selection.
- `src/utils/train_utils.py`: Random seeding for reproducibility and permutation-based causal effect strength estimation (`delta_ci`).
- `src/utils/io_utils.py`: Directory creation and JSON serialization utilities.



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

  
