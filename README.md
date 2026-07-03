# MuHC-Net

Code repository for MuHC-Net: an interpretable deep learning framework with causal hypergraph neural network for cancer prognostication and risk stratification from multi-omics data.



##  Project Layout

```
MuHC-Net/
├── configs/
│   └── default.yaml              # Centralized configuration (paths + hyperparameters)
├── data/
│   ├── BRCA/                     # Preprocessed multi-omics data (BRCA cohort)
│   │   ├── BRCA_miRNA_preprocess.csv          # miRNA expression: N × 200
│   │   ├── BRCA_meth_preprocess.csv           # DNA methylation: N × 600
│   │   ├── BRCA_mRNA_preprocess.csv           # mRNA expression: N × 600
│   │   └── BRCA_clinical_delete_process.csv   # Survival data: os_time, os_status
│   ├── LUAD/
│   ├── STAD/
│   ├── COAD/
│   ├── LIHC/
│   ├── LGG/
│   └── LAML/
├── src/
│   ├── models/
│   │   ├── init.py
│   │   ├── ae.py                 # Multi-omics Shared-Private Autoencoder (MSPAE)
│   │   ├── hgnn.py               # Causal Hypergraph Neural Network components
│   │   └── losses.py             # Cox, consistency, and ranking distillation losses
│   ├── utils/
│   │   ├── init.py
│   │   ├── config.py             # YAML configuration loader
│   │   ├── data_utils.py         # Data loading and patient alignment
│   │   ├── graph_utils.py        # Cosine-similarity hypergraph construction
│   │   ├── train_utils.py        # Random seeding and causal effect estimation
│   │   └── io_utils.py           # Directory and checkpoint utilities
│   ├── scripts/
│   │   ├── init.py
│   │   ├── train_ae.py           # Train MSPAE representations
│   │   ├── train_ci.py           # Train C-index prognosis model
│   │   ├── train_auc.py          # Train AUC survival classification model
│   │   └── eval_ci.py            # Evaluate CI 
│   └── external/
│       └── TARGET-LAML/          # External validation cohort (TARGET project)
│           ├── data/             # Preprocessed TARGET-LAML omics + clinical
│           ├── RAW/              # Original downloaded data
│           ├── model_ae/         # AE model transferred from TCGA-LAML
│           ├── model_hgnn/       # 5-fold prognosis models from TCGA-LAML
│           ├── results/          # External validation outputs
│           └── scripts/          # External validation scripts
├── results/
│   ├── BRCA/                     # BRCA experiment outputs
│   │   ├── result_AE/            # MSPAE outputs
│   │   │   ├── ae_model.pth              # Trained AE model weights
│   │   │   ├── h_shared_64d.csv          # Shared representation: N × 64
│   │   │   ├── p_mir_64d.csv             # miRNA-private representation: N × 64
│   │   │   ├── p_meth_64d.csv            # Methylation-private representation: N × 64
│   │   │   ├── p_mrna_64d.csv            # mRNA-private representation: N × 64
│   │   ├── result_CI/            # C-index prognosis model outputs
│   │   │   ├── best_model_foldX.pth      # Fold X optimal model
│   │   │   └── final_summary.json        # Best hyperparameters + per-fold CI
│   │   └── result_AUC/           # AUC classification model outputs
...
│   └── LAML/
├── checkpoints/
│   ├── CI/                       # Grid search intermediate checkpoints
│   └── AUC/
├── requirements.txt
└── README.md
```



## Configuration

All paths and hyperparameters are centralized in `configs/default.yaml`.
Modify the `base_dir` field to match your local environment before running any script.



## Main Modules

- `src/models/ae.py`: Multi-omics Shared-Private Autoencoder (MSPAE). 
- `src/models/hgnn.py`: Causal Hypergraph Neural Network.
- `src/models/losses.py`: Survival analysis losses. 
- `src/utils/config.py`: YAML configuration loader with CLI argument merging.
- `src/utils/data_utils.py`: Multi-omics data loading and patient index alignment across miRNA, methylation, mRNA, and clinical tables.
- `src/utils/graph_utils.py`: Hypergraph adjacency construction .
- `src/utils/io_utils.py`: Directory resolution and checkpoint serialization utilities.



### Data Preparation

lace preprocessed files in `data/{CANCER}/` (e.g., `data/BRCA/`):

- `{CANCER}_miRNA_preprocess.csv` (N × 200)
- `{CANCER}_meth_preprocess.csv` (N × 600)
- `{CANCER}_mRNA_preprocess.csv` (N × 600)
- `{CANCER}_clinical_delete_process.csv` (must contain `os_time` and `os_status` columns)



### Representations

After MSPAE training, the following representations are saved to `results/{CANCER}/result_AE/`:

- `h_shared_64d.csv` (N × 64): Multi-omics shared representation
- `p_mrna_64d.csv` (N × 64): mRNA-private representation
- `p_mir_64d.csv` (N × 64): miRNA-private representation
- `p_meth_64d.csv` (N × 64): Methylation-private representation



## External Validation (TARGET-LAML)

The `src/external/TARGET-LAML/` directory contains the independent pediatric AML cohort from the TARGET project for cross-age validation.

- `src/external/TARGET-LAML/data/`: Preprocessed miRNA, methylation, mRNA, and clinical files
- `src/external/TARGET-LAML/model_ae/`: AE model transferred from TCGA-LAML
- `src/external/TARGET-LAML/model_hgnn/`: 5-fold prognosis models trained on TCGA-LAML
- `src/external/TARGET-LAML/RAW/`: Original downloaded data file



