import pandas as pd
import torch
from pathlib import Path


def load_omics(data_dir, cancer):
    d = Path(data_dir) / cancer
    mir = pd.read_csv(d / f"{cancer}_miRNA_preprocess.csv", index_col=0)
    meth = pd.read_csv(d / f"{cancer}_meth_preprocess.csv", index_col=0)
    mrna = pd.read_csv(d / f"{cancer}_mRNA_preprocess.csv", index_col=0)
    clinical = pd.read_csv(d / f"{cancer}_clinical_delete_process.csv", index_col=0)
    return mir, meth, mrna, clinical


def align_samples(*dfs):
    idx = dfs[0].index
    for df in dfs[1:]:
        idx = idx.intersection(df.index)
    return [df.loc[idx] for df in dfs]


def to_tensor(x, device, dtype=torch.float32):
    return torch.as_tensor(x, device=device, dtype=dtype)