#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loading and alignment utilities.
"""

import pandas as pd
import torch
from pathlib import Path


def load_omics(data_dir, cancer):
    """
    Load preprocessed miRNA, methylation, and mRNA data.
    Returns tensors aligned by patient index.
    """
    data_dir = Path(data_dir)
    mir = pd.read_csv(data_dir / f"{cancer}_miRNA_preprocess.csv", index_col=0)
    meth = pd.read_csv(data_dir / f"{cancer}_meth_preprocess.csv", index_col=0)
    mrna = pd.read_csv(data_dir / f"{cancer}_mRNA_preprocess.csv", index_col=0)

    common = mir.index.intersection(meth.index).intersection(mrna.index)
    return mir.loc[common], meth.loc[common], mrna.loc[common], common


def load_clinical(data_dir, cancer):
    """Load clinical survival data."""
    path = Path(data_dir) / f"{cancer}_clinical_delete_process.csv"
    clin = pd.read_csv(path, index_col=0)
    return clin


def align_clinical_to_omics(clinical, omics_index):
    """Subset clinical data to match omics patient IDs."""
    common = clinical.index.intersection(omics_index)
    return clinical.loc[common]


def to_tensor(x, device='cpu', dtype=torch.float32):
    """Convert numpy array to torch tensor on target device."""
    return torch.as_tensor(x, device=device, dtype=dtype)