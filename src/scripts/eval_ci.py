#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate CI prognosis using saved models and AE representations.
No files are saved; results are printed only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index

from src.models.hgnn import HazardHGNN, CausalGate, MultiHeadAttnFusion
from src.utils.graph_utils import build_H

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_config():
    cfg_path = Path(__file__).resolve().parents[2] / 'configs' / 'default.yaml'
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


cfg = load_config()
CANCER = cfg['project']['cancer']
ROOT = Path(cfg['paths']['root']).resolve()
DATA_DIR = ROOT / cfg['paths']['results'] / 'result_CI' / 'data'
MODEL_DIR = ROOT / cfg['paths']['results'] / 'result_CI' / 'model_hgnn'
CLINICAL_DIR = ROOT / cfg['paths']['data']

k_neighbor = cfg['hyperparams']['hgnn']['k_neighbor']
n_fold = cfg['hyperparams']['hgnn']['n_fold']
DROP_P = cfg['hyperparams']['hgnn']['drop_p']


def load_and_evaluate_fold(fold_idx, X_s, X_m, X_p, X_r, t, e, val_idx):
    """Load model for specified fold and evaluate on validation set."""
    X_s_val = torch.as_tensor(X_s[val_idx], device=device)
    X_m_val = torch.as_tensor(X_m[val_idx], device=device)
    X_p_val = torch.as_tensor(X_p[val_idx], device=device)
    X_r_val = torch.as_tensor(X_r[val_idx], device=device)
    t_val = torch.as_tensor(t[val_idx], device=device)
    e_val = torch.as_tensor(e[val_idx], device=device)

    H_val = [
        build_H(X_s[val_idx], k=k_neighbor),
        build_H(X_m[val_idx], k=k_neighbor),
        build_H(X_p[val_idx], k=k_neighbor),
        build_H(X_r[val_idx], k=k_neighbor)
    ]

    model_path = MODEL_DIR / f'best_model_fold{fold_idx}.pth'
    if not model_path.exists():
        print(f'  [WARN] Model not found: {model_path}')
        return None

    checkpoint = torch.load(model_path, map_location=device)
    has_cls = 'cls.0.weight' in checkpoint['nets'].keys() if 'nets' in checkpoint else False

    nets = nn.ModuleList([HazardHGNN(i, has_cls=has_cls).to(device) for i in range(4)])
    gate = CausalGate().to(device)
    fuse = MultiHeadAttnFusion().to(device)

    try:
        nets.load_state_dict(checkpoint['nets'], strict=True)
        gate.load_state_dict(checkpoint['gate'], strict=True)
        fuse.load_state_dict(checkpoint['fuse'], strict=True)
    except RuntimeError:
        nets.load_state_dict(checkpoint['nets'], strict=False)
        gate.load_state_dict(checkpoint['gate'], strict=False)
        fuse.load_state_dict(checkpoint['fuse'], strict=False)

    nets.eval()
    gate.eval()
    fuse.eval()

    with torch.no_grad():
        inputs_val = [X_s_val, X_m_val, X_p_val, X_r_val]
        outputs = []
        for i in range(4):
            out = nets[i](inputs_val[i], H_val[i])
            outputs.append(out[0] if has_cls else out[0])

        haz4_v = torch.cat(outputs, dim=1)
        haz4_v = gate(haz4_v)
        haz_f_v, _ = fuse(haz4_v)
        haz_np = haz_f_v.cpu().numpy().flatten()
        ci = concordance_index(t_val.cpu().numpy(), -haz_np, e_val.cpu().numpy())

    del nets, gate, fuse, checkpoint
    torch.cuda.empty_cache()

    return ci


def main():
    print('=' * 70)
    print(f'{CANCER} CI Prognosis Evaluation')
    print('=' * 70)

    h_shared = pd.read_csv(DATA_DIR / 'h_shared_64d.csv', index_col=0).astype('float32')
    p_mir = pd.read_csv(DATA_DIR / 'p_mir_64d.csv', index_col=0).astype('float32')
    p_meth = pd.read_csv(DATA_DIR / 'p_meth_64d.csv', index_col=0).astype('float32')
    p_mrna = pd.read_csv(DATA_DIR / 'p_mrna_64d.csv', index_col=0).astype('float32')
    clinical = pd.read_csv(CLINICAL_DIR / cfg['files']['clinical'], index_col=0)

    common_idx = h_shared.index.intersection(p_mir.index).intersection(p_meth.index).intersection(p_mrna.index).intersection(clinical.index)
    h_shared = h_shared.loc[common_idx]
    p_mir = p_mir.loc[common_idx]
    p_meth = p_meth.loc[common_idx]
    p_mrna = p_mrna.loc[common_idx]
    clinical = clinical.loc[common_idx]

    X_s = h_shared.values.astype('float32')
    X_m = p_mir.values.astype('float32')
    X_p = p_meth.values.astype('float32')
    X_r = p_mrna.values.astype('float32')

    t = clinical['os_time'].values.astype('float32')
    e = clinical['os_status'].map({'Dead': 1, 'Alive': 0}).values.astype('float32')

    print(f'[{CANCER}] Loaded {len(t)} samples, {int(e.sum())} events')
    print(f'[{CANCER}] Loading CI models from {MODEL_DIR}/')

    outer_kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    fold_cis = []

    for fold_idx, (tr_idx, val_idx) in enumerate(outer_kf.split(range(len(t)))):
        ci = load_and_evaluate_fold(fold_idx, X_s, X_m, X_p, X_r, t, e, val_idx)
        if ci is not None:
            fold_cis.append(ci)
            print(f'Fold {fold_idx} CI: {ci:.4f}')

    if len(fold_cis) > 0:
        mean_ci = np.mean(fold_cis)
        std_ci = np.std(fold_cis)
        print(f'\nMean CI: {mean_ci:.4f} +/- {std_ci:.4f}')


if __name__ == '__main__':
    main()