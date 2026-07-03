#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate CI prognosis model.
Loads result_CI/data/ representations and result_CI/model_hgnn_v2/ best models.
Prints only. No files saved.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.hgnn import HazardHGNN, ModalityCausalGate, MultiHeadAttnFusion
from src.utils.config import load_config, merge_cli_args
from src.utils.graph_utils import build_H
from src.utils.io_utils import resolve_paths


def evaluate(cfg, cancer):
    print(f"\n[INFO] {cancer} | CI Evaluation")
    print(f"[INFO] Config: {cfg.get('config_path', 'default.yaml')}")
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    
    paths = resolve_paths(cfg, cancer)
    ae_dir = paths['ci_data_dir']
    ci_dir = paths['ci_model_dir']
    
    h_shared = pd.read_csv(ae_dir / 'h_shared_64d.csv', index_col=0).astype('float32')
    p_mir = pd.read_csv(ae_dir / 'p_mir_64d.csv', index_col=0).astype('float32')
    p_meth = pd.read_csv(ae_dir / 'p_meth_64d.csv', index_col=0).astype('float32')
    p_mrna = pd.read_csv(ae_dir / 'p_mrna_64d.csv', index_col=0).astype('float32')
    clinical = pd.read_csv(paths['data_dir'] / f"{cancer}_clinical_delete_process.csv", index_col=0)
    
    common_idx = h_shared.index.intersection(p_mir.index).intersection(p_meth.index).intersection(p_mrna.index).intersection(clinical.index)
    h_shared = h_shared.loc[common_idx]; p_mir = p_mir.loc[common_idx]
    p_meth = p_meth.loc[common_idx]; p_mrna = p_mrna.loc[common_idx]
    clinical = clinical.loc[common_idx]
    
    X_s = h_shared.values.astype('float32'); X_m = p_mir.values.astype('float32')
    X_p = p_meth.values.astype('float32'); X_r = p_mrna.values.astype('float32')
    t = clinical['os_time'].values.astype('float32')
    e = clinical['os_status'].map({'Dead': 1, 'Alive': 0}).values.astype('float32')
    n_total = len(t)
    
    print(f"[INFO] Samples: {n_total}")
    print(f"[INFO] Loaded 5-fold models")
    
    outer_kf = KFold(n_splits=cfg['hgnn']['n_fold'], shuffle=True, random_state=cfg['seed'])
    fold_cis = []
    
    for fold_idx, (tr_idx, val_idx) in enumerate(outer_kf.split(range(n_total))):
        X_s_val = torch.as_tensor(X_s[val_idx], device=device, dtype=torch.float32)
        X_m_val = torch.as_tensor(X_m[val_idx], device=device, dtype=torch.float32)
        X_p_val = torch.as_tensor(X_p[val_idx], device=device, dtype=torch.float32)
        X_r_val = torch.as_tensor(X_r[val_idx], device=device, dtype=torch.float32)
        t_val = t[val_idx]; e_val = e[val_idx]
        
        H_val = [torch.as_tensor(build_H(X_s[val_idx], k=cfg['hgnn']['k_neighbor'], device=device), device=device),
                 torch.as_tensor(build_H(X_m[val_idx], k=cfg['hgnn']['k_neighbor'], device=device), device=device),
                 torch.as_tensor(build_H(X_p[val_idx], k=cfg['hgnn']['k_neighbor'], device=device), device=device),
                 torch.as_tensor(build_H(X_r[val_idx], k=cfg['hgnn']['k_neighbor'], device=device), device=device)]
        
        model_path = ci_dir / f"best_model_fold{fold_idx}.pth"
        if not model_path.exists():
            print(f"[WARN] Model not found: {model_path}")
            continue
        
        state = torch.load(model_path, map_location=device)
        
        nets = nn.ModuleList([HazardHGNN(i, use_cls=False, drop_p=cfg['hgnn']['drop_p']).to(device) for i in range(4)])
        gate = ModalityCausalGate().to(device)
        fuse = MultiHeadAttnFusion().to(device)
        
        nets.load_state_dict(state['nets'])
        gate.load_state_dict(state['gate'])
        fuse.load_state_dict(state['fuse'])
        
        nets.eval(); gate.eval(); fuse.eval()
        with torch.no_grad():
            inputs_val = [X_s_val, X_m_val, X_p_val, X_r_val]
            outputs_v = [nets[i](inputs_val[i], H_val[i]) for i in range(4)]
            emb4_v = [o[1] for o in outputs_v]
            haz_f_v, _ = fuse(emb4_v)
            haz_np = haz_f_v.cpu().numpy().flatten()
            ci = concordance_index(t_val, -haz_np, e_val)
            fold_cis.append(ci)
            print(f"[INFO] Fold {fold_idx} CI: {ci:.4f}")
    
    if fold_cis:
        mean_ci = np.mean(fold_cis)
        std_ci = np.std(fold_cis)
        print(f"[INFO] Ensemble Mean CI: {mean_ci:.4f} ± {std_ci:.4f}")
    print(f"[INFO] Completed. No files saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--cancer', type=str, required=True)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    cfg = merge_cli_args(cfg, args)
    cfg['config_path'] = args.config
    evaluate(cfg, args.cancer)


if __name__ == '__main__':
    main()