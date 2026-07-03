#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train AUC prognosis model.
Grid search over alpha and beta with 5-fold CV.
Outputs: best_model_fold*.pth, final_summary.json under model_auc_v2/
"""

import json
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from multiprocessing import get_context

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.hgnn import HazardHGNN, ModalityCausalGate, MultiHeadAttnFusion
from src.models.losses import ranking_distill_loss_v2, ConsistencyLoss
from src.utils.config import load_config, merge_cli_args
from src.utils.graph_utils import build_H
from src.utils.train_utils import seed_everything, CausalEffectEstimator
from src.utils.io_utils import resolve_paths


def train_single_fold(args_tuple):
    fold_idx, alpha, beta, fold_data, cfg = args_tuple
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    hgnn_cfg = cfg['hgnn']
    seed_everything(cfg['seed'] + fold_idx + int(alpha * 10) + int(beta * 10))
    
    def _to_tensor(x, dtype=torch.float32):
        return torch.as_tensor(x, device=device, dtype=dtype)
    
    x_s_tr = _to_tensor(fold_data['X_s_tr']); x_m_tr = _to_tensor(fold_data['X_m_tr'])
    x_p_tr = _to_tensor(fold_data['X_p_tr']); x_r_tr = _to_tensor(fold_data['X_r_tr'])
    x_s_val = _to_tensor(fold_data['X_s_val']); x_m_val = _to_tensor(fold_data['X_m_val'])
    x_p_val = _to_tensor(fold_data['X_p_val']); x_r_val = _to_tensor(fold_data['X_r_val'])
    t_tr = _to_tensor(fold_data['t_tr']); e_tr = _to_tensor(fold_data['e_tr'], dtype=torch.long)
    t_val = _to_tensor(fold_data['t_val']); e_val = _to_tensor(fold_data['e_val'], dtype=torch.long)
    
    H_tr = [_to_tensor(h) for h in fold_data['H_tr']]
    H_val = [_to_tensor(h) for h in fold_data['H_val']]
    
    nets = nn.ModuleList([HazardHGNN(i, use_cls=True, drop_p=hgnn_cfg['drop_p']).to(device) for i in range(4)])
    gate = ModalityCausalGate().to(device)
    fuse = MultiHeadAttnFusion().to(device)
    
    causal_est = CausalEffectEstimator(device=device)
    causal_est.set_train_means(fold_data['train_means'])
    
    optimizer = optim.Adam(
        list(nets.parameters()) + list(gate.parameters()) + list(fuse.parameters()),
        lr=hgnn_cfg['lr'], weight_decay=hgnn_cfg['weight_decay']
    )
    consistency_fn = ConsistencyLoss()
    
    best_auc = 0.0; counter = 0; best_state = None
    
    for epoch in range(1, hgnn_cfg['epochs'] + 1):
        for net in nets: net.train()
        gate.train(); fuse.train(); optimizer.zero_grad()
        
        inputs_tr = [x_s_tr, x_m_tr, x_p_tr, x_r_tr]
        outputs = [nets[i](inputs_tr[i], H_tr[i]) for i in range(4)]
        haz4 = torch.cat([o[0] for o in outputs], dim=1)
        logits4 = torch.stack([o[1] for o in outputs], dim=1)
        emb4 = [o[2] for o in outputs]
        
        haz4_gated, gate_w = gate(haz4)
        haz_f, attn = fuse(emb4)
        
        logits_f = (attn.unsqueeze(-1) * logits4).sum(dim=1)
        
        loss_main = F.cross_entropy(logits_f, e_tr)
        loss_cons = consistency_fn(haz4_gated, t_tr, e_tr)
        loss_dis = ranking_distill_loss_v2(haz_f, haz4_gated, t_tr, e_tr)
        
        loss = loss_main + alpha * loss_cons + beta * loss_dis
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(nets.parameters()) + list(gate.parameters()) + list(fuse.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        
        if epoch % 10 == 0:
            gate.eval(); fuse.eval()
            with torch.no_grad():
                effects = causal_est.compute_ranking_effects(nets, fuse, inputs_tr, H_tr, t_tr, e_tr)
                gate.update_effects(effects)
            gate.train()
        
        if epoch % 10 == 0:
            for net in nets: net.eval()
            gate.eval(); fuse.eval()
            with torch.no_grad():
                inputs_val = [x_s_val, x_m_val, x_p_val, x_r_val]
                outputs_v = [nets[i](inputs_val[i], H_val[i]) for i in range(4)]
                haz4_v = torch.cat([o[0] for o in outputs_v], dim=1)
                logits4_v = torch.stack([o[1] for o in outputs_v], dim=1)
                emb4_v = [o[2] for o in outputs_v]
                
                haz4_v = gate(haz4_v)
                haz_f_v, attn_v = fuse(emb4_v)
                logits_f_v = (attn_v.unsqueeze(-1) * logits4_v).sum(dim=1)
                
                probs = F.softmax(logits_f_v, dim=1)[:, 1].cpu().numpy()
                auc = roc_auc_score(e_val.cpu().numpy(), probs)
                
                if auc > best_auc:
                    best_auc = auc; counter = 0
                    best_state = {
                        'nets': {k: v.detach().cpu().clone() for k, v in nets.state_dict().items()},
                        'gate': {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()},
                        'fuse': {k: v.detach().cpu().clone() for k, v in fuse.state_dict().items()},
                        'epoch': epoch, 'auc': auc,
                        'alpha': alpha, 'beta': beta, 'fold': fold_idx,
                    }
                else:
                    counter += 1
                    if counter >= hgnn_cfg['patience']:
                        break
    
    del nets, gate, fuse, optimizer
    torch.cuda.empty_cache()
    
    return {'fold': fold_idx, 'alpha': alpha, 'beta': beta, 'auc': best_auc, 'best_state': best_state}


def process_single_run(cfg, cancer):
    print(f"\n[INFO] {cancer} | AUC Training")
    print(f"[INFO] Config: {cfg.get('config_path', 'default.yaml')}")
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    
    paths = resolve_paths(cfg, cancer)
    ae_dir = paths['auc_data_dir']
    out_dir = paths['auc_model_dir']
    out_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    print(f"[INFO] Samples: {n_total} | Events: {int(e.sum())}")
    
    grid = cfg['grid_search']
    n_total_train = len(grid['alpha']) * len(grid['beta']) * cfg['hgnn']['n_fold']
    print(f"[INFO] Grid search: {len(grid['alpha'])} x {len(grid['beta'])} combos x {cfg['hgnn']['n_fold']} folds = {n_total_train} trainings")
    
    outer_kf = KFold(n_splits=cfg['hgnn']['n_fold'], shuffle=True, random_state=cfg['seed'])
    all_folds_data = []
    
    for fold_idx, (tr_idx, val_idx) in enumerate(outer_kf.split(range(n_total))):
        X_s_tr, X_m_tr, X_p_tr, X_r_tr = X_s[tr_idx], X_m[tr_idx], X_p[tr_idx], X_r[tr_idx]
        X_s_val, X_m_val, X_p_val, X_r_val = X_s[val_idx], X_m[val_idx], X_p[val_idx], X_r[val_idx]
        t_tr, e_tr = t[tr_idx], e[tr_idx]; t_val, e_val = t[val_idx], e[val_idx]
        
        H_tr_full = [build_H(X_s_tr, k=cfg['hgnn']['k_neighbor'], device=device),
                     build_H(X_m_tr, k=cfg['hgnn']['k_neighbor'], device=device),
                     build_H(X_p_tr, k=cfg['hgnn']['k_neighbor'], device=device),
                     build_H(X_r_tr, k=cfg['hgnn']['k_neighbor'], device=device)]
        H_val_full = [build_H(X_s_val, k=cfg['hgnn']['k_neighbor'], device=device),
                      build_H(X_m_val, k=cfg['hgnn']['k_neighbor'], device=device),
                      build_H(X_p_val, k=cfg['hgnn']['k_neighbor'], device=device),
                      build_H(X_r_val, k=cfg['hgnn']['k_neighbor'], device=device)]
        
        train_means = [X_s_tr.mean(axis=0), X_m_tr.mean(axis=0),
                       X_p_tr.mean(axis=0), X_r_tr.mean(axis=0)]
        
        all_folds_data.append({
            'fold': fold_idx,
            'X_s_tr': X_s_tr, 'X_m_tr': X_m_tr, 'X_p_tr': X_p_tr, 'X_r_tr': X_r_tr,
            'X_s_val': X_s_val, 'X_m_val': X_m_val, 'X_p_val': X_p_val, 'X_r_val': X_r_val,
            't_tr': t_tr, 'e_tr': e_tr, 't_val': t_val, 'e_val': e_val,
            'H_tr': H_tr_full, 'H_val': H_val_full,
            'train_means': train_means,
        })
    
    all_tasks = []
    for fold_data in all_folds_data:
        for A in grid['alpha']:
            for B in grid['beta']:
                all_tasks.append((fold_data['fold'], A, B, fold_data, cfg))
    
    print(f"[INFO] Starting parallel training...")
    ctx = get_context('spawn')
    all_results = []
    
    with ctx.Pool(processes=grid.get('max_workers', 40)) as pool:
        for result in tqdm(
            pool.imap_unordered(train_single_fold, all_tasks),
            total=len(all_tasks),
            desc="AUC search"
        ):
            all_results.append(result)
    
    combo_results = defaultdict(lambda: {'aucs': [], 'folds': []})
    for res in all_results:
        key = (res['alpha'], res['beta'])
        combo_results[key]['aucs'].append(res['auc'])
        combo_results[key]['folds'].append(res['fold'])
    
    combo_summary = []
    for (A, B), data in combo_results.items():
        mean_auc = np.mean(data['aucs'])
        std_auc = np.std(data['aucs'])
        combo_summary.append({
            'alpha': A, 'beta': B,
            'mean_auc': mean_auc, 'std_auc': std_auc,
            'fold_aucs': data['aucs'],
        })
    
    combo_summary.sort(key=lambda x: x['mean_auc'], reverse=True)
    
    print(f"\n[INFO] Top 10 hyperparameter combinations:")
    for i, item in enumerate(combo_summary[:10]):
        print(f"  {i+1}. (alpha={item['alpha']}, beta={item['beta']}) -> Mean AUC: {item['mean_auc']:.4f} ± {item['std_auc']:.4f} | Folds: {np.round(item['fold_aucs'], 4).tolist()}")
    
    best_combo = combo_summary[0]
    best_A, best_B = best_combo['alpha'], best_combo['beta']
    print(f"\n[INFO] Best: alpha={best_A}, beta={best_B}")
    print(f"[INFO] Mean AUC: {best_combo['mean_auc']:.4f} ± {best_combo['std_auc']:.4f}")
    
    for res in all_results:
        if res['alpha'] == best_A and res['beta'] == best_B and res['best_state'] is not None:
            fold_idx = res['fold']
            save_path = out_dir / f"best_model_fold{fold_idx}.pth"
            torch.save(res['best_state'], save_path)
            print(f"[INFO] Saved: {save_path}")
    
    final_summary = {
        'cancer': cancer,
        'n_samples': int(n_total), 'n_events': int(e.sum()),
        'best_params': {'alpha': best_A, 'beta': best_B},
        'best_mean_auc': float(best_combo['mean_auc']),
        'best_std_auc': float(best_combo['std_auc']),
        'fold_aucs': [float(c) for c in best_combo['fold_aucs']],
        'all_combos': combo_summary[:50],
    }
    
    with open(out_dir / 'final_summary.json', 'w') as f:
        json.dump(final_summary, f, indent=2)
    print(f"[INFO] Saved: {out_dir / 'final_summary.json'}")
    print(f"[INFO] Completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--cancer', type=str, required=True)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    cfg = merge_cli_args(cfg, args)
    cfg['config_path'] = args.config
    process_single_run(cfg, args.cancer)


if __name__ == '__main__':
    main()