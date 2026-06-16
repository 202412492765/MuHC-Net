#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train AUC survival classification model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import shutil
import yaml
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from multiprocessing import get_context
from collections import defaultdict

from src.models.hgnn import HazardHGNN, CausalGate, MultiHeadAttnFusion
from src.models.losses import ConsistencyLoss, ranking_distill_loss
from src.utils.train_utils import seed_everything, delta_ci
from src.utils.graph_utils import build_H

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_config():
    cfg_path = Path(__file__).resolve().parents[2] / 'configs' / 'default.yaml'
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


cfg = load_config()
CANCER = cfg['project']['cancer']
ROOT = Path(cfg['paths']['root']).resolve()
DATA_DIR = ROOT / cfg['paths']['results'] / 'result_AE' / 'data'
CLINICAL_DIR = ROOT / cfg['paths']['data']
SAVE_DIR = ROOT / cfg['paths']['results'] / 'result_AUC'
CHECKPOINT_DIR = ROOT / cfg['paths']['checkpoints'] / 'AUC'

k_neighbor = cfg['hyperparams']['hgnn']['k_neighbor']
n_fold = cfg['hyperparams']['hgnn']['n_fold']
patience = cfg['hyperparams']['hgnn']['patience']
n_epochs = cfg['hyperparams']['hgnn']['epochs']
DROP_P = cfg['hyperparams']['hgnn']['drop_p']
ALPHA_GRID = cfg['hyperparams']['grid_search']['alpha']
BETA_GRID = cfg['hyperparams']['grid_search']['beta']
MAX_WORKERS = cfg['hyperparams']['grid_search']['max_workers']


def train_single_fold(args):
    fold_idx, ALPHA, BETA, fold_data, save_dir = args
    device_local = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(42 + fold_idx + int(ALPHA * 10) + int(BETA * 10))

    def to_tensor(x, dtype=torch.float32):
        return torch.as_tensor(x, device=device_local, dtype=dtype)

    x_s_tr = to_tensor(fold_data['X_s_tr'])
    x_m_tr = to_tensor(fold_data['X_m_tr'])
    x_p_tr = to_tensor(fold_data['X_p_tr'])
    x_r_tr = to_tensor(fold_data['X_r_tr'])
    x_s_val = to_tensor(fold_data['X_s_val'])
    x_m_val = to_tensor(fold_data['X_m_val'])
    x_p_val = to_tensor(fold_data['X_p_val'])
    x_r_val = to_tensor(fold_data['X_r_val'])
    t_tr = to_tensor(fold_data['t_tr'])
    e_tr = to_tensor(fold_data['e_tr'], dtype=torch.long)
    t_val = to_tensor(fold_data['t_val'])
    e_val = to_tensor(fold_data['e_val'], dtype=torch.long)

    H_tr = [to_tensor(h) for h in fold_data['H_tr']]
    H_val = [to_tensor(h) for h in fold_data['H_val']]
    delta_star = fold_data['delta_star']

    nets = nn.ModuleList([HazardHGNN(i, has_cls=True).to(device_local) for i in range(4)])
    gate = CausalGate(init_weights=delta_star.tolist() if isinstance(delta_star, np.ndarray) else None).to(device_local)
    fuse = MultiHeadAttnFusion().to(device_local)

    optimizer = optim.Adam(
        list(nets.parameters()) + list(gate.parameters()) + list(fuse.parameters()),
        lr=cfg['hyperparams']['hgnn']['lr'], weight_decay=cfg['hyperparams']['hgnn']['weight_decay']
    )
    consistency_fn = ConsistencyLoss()

    best_auc = 0.0
    counter = 0
    best_state = None

    for epoch in range(1, n_epochs + 1):
        for net in nets:
            net.train()
        gate.train()
        fuse.train()
        optimizer.zero_grad()

        inputs_tr = [x_s_tr, x_m_tr, x_p_tr, x_r_tr]
        outputs = [nets[i](inputs_tr[i], H_tr[i]) for i in range(4)]
        haz4 = torch.cat([o[0] for o in outputs], dim=1)
        logits4 = torch.stack([o[1] for o in outputs], dim=1)

        haz4 = gate(haz4)
        haz_f, attn = fuse(haz4)
        logits_f = (attn.unsqueeze(-1) * logits4).sum(dim=1)

        loss_main = F.cross_entropy(logits_f, e_tr)
        loss_cons = consistency_fn(haz4, t_tr, e_tr)
        loss_dis = ranking_distill_loss(haz_f, haz4, t_tr, e_tr)

        loss = loss_main + ALPHA * loss_cons + BETA * loss_dis

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(nets.parameters()) + list(gate.parameters()) + list(fuse.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        if epoch % 10 == 0:
            for net in nets:
                net.eval()
            gate.eval()
            fuse.eval()
            with torch.no_grad():
                inputs_val = [x_s_val, x_m_val, x_p_val, x_r_val]
                outputs_v = [nets[i](inputs_val[i], H_val[i]) for i in range(4)]
                haz4_v = torch.cat([o[0] for o in outputs_v], dim=1)
                logits4_v = torch.stack([o[1] for o in outputs_v], dim=1)

                haz4_v = gate(haz4_v)
                haz_f_v, attn_v = fuse(haz4_v)
                logits_f_v = (attn_v.unsqueeze(-1) * logits4_v).sum(dim=1)

                probs = F.softmax(logits_f_v, dim=1)[:, 1].cpu().numpy()
                auc = roc_auc_score(e_val.cpu().numpy(), probs)

                if auc > best_auc:
                    best_auc = auc
                    counter = 0
                    best_state = {
                        'nets': {k: v.detach().cpu().clone() for k, v in nets.state_dict().items()},
                        'gate': {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()},
                        'fuse': {k: v.detach().cpu().clone() for k, v in fuse.state_dict().items()},
                        'epoch': epoch,
                        'auc': auc
                    }
                else:
                    counter += 1
                    if counter >= patience:
                        break

    save_name = f'fold{fold_idx}_A{ALPHA:.1f}_B{BETA:.1f}_AUC{best_auc:.4f}.pth'
    save_path = save_dir / 'checkpoints' / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        torch.save(best_state, save_path)

    del nets, gate, fuse, optimizer
    torch.cuda.empty_cache()

    return {'fold': fold_idx, 'alpha': ALPHA, 'beta': BETA, 'auc': best_auc, 'save_path': str(save_path)}


def process_single_run():
    print('=' * 70)
    print(f'{CANCER} AUC Survival Classification Training')
    print('=' * 70)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

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
    n_total = len(t)

    print(f'[{CANCER} result_AUC] n_total: {n_total}, events: {int(e.sum())}')
    print(f'Hyperparameter search: {len(ALPHA_GRID) * len(BETA_GRID)} combinations x {n_fold} folds = {len(ALPHA_GRID) * len(BETA_GRID) * n_fold} training runs')

    outer_kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    all_folds_data = []

    for fold_idx, (tr_idx, val_idx) in enumerate(outer_kf.split(range(n_total))):
        X_s_tr, X_m_tr, X_p_tr, X_r_tr = X_s[tr_idx], X_m[tr_idx], X_p[tr_idx], X_r[tr_idx]
        X_s_val, X_m_val, X_p_val, X_r_val = X_s[val_idx], X_m[val_idx], X_p[val_idx], X_r[val_idx]
        t_tr, e_tr = t[tr_idx], e[tr_idx]
        t_val, e_val = t[val_idx], e[val_idx]

        H_tr_full = [build_H(X_s_tr, k=k_neighbor), build_H(X_m_tr, k=k_neighbor),
                     build_H(X_p_tr, k=k_neighbor), build_H(X_r_tr, k=k_neighbor)]
        H_val_full = [build_H(X_s_val, k=k_neighbor), build_H(X_m_val, k=k_neighbor),
                      build_H(X_p_val, k=k_neighbor), build_H(X_r_val, k=k_neighbor)]

        inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)
        delta_buf = []

        for inner_tr, _ in inner_kf.split(tr_idx):
            inner_tr_global = tr_idx[inner_tr]
            sub_x = [
                torch.as_tensor(X_s[inner_tr_global], device=device),
                torch.as_tensor(X_m[inner_tr_global], device=device),
                torch.as_tensor(X_p[inner_tr_global], device=device),
                torch.as_tensor(X_r[inner_tr_global], device=device)
            ]
            sub_t = torch.as_tensor(t[inner_tr_global], device=device)
            sub_e = torch.as_tensor(e[inner_tr_global], device=device)

            sub_h = []
            for i in range(4):
                X_tmp = sub_x[i].cpu().numpy()
                sub_h.append(torch.as_tensor(build_H(X_tmp, k=k_neighbor), device=device))

            nets_tmp = nn.ModuleList([HazardHGNN(has_cls=True).to(device) for _ in range(4)])
            gate_tmp = CausalGate().to(device)

            with torch.no_grad():
                haz4 = torch.cat([nets_tmp[i](sub_x[i], sub_h[i])[0] for i in range(4)], dim=1)
                haz4 = gate_tmp(haz4)

            delta_buf.append(delta_ci(
                haz4.cpu().numpy(), sub_t.cpu().numpy(), sub_e.cpu().numpy()
            ))

        delta_star = np.median(delta_buf, axis=0)

        all_folds_data.append({
            'fold': fold_idx,
            'X_s_tr': X_s_tr, 'X_m_tr': X_m_tr, 'X_p_tr': X_p_tr, 'X_r_tr': X_r_tr,
            'X_s_val': X_s_val, 'X_m_val': X_m_val, 'X_p_val': X_p_val, 'X_r_val': X_r_val,
            't_tr': t_tr, 'e_tr': e_tr, 't_val': t_val, 'e_val': e_val,
            'H_tr': H_tr_full, 'H_val': H_val_full,
            'delta_star': delta_star
        })

    all_tasks = []
    for fold_data in all_folds_data:
        for A in ALPHA_GRID:
            for B in BETA_GRID:
                all_tasks.append((fold_data['fold'], A, B, fold_data, CHECKPOINT_DIR))

    print(f'\nTotal tasks: {len(all_tasks)} ({len(ALPHA_GRID) * len(BETA_GRID)} hyperparams x {n_fold} folds)')

    print('Starting parallel training...')
    ctx = get_context('spawn')
    all_results = []

    with ctx.Pool(processes=MAX_WORKERS) as pool:
        for result in tqdm(
            pool.imap_unordered(train_single_fold, all_tasks),
            total=len(all_tasks),
            desc='AUC search'
        ):
            all_results.append(result)

    combo_results = defaultdict(lambda: {'aucs': [], 'folds': [], 'paths': []})

    for res in all_results:
        key = (res['alpha'], res['beta'])
        combo_results[key]['aucs'].append(res['auc'])
        combo_results[key]['folds'].append(res['fold'])
        combo_results[key]['paths'].append(res['save_path'])

    combo_summary = []
    for (A, B), data in combo_results.items():
        mean_auc = np.mean(data['aucs'])
        std_auc = np.std(data['aucs'])
        combo_summary.append({
            'alpha': A, 'beta': B,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'fold_aucs': data['aucs'],
            'paths': data['paths']
        })

    combo_summary.sort(key=lambda x: x['mean_auc'], reverse=True)

    print(f'\n[{CANCER} result_AUC] Top 10 Hyperparameter Combinations (based on 5-fold mean AUC):')
    for i, item in enumerate(combo_summary[:10]):
        print(f'{i+1}. (alpha={item["alpha"]}, beta={item["beta"]}) '
              f'-> Mean AUC: {item["mean_auc"]:.4f} +/- {item["std_auc"]:.4f}')

    best_combo = combo_summary[0]
    best_A, best_B = best_combo['alpha'], best_combo['beta']
    print(f'\n[{CANCER} result_AUC] Best hyperparameters: alpha={best_A}, beta={best_B}')
    print(f'5-fold AUC: {np.round(best_combo["fold_aucs"], 4)}')
    print(f'Mean +/- Std: {best_combo["mean_auc"]:.4f} +/- {best_combo["std_auc"]:.4f}')

    print(f'\n[{CANCER} result_AUC] Saving best models...')
    model_dir = SAVE_DIR / 'model_auc'
    model_dir.mkdir(parents=True, exist_ok=True)
    for fold_idx in range(n_fold):
        for res in all_results:
            if res['fold'] == fold_idx and res['alpha'] == best_A and res['beta'] == best_B:
                src_path = Path(res['save_path'])
                dst_path = model_dir / f'best_model_fold{fold_idx}.pth'
                if src_path.exists():
                    shutil.copy(src_path, dst_path)
                    print(f'  Fold {fold_idx}: {dst_path}')
                break

    final_summary = {
        'cancer': CANCER,
        'best_params': {'alpha': best_A, 'beta': best_B},
        'best_mean_auc': float(best_combo['mean_auc']),
        'best_std_auc': float(best_combo['std_auc']),
        'fold_aucs': [float(c) for c in best_combo['fold_aucs']],
        'all_combos': combo_summary[:50],
    }

    with open(model_dir / 'final_summary.json', 'w') as f:
        json.dump(final_summary, f, indent=2)

    print(f'\n[{CANCER} result_AUC] Results saved to: {SAVE_DIR}/')
    print(f'  - Best models: {model_dir}/best_model_fold*.pth')
    print(f'  - Summary: {model_dir}/final_summary.json')
    print('=' * 60)

    return final_summary


if __name__ == '__main__':
    process_single_run()