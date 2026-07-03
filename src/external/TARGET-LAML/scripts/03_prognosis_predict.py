#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = Path('')
FEATURE_DIR = BASE_DIR / 'TARGET-LAML' / 'results'
CLINICAL_PATH = BASE_DIR / 'TARGET-LAML' / 'data' / 'TARGET_clinical_delete_process.csv'
MODEL_DIR = BASE_DIR / 'TARGET-LAML' / 'Cancer-LAML' / 'model_hgnn'
OUTPUT_DIR = FEATURE_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class HGNNconv(nn.Module):
    def __init__(self, in_dim=64, hid=64, p=0.3):
        super().__init__()
        self.W = nn.Linear(in_dim, hid)
        self.dropout = nn.Dropout(p)
    def forward(self, X, H):
        dv = 1. / (H.sum(1) + 1e-6).clamp(min=1e-6)
        de = 1. / (H.sum(0) + 1e-6).clamp(min=1e-6)
        Xe = de.view(-1, 1) * (H.t() @ X)
        Xv = (H @ Xe) * dv.view(-1, 1)
        return self.dropout(F.relu(self.W(Xv)))

class HazardHGNN(nn.Module):
    def __init__(self, idx=0):
        super().__init__()
        self.conv = HGNNconv(64, 64)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))
    def forward(self, X, H):
        emb = self.conv(X, H)
        haz = self.head(emb)
        return haz, emb

class ModalityCausalGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_ca = nn.Parameter(torch.zeros(4))
        self.register_buffer('effect_prior', torch.zeros(4))
        self.register_buffer('ema_effects', torch.ones(4) * 0.25)
    def forward(self, haz4):
        combined = self.w_ca + self.effect_prior
        w = F.softmax(combined, dim=0)
        return haz4 * w.view(1, 4), w

class MultiHeadAttnFusion(nn.Module):
    def __init__(self, n_heads=4, hid=64):
        super().__init__()
        self.n_heads = n_heads
        self.hid = hid
        self.q = nn.Sequential(nn.Linear(4 * hid, hid), nn.Tanh(), nn.Linear(hid, n_heads))
        self.out_proj = nn.Linear(n_heads * hid, 1, bias=False)
    def forward(self, emb4):
        emb_stack = torch.stack(emb4, dim=1)
        batch_size = emb_stack.size(0)
        emb_flat = emb_stack.view(batch_size, -1)
        logits = self.q(emb_flat)
        attn = torch.softmax(logits, dim=1)
        emb_expand = emb_stack.unsqueeze(1)
        attn_expand = attn.unsqueeze(2).unsqueeze(3)
        weighted = (attn_expand * emb_expand).sum(dim=2)
        fused = weighted.view(batch_size, self.n_heads * self.hid)
        haz_f = self.out_proj(fused)
        return haz_f, attn

def build_H(X, k=10):
    Xt = torch.as_tensor(X, device=device)
    n = Xt.size(0)
    if n == 0:
        return torch.zeros((0, 0), device=device)
    cos = F.normalize(Xt, p=2, dim=1)
    sim = cos @ cos.t()
    _, idx = torch.topk(sim, k=min(k+1, n), dim=1)
    H = torch.zeros(n, n, device=device)
    src = torch.arange(n, device=device).view(-1, 1).expand_as(idx)
    H[src.reshape(-1), idx.reshape(-1)] = 1.
    H.fill_diagonal_(0)
    return H

h_shared_df = pd.read_csv(FEATURE_DIR / 'TARGET_h_shared_64d.csv', index_col=0).astype('float32')
p_mir_df    = pd.read_csv(FEATURE_DIR / 'TARGET_p_mir_64d.csv',    index_col=0).astype('float32')
p_meth_df   = pd.read_csv(FEATURE_DIR / 'TARGET_p_meth_64d.csv',   index_col=0).astype('float32')
p_mrna_df   = pd.read_csv(FEATURE_DIR / 'TARGET_p_mrna_64d.csv',   index_col=0).astype('float32')
cli = pd.read_csv(CLINICAL_PATH, index_col=0)

common_idx = (h_shared_df.index.intersection(p_mir_df.index)
              .intersection(p_meth_df.index).intersection(p_mrna_df.index)
              .intersection(cli.index))

X_s = h_shared_df.loc[common_idx].values.astype('float32')
X_m = p_mir_df.loc[common_idx].values.astype('float32')
X_p = p_meth_df.loc[common_idx].values.astype('float32')
X_r = p_mrna_df.loc[common_idx].values.astype('float32')

status_map = {'1:DECEASED':1, '0:LIVING':0, 'DECEASED':1, 'LIVING':0, 'Dead':1, 'Alive':0, '1':1, '0':0}
e_true = cli['os_status'].astype(str).map(status_map).values
t_true = cli['os_time'].values.astype(float)
pat_id = common_idx

print(f"Aligned samples: {len(common_idx)}, Events: {int(e_true.sum())}")

inputs = [
    torch.as_tensor(X_s, device=device),
    torch.as_tensor(X_m, device=device),
    torch.as_tensor(X_p, device=device),
    torch.as_tensor(X_r, device=device)
]
H_ext = [build_H(X_s), build_H(X_m), build_H(X_p), build_H(X_r)]

def plot_km(t_true, e_true, mask_high, mask_low, title, save_path, ci=None, p=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    kmf = KaplanMeierFitter()
    lh = f'High Risk (n={mask_high.sum()}, events={int(e_true[mask_high].sum())})'
    ll = f'Low Risk  (n={mask_low.sum()},  events={int(e_true[mask_low].sum())})'
    for mask, color, label in [(mask_high, '#e74c3c', lh), (mask_low, '#3498db', ll)]:
        kmf.fit(t_true[mask], event_observed=e_true[mask], label=label)
        kmf.plot_survival_function(ax=ax, color=color, linewidth=3.0, ci_show=True)
    parts = [title]
    if p is not None: parts.append(f'Log-rank p={p:.2e}')
    if ci is not None: parts.append(f'C-index={ci:.3f}')
    ax.set_title('\n'.join(parts), fontsize=15, fontweight='bold')
    ax.set_xlabel('Time (days)', fontsize=14)
    ax.set_ylabel('Survival Probability', fontsize=14)
    ax.legend(fontsize=12, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()

print(f"\n{'='*70}")
print("External Validation (5-Fold Ensemble, No Training)")
print(f"{'='*70}")

all_hazards = []

for fold in range(5):
    ckpt_path = MODEL_DIR / f'best_model_fold{fold}.pth'
    if not ckpt_path.exists():
        continue

    ckpt = torch.load(ckpt_path, map_location=device)
    nets = nn.ModuleList([HazardHGNN(i).to(device) for i in range(4)])
    gate = ModalityCausalGate().to(device)
    fuse = MultiHeadAttnFusion().to(device)
    nets.load_state_dict(ckpt['nets'])
    gate.load_state_dict(ckpt['gate'])
    fuse.load_state_dict(ckpt['fuse'])
    nets.eval(); gate.eval(); fuse.eval()

    with torch.no_grad():
        outputs = [nets[i](inputs[i], H_ext[i]) for i in range(4)]
        haz4 = torch.cat([o[0] for o in outputs], dim=1)
        emb4 = [o[1] for o in outputs]
        haz4_gated, _ = gate(haz4)
        haz_f, _ = fuse(emb4)
        haz_np = haz_f.cpu().numpy().flatten()

    all_hazards.append(haz_np)

    del nets, gate, fuse, ckpt
    torch.cuda.empty_cache()

vote_hazard = np.mean(all_hazards, axis=0)

ci_ens = concordance_index(t_true, -vote_hazard, e_true)
median_haz = np.median(vote_hazard)
high_mask = vote_hazard >= median_haz
low_mask = ~high_mask

lr = logrank_test(
    t_true[high_mask], t_true[low_mask],
    event_observed_A=e_true[high_mask],
    event_observed_B=e_true[low_mask]
)

print(f"\n{'='*70}")
print("[5-Fold Ensemble Prediction Results]")
print(f"{'='*70}")
print(f"Ensemble C-index:     {ci_ens:.4f}")
print(f"Log-rank p-value:     {lr.p_value:.2e}")

plot_km(t_true, e_true, high_mask, low_mask,
        'TARGET-LAML (5-Fold Ensemble)',
        OUTPUT_DIR / 'TARGET_KM_ensemble.png',
        ci=ci_ens, p=lr.p_value)

pd.DataFrame({
    'patient_id': pat_id,
    'risk_score': vote_hazard,
    'true_label': e_true
}).to_csv(OUTPUT_DIR / 'TARGET_ensemble_predictions.csv', index=False)

print(f"\nPredictions saved to: {OUTPUT_DIR / 'TARGET_ensemble_predictions.csv'}")
print(f"{'='*70}")