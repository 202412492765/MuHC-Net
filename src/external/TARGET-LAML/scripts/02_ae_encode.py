#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path('')
RESULT_DIR = BASE_DIR / 'results'
AE_DIR = BASE_DIR / 'Cancer-LAML' / 'model_ae'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

print("="*60)
print("Cell 2: AE Encoding")
print("="*60)

class Enc(nn.Module):
    def __init__(self, in_dim, hid=64):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(in_dim, hid), nn.LeakyReLU(0.1))
        self.priv   = nn.Sequential(nn.Linear(in_dim, hid), nn.LeakyReLU(0.1))
    def forward(self, x):
        return self.shared(x), self.priv(x)

class AttnFusion(nn.Module):
    def __init__(self, hid=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3*hid, 64), nn.Tanh(), nn.Linear(64, 3))
    def forward(self, s_mir, s_meth, s_mrna):
        s_cat = torch.cat([s_mir, s_meth, s_mrna], dim=1)
        alpha = torch.softmax(self.net(s_cat), dim=1)
        h = alpha[:,0:1]*s_mir + alpha[:,1:2]*s_meth + alpha[:,2:3]*s_mrna
        return h, alpha

X_mir  = np.load(RESULT_DIR / 'TARGET_X_mir_200.npy')
X_meth = np.load(RESULT_DIR / 'TARGET_X_meth_600.npy')
X_mrna = np.load(RESULT_DIR / 'TARGET_X_mrna_600.npy')
pat_id = pd.read_csv(RESULT_DIR / 'TARGET_common_index.csv')['patient_id'].values

print(f"Input dims: mir={X_mir.shape[1]}, meth={X_meth.shape[1]}, mrna={X_mrna.shape[1]}")
print(f"Samples: {len(pat_id)}")

enc_mir  = Enc(X_mir.shape[1],  64).to(device)
enc_meth = Enc(X_meth.shape[1], 64).to(device)
enc_mrna = Enc(X_mrna.shape[1], 64).to(device)
attn_fus = AttnFusion(64).to(device)

ae_model_path = AE_DIR / 'ae_model.pth'
print(f"Loading AE weights: {ae_model_path}")
ckpt = torch.load(ae_model_path, map_location=device)
enc_mir.load_state_dict(ckpt['enc_mir'])
enc_meth.load_state_dict(ckpt['enc_meth'])
enc_mrna.load_state_dict(ckpt['enc_mrna'])
attn_fus.load_state_dict(ckpt['attn_fusion'])

x_mir  = torch.tensor(X_mir,  dtype=torch.float32, device=device)
x_meth = torch.tensor(X_meth, dtype=torch.float32, device=device)
x_mrna = torch.tensor(X_mrna, dtype=torch.float32, device=device)

enc_mir.eval(); enc_meth.eval(); enc_mrna.eval(); attn_fus.eval()
with torch.no_grad():
    s_mir,  p_mir  = enc_mir(x_mir)
    s_meth, p_meth = enc_meth(x_meth)
    s_mrna, p_mrna = enc_mrna(x_mrna)
    h_shared, alpha = attn_fus(s_mir, s_meth, s_mrna)

h_shared_df = pd.DataFrame(h_shared.cpu().numpy(), index=pat_id, columns=[f'shared_{i}' for i in range(64)])
p_mir_df    = pd.DataFrame(p_mir.cpu().numpy(),    index=pat_id, columns=[f'priv_mir_{i}' for i in range(64)])
p_meth_df   = pd.DataFrame(p_meth.cpu().numpy(),   index=pat_id, columns=[f'priv_meth_{i}' for i in range(64)])
p_mrna_df   = pd.DataFrame(p_mrna.cpu().numpy(),   index=pat_id, columns=[f'priv_mrna_{i}' for i in range(64)])

h_shared_df.to_csv(RESULT_DIR / 'TARGET_h_shared_64d.csv')
p_mir_df.to_csv(RESULT_DIR / 'TARGET_p_mir_64d.csv')
p_meth_df.to_csv(RESULT_DIR / 'TARGET_p_meth_64d.csv')
p_mrna_df.to_csv(RESULT_DIR / 'TARGET_p_mrna_64d.csv')

print(f"\nDone.")
print(f"  h_shared: {h_shared_df.shape}")
print(f"  p_mir: {p_mir_df.shape}")
print(f"  p_meth: {p_meth_df.shape}")
print(f"  p_mrna: {p_mrna_df.shape}")
print(f"  Attention (first sample): mir={alpha[0,0].item():.3f}, meth={alpha[0,1].item():.3f}, mrna={alpha[0,2].item():.3f}")