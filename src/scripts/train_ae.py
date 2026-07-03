#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train MSPAE for a given cancer type.
Outputs: ae_model.pth, 4 representation csvs, AE_training.png
All saved under results/{cancer}/result_CI/data/ and models/.
"""

import os
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.ae import Enc, AttnFusion
from src.utils.config import load_config, merge_cli_args
from src.utils.train_utils import seed_everything


def cosine_orth_loss(s, p, eps=1e-8):
    p_norm = torch.norm(p, p=2, dim=1, keepdim=True)
    if torch.mean(p_norm) < eps:
        return torch.tensor(1.0, device=p.device)
    s_norm = F.normalize(s, p=2, dim=1, eps=eps)
    p_norm = F.normalize(p, p=2, dim=1, eps=eps)
    cos_sim = torch.sum(s_norm * p_norm, dim=1)
    return torch.mean(torch.abs(cos_sim))


def var_preservation(p, min_var=0.001):
    var = torch.var(p, dim=0).mean()
    return F.relu(min_var - var) * 10.0


def train_ae(cfg, cancer):
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    ae_cfg = cfg['ae']
    
    data_dir = Path(cfg['data_dir']) / cancer
    out_dir = Path(cfg['results_dir']) / cancer / "result_CI"
    data_out = out_dir / "data"
    model_out = out_dir / "models"
    data_out.mkdir(parents=True, exist_ok=True)
    model_out.mkdir(parents=True, exist_ok=True)
    
    x_mir = torch.tensor(pd.read_csv(data_dir / f"{cancer}_miRNA_preprocess.csv", index_col=0).values, dtype=torch.float32).to(device)
    x_meth = torch.tensor(pd.read_csv(data_dir / f"{cancer}_meth_preprocess.csv", index_col=0).values, dtype=torch.float32).to(device)
    x_mrna = torch.tensor(pd.read_csv(data_dir / f"{cancer}_mRNA_preprocess.csv", index_col=0).values, dtype=torch.float32).to(device)
    
    n_samples = x_mir.shape[0]
    print(f"[INFO] {cancer} | AE Training")
    print(f"[INFO] Config: {cfg.get('config_path', 'default.yaml')}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Samples: {n_samples} | miRNA: {x_mir.shape[1]} | methylation: {x_meth.shape[1]} | mRNA: {x_mrna.shape[1]}")
    print(f"[INFO] lambdas: align={ae_cfg['lambda_align']} | orth={ae_cfg['lambda_orth']} | preserve={ae_cfg['lambda_preserve']}")
    
    enc_mir = Enc(x_mir.shape[1], ae_cfg['hidden_dim']).to(device)
    enc_meth = Enc(x_meth.shape[1], ae_cfg['hidden_dim']).to(device)
    enc_mrna = Enc(x_mrna.shape[1], ae_cfg['hidden_dim']).to(device)
    attn_fusion = AttnFusion(ae_cfg['hidden_dim']).to(device)
    
    decoders = {
        'shared_mir': nn.Sequential(nn.Linear(64, x_mir.shape[1]), nn.LeakyReLU(0.1)).to(device),
        'shared_meth': nn.Sequential(nn.Linear(64, x_meth.shape[1]), nn.LeakyReLU(0.1)).to(device),
        'shared_mrna': nn.Sequential(nn.Linear(64, x_mrna.shape[1]), nn.LeakyReLU(0.1)).to(device),
        'priv_mir': nn.Sequential(nn.Linear(64, x_mir.shape[1]), nn.LeakyReLU(0.1)).to(device),
        'priv_meth': nn.Sequential(nn.Linear(64, x_meth.shape[1]), nn.LeakyReLU(0.1)).to(device),
        'priv_mrna': nn.Sequential(nn.Linear(64, x_mrna.shape[1]), nn.LeakyReLU(0.1)).to(device),
    }
    
    params = (list(enc_mir.parameters()) + list(enc_meth.parameters()) + list(enc_mrna.parameters()) +
              list(attn_fusion.parameters()) + sum([list(d.parameters()) for d in decoders.values()], []))
    optimizer = torch.optim.Adam(params, lr=ae_cfg['lr'])
    
    seed_everything(cfg['seed'])
    
    log = {k: [] for k in ['epoch', 'loss', 'recon', 'align', 'orth', 'preserve',
                           'a_mir', 'a_meth', 'a_mrna', 'var_mir', 'var_meth', 'var_mrna']}
    
    for epoch in range(1, ae_cfg['epochs'] + 1):
        optimizer.zero_grad()
        
        s_mir, p_mir = enc_mir(x_mir)
        s_meth, p_meth = enc_meth(x_meth)
        s_mrna, p_mrna = enc_mrna(x_mrna)
        h_shared, alpha = attn_fusion(s_mir, s_meth, s_mrna)
        
        rec_mir = decoders['shared_mir'](h_shared) + decoders['priv_mir'](p_mir)
        rec_meth = decoders['shared_meth'](h_shared) + decoders['priv_meth'](p_meth)
        rec_mrna = decoders['shared_mrna'](h_shared) + decoders['priv_mrna'](p_mrna)
        
        loss_recon = (F.mse_loss(rec_mir, x_mir) + F.mse_loss(rec_meth, x_meth) +
                      F.mse_loss(rec_mrna, x_mrna))
        loss_align = (F.mse_loss(s_mir, h_shared) + F.mse_loss(s_meth, h_shared) +
                      F.mse_loss(s_mrna, h_shared))
        loss_orth = (cosine_orth_loss(s_mir, p_mir) +
                     cosine_orth_loss(s_meth, p_meth) +
                     cosine_orth_loss(s_mrna, p_mrna))
        loss_preserve = var_preservation(p_mir) + var_preservation(p_meth) + var_preservation(p_mrna)
        
        loss = (loss_recon + ae_cfg['lambda_align'] * loss_align +
                ae_cfg['lambda_orth'] * loss_orth +
                ae_cfg['lambda_preserve'] * loss_preserve)
        
        if torch.isnan(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=ae_cfg['clip_grad_norm'])
        optimizer.step()
        
        if epoch % 50 == 0 or epoch == ae_cfg['epochs']:
            with torch.no_grad():
                var_m = torch.var(p_mir).item()
                var_me = torch.var(p_meth).item()
                var_r = torch.var(p_mrna).item()
                log['epoch'].append(epoch); log['loss'].append(loss.item())
                log['recon'].append(loss_recon.item()); log['align'].append(loss_align.item())
                log['orth'].append(loss_orth.item()); log['preserve'].append(loss_preserve.item())
                log['a_mir'].append(alpha[0, 0].item()); log['a_meth'].append(alpha[0, 1].item()); log['a_mrna'].append(alpha[0, 2].item())
                log['var_mir'].append(var_m); log['var_meth'].append(var_me); log['var_mrna'].append(var_r)
                print(f"[INFO] Epoch {epoch:03d}/{ae_cfg['epochs']} | loss={loss.item():.4f} | recon={loss_recon.item():.4f} | align={loss_align.item():.4f} | orth={loss_orth.item():.4f} | preserve={loss_preserve.item():.4f}")
    
    enc_mir.eval(); enc_meth.eval(); enc_mrna.eval(); attn_fusion.eval()
    with torch.no_grad():
        s_mir, p_mir = enc_mir(x_mir)
        s_meth, p_meth = enc_meth(x_meth)
        s_mrna, p_mrna = enc_mrna(x_mrna)
        h_shared, _ = attn_fusion(s_mir, s_meth, s_mrna)
        
        pat_id = pd.read_csv(data_dir / f"{cancer}_miRNA_preprocess.csv", index_col=0).index
        
        pd.DataFrame(h_shared.cpu().numpy(), index=pat_id, columns=[f'shared_{i}' for i in range(64)]).to_csv(data_out / 'h_shared_64d.csv')
        pd.DataFrame(p_mir.cpu().numpy(), index=pat_id, columns=[f'priv_mir_{i}' for i in range(64)]).to_csv(data_out / 'p_mir_64d.csv')
        pd.DataFrame(p_meth.cpu().numpy(), index=pat_id, columns=[f'priv_meth_{i}' for i in range(64)]).to_csv(data_out / 'p_meth_64d.csv')
        pd.DataFrame(p_mrna.cpu().numpy(), index=pat_id, columns=[f'priv_mrna_{i}' for i in range(64)]).to_csv(data_out / 'p_mrna_64d.csv')
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    plot_items = [
        ('loss', 'Total Loss'), ('recon', 'Recon'), ('align', 'Align'), ('orth', 'Orth'),
        ('preserve', 'Preserve'), ('a_mir', 'Attn(mir)'), ('a_meth', 'Attn(meth)'), ('a_mrna', 'Attn(mrna)')
    ]
    for i, (k, t) in enumerate(plot_items):
        axes[i].plot(log['epoch'], log[k]); axes[i].set_title(t); axes[i].set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(data_out.parent / 'AE_training.png', dpi=300)
    plt.close()
    
    model_path = model_out / 'ae_model.pth'
    torch.save({
        'enc_mir': enc_mir.state_dict(), 'enc_meth': enc_meth.state_dict(), 'enc_mrna': enc_mrna.state_dict(),
        'attn_fusion': attn_fusion.state_dict(),
        'dec_shared_mir': decoders['shared_mir'].state_dict(), 'dec_shared_meth': decoders['shared_meth'].state_dict(), 'dec_shared_mrna': decoders['shared_mrna'].state_dict(),
        'dec_priv_mir': decoders['priv_mir'].state_dict(), 'dec_priv_meth': decoders['priv_meth'].state_dict(), 'dec_priv_mrna': decoders['priv_mrna'].state_dict(),
        'optimizer': optimizer.state_dict(), 'epoch': ae_cfg['epochs'],
        'lambda_align': ae_cfg['lambda_align'], 'lambda_orth': ae_cfg['lambda_orth'], 'lambda_preserve': ae_cfg['lambda_preserve'],
    }, model_path)
    
    print(f"[INFO] Saved: {data_out / 'h_shared_64d.csv'}")
    print(f"[INFO] Saved: {data_out / 'p_mir_64d.csv'}")
    print(f"[INFO] Saved: {data_out / 'p_meth_64d.csv'}")
    print(f"[INFO] Saved: {data_out / 'p_mrna_64d.csv'}")
    print(f"[INFO] Saved: {model_path}")
    print(f"[INFO] Saved: {data_out.parent / 'AE_training.png'}")
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
    train_ae(cfg, args.cancer)


if __name__ == '__main__':
    main()