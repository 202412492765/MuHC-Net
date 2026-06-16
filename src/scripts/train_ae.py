#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Multi-omics Shared-Private Autoencoder (MSPAE).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import yaml
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.ae import Enc, AttnFusion, cosine_orth_loss, var_preservation
from src.utils.train_utils import seed_everything

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_config():
    cfg_path = Path(__file__).resolve().parents[2] / 'configs' / 'default.yaml'
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    CANCER = cfg['project']['cancer']
    ROOT = Path(cfg['paths']['root']).resolve()
    DATA_DIR = ROOT / cfg['paths']['data']
    SAVE_DIR = ROOT / cfg['paths']['results'] / 'result_AE'
    DATA_OUT = SAVE_DIR / 'data'

    LAMBDA_ALIGN = cfg['hyperparams']['ae']['lambda_align']
    LAMBDA_ORTH = cfg['hyperparams']['ae']['lambda_orth']
    LAMBDA_PRESERVE = cfg['hyperparams']['ae']['lambda_preserve']
    MAX_EPOCH = cfg['hyperparams']['ae']['epochs']
    LR = cfg['hyperparams']['ae']['lr']
    HID = cfg['hyperparams']['ae']['hid_dim']
    SEED = cfg['hyperparams']['ae']['seed']

    print('=' * 70)
    print(f'{CANCER} Autoencoder Training')
    print('=' * 70)
    print(f'[{CANCER} result_AE] Training AE | align={LAMBDA_ALIGN}, orth={LAMBDA_ORTH}, preserve={LAMBDA_PRESERVE}')

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    x_mir = torch.tensor(pd.read_csv(DATA_DIR / cfg['files']['miRNA'], index_col=0).values, dtype=torch.float32).to(device)
    x_meth = torch.tensor(pd.read_csv(DATA_DIR / cfg['files']['meth'], index_col=0).values, dtype=torch.float32).to(device)
    x_mrna = torch.tensor(pd.read_csv(DATA_DIR / cfg['files']['mRNA'], index_col=0).values, dtype=torch.float32).to(device)
    pat_id = pd.read_csv(DATA_DIR / cfg['files']['miRNA'], index_col=0).index

    enc_mir = Enc(x_mir.shape[1], HID).to(device)
    enc_meth = Enc(x_meth.shape[1], HID).to(device)
    enc_mrna = Enc(x_mrna.shape[1], HID).to(device)
    attn_fusion = AttnFusion(HID).to(device)

    decoders = {
        'shared_mir': nn.Sequential(nn.Linear(HID, x_mir.shape[1]), nn.LeakyReLU(0.1)).to(device),
        'shared_meth': nn.Sequential(nn.Linear(HID, x_meth.shape[1]), nn.LeakyReLU(0.1)).to(device),
        'shared_mrna': nn.Sequential(nn.Linear(HID, x_mrna.shape[1]), nn.LeakyReLU(0.1)).to(device),
        'priv_mir': nn.Sequential(nn.Linear(HID, x_mir.shape[1]), nn.LeakyReLU(0.1)).to(device),
        'priv_meth': nn.Sequential(nn.Linear(HID, x_meth.shape[1]), nn.LeakyReLU(0.1)).to(device),
        'priv_mrna': nn.Sequential(nn.Linear(HID, x_mrna.shape[1]), nn.LeakyReLU(0.1)).to(device),
    }

    params = (list(enc_mir.parameters()) + list(enc_meth.parameters()) + list(enc_mrna.parameters()) +
              list(attn_fusion.parameters()) + sum([list(d.parameters()) for d in decoders.values()], []))
    optimizer = torch.optim.Adam(params, lr=LR)

    seed_everything(SEED)

    log = {k: [] for k in ['epoch', 'loss', 'recon', 'align', 'orth', 'preserve',
                             'a_mir', 'a_meth', 'a_mrna', 'var_mir', 'var_meth', 'var_mrna']}

    for epoch in range(1, MAX_EPOCH + 1):
        optimizer.zero_grad()

        s_mir, p_mir = enc_mir(x_mir)
        s_meth, p_meth = enc_meth(x_meth)
        s_mrna, p_mrna = enc_mrna(x_mrna)
        h_shared, alpha = attn_fusion(s_mir, s_meth, s_mrna)

        rec_mir = decoders['shared_mir'](h_shared) + decoders['priv_mir'](p_mir)
        rec_meth = decoders['shared_meth'](h_shared) + decoders['priv_meth'](p_meth)
        rec_mrna = decoders['shared_mrna'](h_shared) + decoders['priv_mrna'](p_mrna)

        loss_recon = (F.mse_loss(rec_mir, x_mir) + F.mse_loss(rec_meth, x_meth) + F.mse_loss(rec_mrna, x_mrna))
        loss_align = (F.mse_loss(s_mir, h_shared) + F.mse_loss(s_meth, h_shared) + F.mse_loss(s_mrna, h_shared))
        loss_orth = (cosine_orth_loss(s_mir, p_mir) + cosine_orth_loss(s_meth, p_meth) + cosine_orth_loss(s_mrna, p_mrna))
        loss_preserve = var_preservation(p_mir) + var_preservation(p_meth) + var_preservation(p_mrna)

        loss = loss_recon + LAMBDA_ALIGN * loss_align + LAMBDA_ORTH * loss_orth + LAMBDA_PRESERVE * loss_preserve

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                var_m = torch.var(p_mir).item()
                var_me = torch.var(p_meth).item()
                var_r = torch.var(p_mrna).item()

                log['epoch'].append(epoch)
                log['loss'].append(loss.item())
                log['recon'].append(loss_recon.item())
                log['align'].append(loss_align.item())
                log['orth'].append(loss_orth.item())
                log['preserve'].append(loss_preserve.item())
                log['a_mir'].append(alpha[0, 0].item())
                log['a_meth'].append(alpha[0, 1].item())
                log['a_mrna'].append(alpha[0, 2].item())
                log['var_mir'].append(var_m)
                log['var_meth'].append(var_me)
                log['var_mrna'].append(var_r)

                if epoch % 50 == 0:
                    print(f'  Epoch {epoch}: loss={loss.item():.4f}, recon={loss_recon.item():.4f}, '
                          f'orth={loss_orth.item():.4f}, preserve={loss_preserve.item():.4f}, '
                          f'var_p=[{var_m:.4f},{var_me:.4f},{var_r:.4f}]')

    enc_mir.eval(); enc_meth.eval(); enc_mrna.eval(); attn_fusion.eval()
    with torch.no_grad():
        s_mir, p_mir = enc_mir(x_mir)
        s_meth, p_meth = enc_meth(x_meth)
        s_mrna, p_mrna = enc_mrna(x_mrna)
        h_shared, _ = attn_fusion(s_mir, s_meth, s_mrna)

        pd.DataFrame(h_shared.cpu().numpy(), index=pat_id, columns=[f'shared_{i}' for i in range(HID)]).to_csv(DATA_OUT / 'h_shared_64d.csv')
        pd.DataFrame(p_mir.cpu().numpy(), index=pat_id, columns=[f'priv_mir_{i}' for i in range(HID)]).to_csv(DATA_OUT / 'p_mir_64d.csv')
        pd.DataFrame(p_meth.cpu().numpy(), index=pat_id, columns=[f'priv_meth_{i}' for i in range(HID)]).to_csv(DATA_OUT / 'p_meth_64d.csv')
        pd.DataFrame(p_mrna.cpu().numpy(), index=pat_id, columns=[f'priv_mrna_{i}' for i in range(HID)]).to_csv(DATA_OUT / 'p_mrna_64d.csv')

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    plot_items = [
        ('loss', 'Total Loss'), ('recon', 'Recon'), ('align', 'Align'), ('orth', 'Orth'),
        ('preserve', 'Preserve'), ('a_mir', 'Attn(mir)'), ('a_meth', 'Attn(meth)'), ('a_mrna', 'Attn(mrna)')
    ]#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Multi-omics Shared-Private Autoencoder (MSPAE).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import yaml
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.ae import Enc, AttnFusion, cosine_orth_loss, var_preservation
from src.utils.train_utils import seed_everything

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_config():
    cfg_path = Path(__file__).resolve().parents[2] / 'configs' / 'default.yaml'
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    CANCER = cfg['project']['cancer']
    ROOT = Path(cfg['paths']['root']).resolve()
    DATA_DIR = ROOT / cfg['paths']['data']
    SAVE_DIR = ROOT / cfg['paths']['results'] / 'result_AE'
    DATA_OUT = SAVE_DIR / 'data'

    LAMBDA_ALIGN = cfg['hyperparams']['ae']['lambda_align']
    LAMBDA_ORTH = cfg['hyperparams']['ae']['lambda_orth']
    LAMBDA_PRESERVE = cfg['hyperparams']['ae']['lambda_preserve']
    MAX_EPOCH = cfg['hyperparams']['ae']['epochs']
    LR = cfg['hyperparams']['ae']['lr']
    HID = cfg['hyperparams']['ae']['hid_dim']
    SEED = cfg['hyperparams']['ae']['seed']

    print('=' * 70)
    print(f'{CANCER} Autoencoder Training')
    print('=' * 70)
    print(f'[{CANCER} result_AE] Training AE | align={LAMBDA_ALIGN}, orth={LAMBDA_ORTH}, preserve={LAMBDA_PRESERVE}')

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    x_mir = torch.tensor(pd.read_csv(DATA_DIR / cfg['files']['miRNA'], index_col=0).values, dtype=torch.float32).to(device)
    x_meth = torch.tensor(pd.read_csv(DATA_DIR / cfg['files']['meth'], index_col=0).values, dtype=torch.float32).to(device)
    x_mrna = torch.tensor(pd.read_csv(DATA_DIR / cfg['files']['mRNA'], index_col=0).values, dtype=torch.float32).to(device)
    pat_id = pd.read_csv(DATA_DIR / cfg['files']['miRNA'], index_col=0).index

    enc_mir = Enc(x_mir.shape[1], HID).to(device)
    enc_meth = Enc(x_meth.shape[1], HID).to(device)
    enc_mrna = Enc(x_mrna.shape[1], HID).to(device)
    attn_fusion = AttnFusion(HID).to(device)

    decoders = {
        'shared_mir': nn.Sequential(nn.Linear(HID, x_mir.shape[1]), nn.LeakyReLU(0.1)).to(device),
        'shared_meth': nn.Sequential(nn.Linear(HID, x_meth.shape[1]), nn.LeakyReLU(0.1)).to(device),
        'shared_mrna': nn.Sequential(nn.Linear(HID, x_mrna.shape[1]), nn.LeakyReLU(0.1)).to(device),
        'priv_mir': nn.Sequential(nn.Linear(HID, x_mir.shape[1]), nn.LeakyReLU(0.1)).to(device),
        'priv_meth': nn.Sequential(nn.Linear(HID, x_meth.shape[1]), nn.LeakyReLU(0.1)).to(device),
        'priv_mrna': nn.Sequential(nn.Linear(HID, x_mrna.shape[1]), nn.LeakyReLU(0.1)).to(device),
    }

    params = (list(enc_mir.parameters()) + list(enc_meth.parameters()) + list(enc_mrna.parameters()) +
              list(attn_fusion.parameters()) + sum([list(d.parameters()) for d in decoders.values()], []))
    optimizer = torch.optim.Adam(params, lr=LR)

    seed_everything(SEED)

    log = {k: [] for k in ['epoch', 'loss', 'recon', 'align', 'orth', 'preserve',
                             'a_mir', 'a_meth', 'a_mrna', 'var_mir', 'var_meth', 'var_mrna']}

    for epoch in range(1, MAX_EPOCH + 1):
        optimizer.zero_grad()

        s_mir, p_mir = enc_mir(x_mir)
        s_meth, p_meth = enc_meth(x_meth)
        s_mrna, p_mrna = enc_mrna(x_mrna)
        h_shared, alpha = attn_fusion(s_mir, s_meth, s_mrna)

        rec_mir = decoders['shared_mir'](h_shared) + decoders['priv_mir'](p_mir)
        rec_meth = decoders['shared_meth'](h_shared) + decoders['priv_meth'](p_meth)
        rec_mrna = decoders['shared_mrna'](h_shared) + decoders['priv_mrna'](p_mrna)

        loss_recon = (F.mse_loss(rec_mir, x_mir) + F.mse_loss(rec_meth, x_meth) + F.mse_loss(rec_mrna, x_mrna))
        loss_align = (F.mse_loss(s_mir, h_shared) + F.mse_loss(s_meth, h_shared) + F.mse_loss(s_mrna, h_shared))
        loss_orth = (cosine_orth_loss(s_mir, p_mir) + cosine_orth_loss(s_meth, p_meth) + cosine_orth_loss(s_mrna, p_mrna))
        loss_preserve = var_preservation(p_mir) + var_preservation(p_meth) + var_preservation(p_mrna)

        loss = loss_recon + LAMBDA_ALIGN * loss_align + LAMBDA_ORTH * loss_orth + LAMBDA_PRESERVE * loss_preserve

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                var_m = torch.var(p_mir).item()
                var_me = torch.var(p_meth).item()
                var_r = torch.var(p_mrna).item()

                log['epoch'].append(epoch)
                log['loss'].append(loss.item())
                log['recon'].append(loss_recon.item())
                log['align'].append(loss_align.item())
                log['orth'].append(loss_orth.item())
                log['preserve'].append(loss_preserve.item())
                log['a_mir'].append(alpha[0, 0].item())
                log['a_meth'].append(alpha[0, 1].item())
                log['a_mrna'].append(alpha[0, 2].item())
                log['var_mir'].append(var_m)
                log['var_meth'].append(var_me)
                log['var_mrna'].append(var_r)

                if epoch % 50 == 0:
                    print(f'  Epoch {epoch}: loss={loss.item():.4f}, recon={loss_recon.item():.4f}, '
                          f'orth={loss_orth.item():.4f}, preserve={loss_preserve.item():.4f}, '
                          f'var_p=[{var_m:.4f},{var_me:.4f},{var_r:.4f}]')

    enc_mir.eval(); enc_meth.eval(); enc_mrna.eval(); attn_fusion.eval()
    with torch.no_grad():
        s_mir, p_mir = enc_mir(x_mir)
        s_meth, p_meth = enc_meth(x_meth)
        s_mrna, p_mrna = enc_mrna(x_mrna)
        h_shared, _ = attn_fusion(s_mir, s_meth, s_mrna)

        pd.DataFrame(h_shared.cpu().numpy(), index=pat_id, columns=[f'shared_{i}' for i in range(HID)]).to_csv(DATA_OUT / 'h_shared_64d.csv')
        pd.DataFrame(p_mir.cpu().numpy(), index=pat_id, columns=[f'priv_mir_{i}' for i in range(HID)]).to_csv(DATA_OUT / 'p_mir_64d.csv')
        pd.DataFrame(p_meth.cpu().numpy(), index=pat_id, columns=[f'priv_meth_{i}' for i in range(HID)]).to_csv(DATA_OUT / 'p_meth_64d.csv')
        pd.DataFrame(p_mrna.cpu().numpy(), index=pat_id, columns=[f'priv_mrna_{i}' for i in range(HID)]).to_csv(DATA_OUT / 'p_mrna_64d.csv')

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    plot_items = [
        ('loss', 'Total Loss'), ('recon', 'Recon'), ('align', 'Align'), ('orth', 'Orth'),
        ('preserve', 'Preserve'), ('a_mir', 'Attn(mir)'), ('a_meth', 'Attn(meth)'), ('a_mrna', 'Attn(mrna)')
    ]
    for i, (k, t) in enumerate(plot_items):
        axes[i].plot(log['epoch'], log[k])
        axes[i].set_title(t)
        axes[i].set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'AE_training.png', dpi=300)
    plt.close()

    model_path = SAVE_DIR / 'ae_model.pth'
    torch.save({
        'enc_mir': enc_mir.state_dict(),
        'enc_meth': enc_meth.state_dict(),
        'enc_mrna': enc_mrna.state_dict(),
        'attn_fusion': attn_fusion.state_dict(),
        'dec_shared_mir': decoders['shared_mir'].state_dict(),
        'dec_shared_meth': decoders['shared_meth'].state_dict(),
        'dec_shared_mrna': decoders['shared_mrna'].state_dict(),
        'dec_priv_mir': decoders['priv_mir'].state_dict(),
        'dec_priv_meth': decoders['priv_meth'].state_dict(),
        'dec_priv_mrna': decoders['priv_mrna'].state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': MAX_EPOCH,
        'lambda_align': LAMBDA_ALIGN,
        'lambda_orth': LAMBDA_ORTH,
        'lambda_preserve': LAMBDA_PRESERVE,
    }, model_path)

    print(f'[{CANCER} result_AE] Completed. Model saved to {model_path}')
    print(f'  - Representations: {DATA_OUT}')

    del enc_mir, enc_meth, enc_mrna, attn_fusion, decoders, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
    for i, (k, t) in enumerate(plot_items):
        axes[i].plot(log['epoch'], log[k])
        axes[i].set_title(t)
        axes[i].set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'AE_training.png', dpi=300)
    plt.close()

    model_path = SAVE_DIR / 'ae_model.pth'
    torch.save({
        'enc_mir': enc_mir.state_dict(),
        'enc_meth': enc_meth.state_dict(),
        'enc_mrna': enc_mrna.state_dict(),
        'attn_fusion': attn_fusion.state_dict(),
        'dec_shared_mir': decoders['shared_mir'].state_dict(),
        'dec_shared_meth': decoders['shared_meth'].state_dict(),
        'dec_shared_mrna': decoders['shared_mrna'].state_dict(),
        'dec_priv_mir': decoders['priv_mir'].state_dict(),
        'dec_priv_meth': decoders['priv_meth'].state_dict(),
        'dec_priv_mrna': decoders['priv_mrna'].state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': MAX_EPOCH,
        'lambda_align': LAMBDA_ALIGN,
        'lambda_orth': LAMBDA_ORTH,
        'lambda_preserve': LAMBDA_PRESERVE,
    }, model_path)

    print(f'[{CANCER} result_AE] Completed. Model saved to {model_path}')
    print(f'  - Representations: {DATA_OUT}')

    del enc_mir, enc_meth, enc_mrna, attn_fusion, decoders, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()