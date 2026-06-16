#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load trained AE model and extract representations without training.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import torch
import yaml

from src.models.ae import Enc, AttnFusion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_config():
    cfg_path = Path(__file__).resolve().parents[2] / 'configs' / 'default.yaml'
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    CANCER = cfg['project']['cancer']
    ROOT = Path(cfg['paths']['root']).resolve()
    MODEL_PATH = ROOT / cfg['paths']['results'] / 'result_AE' / 'ae_model.pth'
    DATA_DIR = ROOT / cfg['paths']['data']
    OUT_DIR = ROOT / cfg['paths']['results'] / 'result_AE' / 'data'
    HID = cfg['hyperparams']['ae']['hid_dim']

    print('=' * 70)
    print(f'{CANCER} Autoencoder Inference')
    print('=' * 70)
    print(f'[{CANCER} result_AE] Loading AE model from {MODEL_PATH}')

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Model not found: {MODEL_PATH}')

    x_mir = torch.tensor(pd.read_csv(DATA_DIR / cfg['files']['miRNA'], index_col=0).values, dtype=torch.float32).to(device)
    x_meth = torch.tensor(pd.read_csv(DATA_DIR / cfg['files']['meth'], index_col=0).values, dtype=torch.float32).to(device)
    x_mrna = torch.tensor(pd.read_csv(DATA_DIR / cfg['files']['mRNA'], index_col=0).values, dtype=torch.float32).to(device)
    pat_id = pd.read_csv(DATA_DIR / cfg['files']['miRNA'], index_col=0).index

    n_samples = len(pat_id)
    print(f'[{CANCER} result_AE] Loaded {n_samples} samples')

    enc_mir = Enc(x_mir.shape[1], HID).to(device)
    enc_meth = Enc(x_meth.shape[1], HID).to(device)
    enc_mrna = Enc(x_mrna.shape[1], HID).to(device)
    attn_fusion = AttnFusion(HID).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    enc_mir.load_state_dict(checkpoint['enc_mir'])
    enc_meth.load_state_dict(checkpoint['enc_meth'])
    enc_mrna.load_state_dict(checkpoint['enc_mrna'])
    attn_fusion.load_state_dict(checkpoint['attn_fusion'])

    enc_mir.eval(); enc_meth.eval(); enc_mrna.eval(); attn_fusion.eval()

    print(f'[{CANCER} result_AE] Extracting representations...')
    with torch.no_grad():
        s_mir, p_mir = enc_mir(x_mir)
        s_meth, p_meth = enc_meth(x_meth)
        s_mrna, p_mrna = enc_mrna(x_mrna)
        h_shared, _ = attn_fusion(s_mir, s_meth, s_mrna)

        print(f'[{CANCER} result_AE] Shared representation shape: {tuple(h_shared.shape)}')
        print(f'[{CANCER} result_AE] Private miRNA representation shape: {tuple(p_mir.shape)}')
        print(f'[{CANCER} result_AE] Private methylation representation shape: {tuple(p_meth.shape)}')
        print(f'[{CANCER} result_AE] Private mRNA representation shape: {tuple(p_mrna.shape)}')

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(h_shared.cpu().numpy(), index=pat_id, columns=[f'shared_{i}' for i in range(HID)]).to_csv(OUT_DIR / 'h_shared_64d.csv')
        pd.DataFrame(p_mir.cpu().numpy(), index=pat_id, columns=[f'priv_mir_{i}' for i in range(HID)]).to_csv(OUT_DIR / 'p_mir_64d.csv')
        pd.DataFrame(p_meth.cpu().numpy(), index=pat_id, columns=[f'priv_meth_{i}' for i in range(HID)]).to_csv(OUT_DIR / 'p_meth_64d.csv')
        pd.DataFrame(p_mrna.cpu().numpy(), index=pat_id, columns=[f'priv_mrna_{i}' for i in range(HID)]).to_csv(OUT_DIR / 'p_mrna_64d.csv')

    print(f'[{CANCER} result_AE] Saved representations to {OUT_DIR}/')
    print('  - h_shared_64d.csv')
    print('  - p_mir_64d.csv')
    print('  - p_meth_64d.csv')
    print('  - p_mrna_64d.csv')


if __name__ == '__main__':
    main()