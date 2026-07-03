import json
import torch
from pathlib import Path


def resolve_paths(cfg, cancer):
    base = Path(cfg['results_dir']) / cancer
    return {
        'data_dir': Path(cfg['data_dir']) / cancer,
        'ci_data_dir': base / "result_CI" / "data",
        'ci_model_dir': base / "result_CI" / "model_hgnn_v2",
        'ci_ae_dir': base / "result_CI" / "models",
        'auc_data_dir': base / "result_AUC" / "data",
        'auc_model_dir': base / "result_AUC" / "model_auc_v2",
        'auc_ae_dir': base / "result_AUC" / "models",
    }


def save_checkpoint(state, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, device='cpu'):
    return torch.load(path, map_location=device)