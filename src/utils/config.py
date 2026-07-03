import yaml
from pathlib import Path


def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    base = Path(cfg.get('base_dir', '.')).resolve()
    cfg['base_dir'] = str(base)
    cfg['data_dir'] = str(base / cfg['data_dir'])
    cfg['results_dir'] = str(base / cfg['results_dir'])
    cfg['checkpoints_dir'] = str(base / cfg['checkpoints_dir'])
    return cfg


def merge_cli_args(cfg, args):
    if hasattr(args, 'cancer') and args.cancer:
        cfg['cancer'] = args.cancer
    if hasattr(args, 'device') and args.device:
        cfg['device'] = args.device
    if hasattr(args, 'seed') and args.seed is not None:
        cfg['seed'] = args.seed
    return cfg