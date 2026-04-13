import os
import pathlib
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import shutil
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from multiprocessing import get_context
from collections import defaultdict
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.hgnn import HazardHGNN, CausalGate, MultiHeadAttnFusion
from config.paths import DATA_DIR, OUTPUT_DIR

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration
k_neighbor = 10
n_fold = 5
patience = 60
n_epochs = 3000
DROP_P = 0.3
L1_LAMBDA = 1e-4

# Save directory for AUC models
SAVE_DIR = OUTPUT_DIR / 'auc_model'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameter grid (729 combinations)
GAMMA_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BETA_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ALPHA_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MAX_WORKERS = 40

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class BCEConsistencyLoss(nn.Module):
    """
    Main BCE loss + ALPHA * consistency BCE loss across modalities
    """
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, logit_f, e, logit4=None):
        loss_main = self.ce(logit_f, e)
        
        if logit4 is None:
            return loss_main
        
        loss_cons = 0
        for c in range(4):
            loss_cons += self.ce(logit4[:, c], e)
        loss_cons = loss_cons / 4
        
        return loss_main + self.alpha * loss_cons

def ranking_distill_loss(haz_f, haz4, time, event):
    mask = event.bool()
    if mask.sum() < 2:
        return torch.tensor(0.0, device=haz_f.device)
    t_e, h_e = time[mask], haz_f[mask]
    p_teacher = torch.sigmoid(h_e.unsqueeze(1) - h_e.unsqueeze(0)).detach().view(-1)
    loss = 0
    for c in range(4):
        p_student = torch.sigmoid(haz4[:, c][mask].unsqueeze(1) - haz4[:, c][mask].unsqueeze(0)).view(-1)
        loss += F.binary_cross_entropy(p_student, p_teacher, reduction='mean')
    return loss / 4

def delta_auc(logit4, e):
    """
    Calculate permutation importance for AUC
    """
    if isinstance(e, torch.Tensor):
        e_np = e.cpu().numpy()
    else:
        e_np = e
    
    probs4 = F.softmax(logit4, dim=2)[:, :, 1].detach().cpu().numpy()
    base_prob = probs4.mean(axis=1)
    base_auc = roc_auc_score(e_np, base_prob)
    
    delta = []
    for c in range(4):
        perm = probs4.copy()
        np.random.shuffle(perm[:, c])
        auc_perm = roc_auc_score(e_np, perm.mean(axis=1))
        delta.append(base_auc - auc_perm)
    
    return np.array(delta)

def build_H(X, k=10):
    X_tensor = torch.as_tensor(X, device=device)
    n = X_tensor.size(0)
    if n == 0:
        return np.zeros((0, 0))
    cos = F.normalize(X_tensor, p=2, dim=1)
    sim = cos @ cos.t()
    _, idx = torch.topk(sim, k=min(k+1, n), dim=1)
    H = torch.zeros(n, n, device=device)
    src = torch.arange(n, device=device).view(-1, 1).expand_as(idx)
    H[src.reshape(-1), idx.reshape(-1)] = 1.
    H.fill_diagonal_(0)
    return H.cpu().numpy()

def train_single_fold(args):
    """
    Train specific fold with AUC as evaluation metric.
    Deep copy fix applied to preserve best model weights.
    """
    fold_idx, ALPHA, BETA, GAMMA, fold_data, save_path = args
    device_local = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(42 + fold_idx + int(ALPHA*10) + int(BETA*10) + int(GAMMA*10))
    
    def to_tensor(x, dtype=torch.float32):
        return torch.as_tensor(x, device=device_local, dtype=dtype)
    
    # Unpack data
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
    
    # Model initialization
    nets = nn.ModuleList([HazardHGNN(i).to(device_local) for i in range(4)])
    gate = CausalGate(init_weights=delta_star.tolist() if isinstance(delta_star, np.ndarray) else None).to(device_local)
    fuse = MultiHeadAttnFusion().to(device_local)
    
    optimizer = optim.Adam(
        list(nets.parameters()) + list(gate.parameters()) + list(fuse.parameters()),
        lr=3e-4, weight_decay=1e-4
    )
    
    criterion = BCEConsistencyLoss(alpha=ALPHA)
    
    best_auc = 0.0
    counter = 0
    best_state = None
    
    # Training loop
    for epoch in range(1, n_epochs + 1):
        for net in nets:
            net.train()
        gate.train()
        fuse.train()
        optimizer.zero_grad()
        
        inputs_tr = [x_s_tr, x_m_tr, x_p_tr, x_r_tr]
        
        # Forward pass
        haz_list = []
        logit_list = []
        for i in range(4):
            haz, logit, _ = nets[i](inputs_tr[i], H_tr[i])
            haz_list.append(haz)
            logit_list.append(logit)
        
        haz4 = torch.cat(haz_list, dim=1)
        logits4 = torch.stack(logit_list, dim=1)
        
        # Gating and fusion
        haz4_gated = gate(haz4)
        haz_f, attn = fuse(haz4_gated)
        
        # Fuse logits for classification
        logits_f = (attn.unsqueeze(-1) * logits4).sum(dim=1)
        
        # Loss computation
        loss_bce_total = criterion(logits_f, e_tr, logits4)
        loss_dis = ranking_distill_loss(haz_f, haz4_gated, t_tr, e_tr)
        
        # Auxiliary classification loss
        logits_aux = torch.stack([nets[i](inputs_tr[i], H_tr[i])[1] for i in range(4)], dim=1)
        logits_aux = (attn.unsqueeze(-1) * logits_aux).sum(dim=1)
        loss_cls = F.cross_entropy(logits_aux, e_tr)
        
        # L1 regularization
        l1_reg = 0
        for i in range(1, 4):
            for param in nets[i].parameters():
                l1_reg += torch.sum(torch.abs(param))
        
        # Total loss
        loss = loss_bce_total + BETA * loss_dis + GAMMA * loss_cls + L1_LAMBDA * l1_reg
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(nets.parameters()) + list(gate.parameters()) + list(fuse.parameters()), 
            max_norm=1.0
        )
        optimizer.step()
        
        # Validation every 10 epochs
        if epoch % 10 == 0:
            for net in nets:
                net.eval()
            gate.eval()
            fuse.eval()
            
            with torch.no_grad():
                inputs_val = [x_s_val, x_m_val, x_p_val, x_r_val]
                
                haz_list_val = []
                logit_list_val = []
                for i in range(4):
                    haz, logit, _ = nets[i](inputs_val[i], H_val[i])
                    haz_list_val.append(haz)
                    logit_list_val.append(logit)
                
                haz4_v = torch.cat(haz_list_val, dim=1)
                logits4_v = torch.stack(logit_list_val, dim=1)
                
                haz4_v = gate(haz4_v)
                haz_f_v, attn_v = fuse(haz4_v)
                logits_f_v = (attn_v.unsqueeze(-1) * logits4_v).sum(dim=1)
                
                # Calculate AUC
                probs = F.softmax(logits_f_v, dim=1)[:, 1]
                auc = roc_auc_score(e_val.cpu().numpy(), probs.cpu().numpy())
                
                if auc > best_auc:
                    best_auc = auc
                    counter = 0
                    # Deep copy fix: ensure exact best epoch weights are preserved
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
    
    # Save model if path provided
    if save_path is not None and best_state is not None:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_path)
    
    # Cleanup
    del nets, gate, fuse, optimizer
    torch.cuda.empty_cache()
    
    return {
        'fold': fold_idx,
        'alpha': ALPHA,
        'beta': BETA,
        'gamma': GAMMA,
        'auc': best_auc,
        'save_path': str(save_path) if save_path else None
    }

def main():
    # Load data
    h_shared = pd.read_csv(DATA_DIR / 'h_shared_64d.csv', index_col=0).astype('float32')
    p_mir = pd.read_csv(DATA_DIR / 'p_mir_64d.csv', index_col=0).astype('float32')
    p_meth = pd.read_csv(DATA_DIR / 'p_meth_64d.csv', index_col=0).astype('float32')
    p_rna = pd.read_csv(DATA_DIR / 'p_mrna_64d.csv', index_col=0).astype('float32')
    clinical = pd.read_csv(DATA_DIR / 'clinical_delete_process.csv', index_col=0)

    # Align indices
    common_idx = h_shared.index.intersection(p_mir.index).intersection(p_meth.index).intersection(p_rna.index).intersection(clinical.index)
    h_shared = h_shared.loc[common_idx]
    p_mir = p_mir.loc[common_idx]
    p_meth = p_meth.loc[common_idx]
    p_rna = p_rna.loc[common_idx]
    clinical = clinical.loc[common_idx]
    
    print(f'Aligned samples: {len(common_idx)}')
    
    X_s = h_shared.values.astype('float32')
    X_m = p_mir.values.astype('float32')
    X_p = p_meth.values.astype('float32')
    X_r = p_rna.values.astype('float32')
    
    t = clinical['os_time'].values.astype('float32')
    e = clinical['os_status'].map({'Dead': 1, 'Alive': 0}).values.astype('float32')
    n_total = len(t)
    
    print(f'[Global] n_total: {n_total}, events: {int(e.sum())}')
    print(f'[AUC Version] Main loss: BCE | Consistency: BCE | Distill: Ranking | Metric: AUC')
    print('='*60)

    # Pre-calculate all fold data
    print('Pre-computing 5-fold data splits (AUC version)...')
    outer_kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    all_folds_data = []
    
    for fold_idx, (tr_idx, val_idx) in enumerate(outer_kf.split(range(n_total))):
        print(f'  Preparing Fold {fold_idx}...')
        
        X_s_tr, X_m_tr, X_p_tr, X_r_tr = X_s[tr_idx], X_m[tr_idx], X_p[tr_idx], X_r[tr_idx]
        X_s_val, X_m_val, X_p_val, X_r_val = X_s[val_idx], X_m[val_idx], X_p[val_idx], X_r[val_idx]
        t_tr, e_tr = t[tr_idx], e[tr_idx]
        t_val, e_val = t[val_idx], e[val_idx]
        
        H_tr_full = [build_H(X_s_tr), build_H(X_m_tr), build_H(X_p_tr), build_H(X_r_tr)]
        H_val_full = [build_H(X_s_val), build_H(X_m_val), build_H(X_p_val), build_H(X_r_val)]
        
        # Calculate delta_star using inner 5-fold (AUC version)
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
            sub_e = torch.as_tensor(e[inner_tr_global], device=device, dtype=torch.long)
            
            nets_tmp = nn.ModuleList([HazardHGNN().to(device) for _ in range(4)])
            
            with torch.no_grad():
                logit_list = []
                for i in range(4):
                    _, logit, _ = nets_tmp[i](sub_x[i], torch.eye(len(inner_tr_global), device=device))
                    logit_list.append(logit)
                logits4_tmp = torch.stack(logit_list, dim=1)
            
            delta_buf.append(delta_auc(logits4_tmp, sub_e))
        
        delta_star = np.median(delta_buf, axis=0)
        print(f'    Fold {fold_idx} delta_star (AUC): {delta_star}')
        
        all_folds_data.append({
            'fold': fold_idx,
            'X_s_tr': X_s_tr, 'X_m_tr': X_m_tr, 'X_p_tr': X_p_tr, 'X_r_tr': X_r_tr,
            'X_s_val': X_s_val, 'X_m_val': X_m_val, 'X_p_val': X_p_val, 'X_r_val': X_r_val,
            't_tr': t_tr, 'e_tr': e_tr, 't_val': t_val, 'e_val': e_val,
            'H_tr': H_tr_full, 'H_val': H_val_full,
            'delta_star': delta_star
        })

    # Generate all tasks
    print('\nGenerating evaluation tasks...')
    all_tasks = []
    for fold_data in all_folds_data:
        for A in ALPHA_GRID:
            for B in BETA_GRID:
                for G in GAMMA_GRID:
                    all_tasks.append((fold_data['fold'], A, B, G, fold_data, None))
    
    print(f'Total tasks: {len(all_tasks)} (729 hyperparams x 5 folds)')
    
    # Parallel execution
    print('Starting parallel training (optimization target: AUC)...')
    ctx = get_context('spawn')
    all_results = []
    
    with ctx.Pool(processes=MAX_WORKERS) as pool:
        for result in tqdm(
            pool.imap_unordered(train_single_fold, all_tasks),
            total=len(all_tasks),
            desc='Global hyperparameter search (AUC)'
        ):
            all_results.append(result)
    
    # Aggregate results
    print('\nAggregating results, computing cross-fold mean AUC...')
    combo_results = defaultdict(lambda: {'aucs': [], 'folds': [], 'paths': []})
    
    for res in all_results:
        key = (res['alpha'], res['beta'], res['gamma'])
        combo_results[key]['aucs'].append(res['auc'])
        combo_results[key]['folds'].append(res['fold'])
        combo_results[key]['paths'].append(res['save_path'])
    
    combo_summary = []
    for (A, B, G), data in combo_results.items():
        mean_auc = np.mean(data['aucs'])
        std_auc = np.std(data['aucs'])
        combo_summary.append({
            'alpha': A, 'beta': B, 'gamma': G,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'fold_aucs': data['aucs'],
            'paths': data['paths']
        })
    
    combo_summary.sort(key=lambda x: x['mean_auc'], reverse=True)
    
    # Output Top 10
    print('\nTop 10 hyperparameter combinations (based on 5-fold mean AUC):')
    for i, item in enumerate(combo_summary[:10]):
        print(f'{i+1}. (A={item["alpha"]}, B={item["beta"]}, G={item["gamma"]}) '
              f'-> Mean AUC: {item["mean_auc"]:.4f} ± {item["std_auc"]:.4f}')
    
    # Select best hyperparameters
    best_combo = combo_summary[0]
    best_A, best_B, best_G = best_combo['alpha'], best_combo['beta'], best_combo['gamma']
    print(f'\nOptimal hyperparameters: A={best_A}, B={best_B}, G={best_G}')
    print(f'5-fold AUC: {np.round(best_combo["fold_aucs"], 4)}')
    print(f'Mean ± Std: {best_combo["mean_auc"]:.4f} ± {best_combo["std_auc"]:.4f}')
    
    # Save best models
    print('\nSaving best models with deep copy fix...')
    for fold_idx in range(n_fold):
        for res in all_results:
            if (res['fold'] == fold_idx and 
                res['alpha'] == best_A and 
                res['beta'] == best_B and 
                res['gamma'] == best_G):
                src_path = pathlib.Path(res['save_path'])
                dst_path = SAVE_DIR / f'best_model_fold{fold_idx}.pth'
                if src_path.exists():
                    shutil.copy(src_path, dst_path)
                    print(f'  Fold {fold_idx}: {dst_path} (AUC={res["auc"]:.4f})')
                break
    
    # Save summary
    final_summary = {
        'best_params': {'alpha': best_A, 'beta': best_B, 'gamma': best_G},
        'best_mean_auc': float(best_combo['mean_auc']),
        'best_std_auc': float(best_combo['std_auc']),
        'fold_aucs': [float(a) for a in best_combo['fold_aucs']],
        'all_combos': combo_summary[:50]
    }
    
    with open(SAVE_DIR / 'final_summary_auc.json', 'w') as f:
        json.dump(final_summary, f, indent=2)
    
    print(f'\nResults saved to: {SAVE_DIR}')
    print('AUC training completed. Main loss: BCE, other losses unchanged.')
    print('='*60)

if __name__ == '__main__':
    main()