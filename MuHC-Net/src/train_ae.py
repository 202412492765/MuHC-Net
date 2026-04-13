import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.ae import Encoder, AttnFusion, GRL
from config.paths import DATA_DIR, OUTPUT_DIR

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load omics data
x_mir = torch.tensor(pd.read_csv(DATA_DIR / 'miRNA_preprocess.csv', index_col=0).values, 
                     dtype=torch.float32).to(device)
x_meth = torch.tensor(pd.read_csv(DATA_DIR / 'meth_preprocess.csv', index_col=0).values, 
                      dtype=torch.float32).to(device)
x_mrna = torch.tensor(pd.read_csv(DATA_DIR / 'mRNA_preprocess.csv', index_col=0).values, 
                      dtype=torch.float32).to(device)

print(f'Loaded -> miRNA: {x_mir.shape} (expected N×200); meth: {x_meth.shape}; mRNA: {x_mrna.shape}')
print(f'Device: {device}')

# Initialize encoders
enc_mir = Encoder(x_mir.shape[1], hid=64).to(device)
enc_meth = Encoder(x_meth.shape[1], hid=64).to(device)
enc_mrna = Encoder(x_mrna.shape[1], hid=64).to(device)
attn_fusion = AttnFusion(hid=64).to(device)

# Decoders with LeakyReLU to avoid dying neurons
dec_shared_mir = nn.Sequential(nn.Linear(64, x_mir.shape[1]), nn.LeakyReLU(0.1)).to(device)
dec_shared_meth = nn.Sequential(nn.Linear(64, x_meth.shape[1]), nn.LeakyReLU(0.1)).to(device)
dec_shared_mrna = nn.Sequential(nn.Linear(64, x_mrna.shape[1]), nn.LeakyReLU(0.1)).to(device)
dec_priv_mir = nn.Sequential(nn.Linear(64, x_mir.shape[1]), nn.LeakyReLU(0.1)).to(device)
dec_priv_meth = nn.Sequential(nn.Linear(64, x_meth.shape[1]), nn.LeakyReLU(0.1)).to(device)
dec_priv_mrna = nn.Sequential(nn.Linear(64, x_mrna.shape[1]), nn.LeakyReLU(0.1)).to(device)

# Cross-modal predictors
grl = GRL(lambda_grl=1.0)
pred_mir2meth = nn.Linear(64, x_meth.shape[1]).to(device)
pred_mir2mrna = nn.Linear(64, x_mrna.shape[1]).to(device)
pred_meth2mir = nn.Linear(64, x_mir.shape[1]).to(device)
pred_meth2mrna = nn.Linear(64, x_mrna.shape[1]).to(device)
pred_mrna2mir = nn.Linear(64, x_mir.shape[1]).to(device)
pred_mrna2meth = nn.Linear(64, x_meth.shape[1]).to(device)

# Collect parameters
params = (list(enc_mir.parameters()) + list(enc_meth.parameters()) + list(enc_mrna.parameters()) +
          list(attn_fusion.parameters()) +
          list(dec_shared_mir.parameters()) + list(dec_shared_meth.parameters()) + list(dec_shared_mrna.parameters()) +
          list(dec_priv_mir.parameters()) + list(dec_priv_meth.parameters()) + list(dec_priv_mrna.parameters()) +
          list(pred_mir2meth.parameters()) + list(pred_mir2mrna.parameters()) +
          list(pred_meth2mir.parameters()) + list(pred_meth2mrna.parameters()) +
          list(pred_mrna2mir.parameters()) + list(pred_mrna2meth.parameters()))

optimizer = torch.optim.Adam(params, lr=1e-3)

# Cosine orthogonality loss
def cosine_orth_loss(s, p, eps=1e-8):
    p_norm = torch.norm(p, p=2, dim=1, keepdim=True)
    if torch.mean(p_norm) < eps:
        return torch.tensor(1.0, device=p.device)
    s_norm = F.normalize(s, p=2, dim=1, eps=eps)
    p_norm = F.normalize(p, p=2, dim=1, eps=eps)
    cos_sim = torch.sum(s_norm * p_norm, dim=1)
    return torch.mean(torch.abs(cos_sim))

# Variance preservation loss
def var_preservation(p, min_var=0.001):
    var = torch.var(p, dim=0).mean()
    return F.relu(min_var - var) * 10.0

# Training configuration
max_epoch = 200
lambda_max = 2.0

# Logging dictionary
log = {k: [] for k in ['epoch', 'loss', 'recon', 'align', 'priv', 
                       'orth', 'preserve', 'lgrl', 'a_mir', 'a_meth', 
                       'a_mrna', 'var_mir', 'var_meth', 'var_mrna']}

print('Start training (LeakyReLU fix + Cosine Orthogonality)...')

# Training loop
for epoch in range(1, max_epoch + 1):
    optimizer.zero_grad()

    s_mir, p_mir = enc_mir(x_mir)
    s_meth, p_meth = enc_meth(x_meth)
    s_mrna, p_mrna = enc_mrna(x_mrna)
    h_shared, alpha = attn_fusion(s_mir, s_meth, s_mrna)

    # Decode
    rec_mir = dec_shared_mir(h_shared) + dec_priv_mir(p_mir)
    rec_meth = dec_shared_meth(h_shared) + dec_priv_meth(p_meth)
    rec_mrna = dec_shared_mrna(h_shared) + dec_priv_mrna(p_mrna)

    loss_recon = (F.mse_loss(rec_mir, x_mir) + F.mse_loss(rec_meth, x_meth) +
                  F.mse_loss(rec_mrna, x_mrna))
    loss_align = (F.mse_loss(s_mir, h_shared) + F.mse_loss(s_meth, h_shared) +
                  F.mse_loss(s_mrna, h_shared))

    # GRL scheduling
    lambda_grl = 0.1 + (lambda_max - 0.1) * (epoch - 1) / (max_epoch - 1)
    grl.lambda_grl = lambda_grl

    p_mir_grl = grl(p_mir)
    p_meth_grl = grl(p_meth)
    p_mrna_grl = grl(p_mrna)
    
    # Cross-modal prediction loss (adversarial)
    loss_priv = (F.mse_loss(pred_mir2meth(p_mir_grl), x_meth) +
                 F.mse_loss(pred_mir2mrna(p_mir_grl), x_mrna) +
                 F.mse_loss(pred_meth2mir(p_meth_grl), x_mir) +
                 F.mse_loss(pred_meth2mrna(p_meth_grl), x_mrna) +
                 F.mse_loss(pred_mrna2mir(p_mrna_grl), x_mir) +
                 F.mse_loss(pred_mrna2meth(p_mrna_grl), x_meth))

    # Orthogonality loss
    orth_mir = cosine_orth_loss(s_mir, p_mir)
    orth_meth = cosine_orth_loss(s_meth, p_meth)
    orth_mrna = cosine_orth_loss(s_mrna, p_mrna)
    loss_orth = orth_mir + orth_meth + orth_mrna
    
    # Information preservation loss
    loss_preserve = var_preservation(p_mir) + var_preservation(p_meth) + var_preservation(p_mrna)
    
    # Total loss
    loss = loss_recon + 0.5*loss_align - lambda_grl * loss_priv + 0.8*loss_orth + 0.3*loss_preserve

    if torch.isnan(loss):
        print(f"Warning: NaN loss at epoch {epoch}")
        continue
        
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    optimizer.step()

    # Logging every 10 epochs
    if epoch % 10 == 0 or epoch == 1:
        with torch.no_grad():
            var_m = torch.var(p_mir).item()
            var_me = torch.var(p_meth).item()
            var_r = torch.var(p_mrna).item()
            
            log['epoch'].append(epoch)
            log['loss'].append(loss.item())
            log['recon'].append(loss_recon.item())
            log['align'].append(loss_align.item())
            log['priv'].append(loss_priv.item())
            log['orth'].append(loss_orth.item())
            log['preserve'].append(loss_preserve.item())
            log['lgrl'].append(lambda_grl)
            log['a_mir'].append(alpha[0,0].item())
            log['a_meth'].append(alpha[0,1].item())
            log['a_mrna'].append(alpha[0,2].item())
            log['var_mir'].append(var_m)
            log['var_meth'].append(var_me)
            log['var_mrna'].append(var_r)

        print(f'Epoch{epoch:03d} lambda={lambda_grl:.3f} loss={loss.item():.3f} '
              f'var_p=[{var_m:.4f},{var_me:.4f},{var_r:.4f}] '
              f'orth={loss_orth.item():.3f}')

print('Training completed.')

# Plot training curves
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, (k, t) in enumerate(zip(['loss','recon','align','priv','orth','lgrl'], 
                                ['Loss','Recon','Align','Priv','Orth','GRL'])):
    axes[i].plot(log['epoch'], log[k])
    axes[i].set_title(t)

# Variance monitoring
axes[6].plot(log['epoch'], log['var_mir'], label='mir', color='orange')
axes[6].plot(log['epoch'], log['var_meth'], label='meth', color='green')
axes[6].plot(log['epoch'], log['var_mrna'], label='mrna', color='blue')
axes[6].axhline(y=0.001, color='r', linestyle='--', label='Min Target')
axes[6].set_title('Private Variance')
axes[6].legend()
axes[6].set_ylim(0, max(max(log['var_meth']), 0.01))

axes[7].stackplot(log['epoch'], log['a_mir'], log['a_meth'], log['a_mrna'], alpha=0.8)
axes[7].set_title('Attention Weights')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'ae_training_curves.png', dpi=300)
plt.show()

# Save model
torch.save({
    'enc_mir': enc_mir.state_dict(),
    'enc_meth': enc_meth.state_dict(),
    'enc_mrna': enc_mrna.state_dict(),
    'attn_fusion': attn_fusion.state_dict(),
}, OUTPUT_DIR / 'ae_model.pth')

print(f'Model saved to {OUTPUT_DIR}/ae_model.pth')