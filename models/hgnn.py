import torch
import torch.nn as nn
import torch.nn.functional as F

class HGNNconv(nn.Module):
    def __init__(self, in_dim=64, hid=64, p=0.3):
        super().__init__()
        self.W = nn.Linear(in_dim, hid)
        self.dropout = nn.Dropout(p)
        self.edge_weight = nn.Parameter(torch.ones(1))
    
    def forward(self, X, H):
        deg_v = 1. / (H.sum(1) + 1e-6).clamp(min=1e-6)
        deg_e = 1. / (H.sum(0) + 1e-6).clamp(min=1e-6)
        H_weighted = H * torch.sigmoid(self.edge_weight)
        X_e = deg_e.view(-1, 1) * (H_weighted.t() @ X)
        X_v = (H_weighted @ X_e) * deg_v.view(-1, 1)
        return self.dropout(F.relu(self.W(X_v)))

class HazardHGNN(nn.Module):
    def __init__(self, modality_idx=0, drop_p=0.3):
        super().__init__()
        self.modality_idx = modality_idx
        self.conv = HGNNconv(64, 64, p=drop_p)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(drop_p), nn.Linear(32, 1))
        self.cls = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(drop_p), nn.Linear(32, 2))
    
    def forward(self, X, H):
        emb = self.conv(X, H)
        haz = self.head(emb)
        logit = self.cls(emb)
        return haz, logit, emb

class CausalGate(nn.Module):
    def __init__(self, init_weights=None):
        super().__init__()
        if init_weights is None:
            init_weights = [1.0, 0.3, 0.3, 0.3]
        self.w_ca = nn.Parameter(torch.tensor(init_weights))
    
    def forward(self, haz4):
        return haz4 * self.w_ca.view(1, 4)

class MultiHeadAttnFusion(nn.Module):
    def __init__(self, n_heads=4, hid=64):
        super().__init__()
        self.n_heads = n_heads
        self.q = nn.Sequential(nn.Linear(4, hid), nn.Tanh(), nn.Linear(hid, n_heads))
        self.out_proj = nn.Linear(n_heads, 1, bias=False)
    
    def forward(self, haz4):
        logits = self.q(haz4)
        attn = torch.softmax(logits, dim=1)
        haz_expand = haz4.unsqueeze(1).expand(-1, self.n_heads, -1)
        weighted = (attn.unsqueeze(-1) * haz_expand).sum(dim=2)
        return self.out_proj(weighted), attn