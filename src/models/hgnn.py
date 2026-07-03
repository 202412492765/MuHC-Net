import torch
import torch.nn as nn
import torch.nn.functional as F


class HGNNconv(nn.Module):
    def __init__(self, in_dim=64, hid=64, p=0.3):
        super().__init__()
        self.W = nn.Linear(in_dim, hid)
        self.dropout = nn.Dropout(p)

    def forward(self, X, H):
        deg_v = 1. / (H.sum(1) + 1e-6).clamp(min=1e-6)
        deg_e = 1. / (H.sum(0) + 1e-6).clamp(min=1e-6)
        X_e = deg_e.view(-1, 1) * (H.t() @ X)
        X_v = (H @ X_e) * deg_v.view(-1, 1)
        return self.dropout(F.relu(self.W(X_v)))


class HazardHGNN(nn.Module):
    def __init__(self, modality_idx=0, use_cls=False, drop_p=0.3):
        super().__init__()
        self.modality_idx = modality_idx
        self.use_cls = use_cls
        self.conv = HGNNconv(64, 64, drop_p)
        self.head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(drop_p), nn.Linear(32, 1)
        )
        if use_cls:
            self.cls = nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(drop_p), nn.Linear(32, 2)
            )

    def forward(self, X, H):
        emb = self.conv(X, H)
        haz = self.head(emb)
        if self.use_cls:
            logit = self.cls(emb)
            return haz, logit, emb
        return haz, emb


class ModalityCausalGate(nn.Module):
    def __init__(self, ema_decay=0.9):
        super().__init__()
        self.w_ca = nn.Parameter(torch.zeros(4))
        self.register_buffer('effect_prior', torch.zeros(4))
        self.register_buffer('ema_effects', torch.ones(4) * 0.25)
        self.ema_decay = ema_decay

    def update_effects(self, new_effects):
        self.ema_effects = self.ema_decay * self.ema_effects + \
                          (1 - self.ema_decay) * new_effects.detach()
        self.effect_prior = self.ema_effects.clone()

    def forward(self, haz4):
        combined = self.w_ca + self.effect_prior
        w = F.softmax(combined, dim=0)
        return haz4 * w.view(1, 4), w


class MultiHeadAttnFusion(nn.Module):
    def __init__(self, n_heads=4, hid=64):
        super().__init__()
        self.n_heads = n_heads
        self.hid = hid
        self.q = nn.Sequential(
            nn.Linear(4 * hid, hid), nn.Tanh(), nn.Linear(hid, n_heads)
        )
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