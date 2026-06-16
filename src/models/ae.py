#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autoencoder components for multi-omics shared-private representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Enc(nn.Module):
    """Dual-branched encoder producing shared and private representations."""
    def __init__(self, in_dim, hid=64):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(in_dim, hid), nn.LeakyReLU(0.1))
        self.priv = nn.Sequential(nn.Linear(in_dim, hid), nn.LeakyReLU(0.1))

    def forward(self, x):
        return self.shared(x), self.priv(x)


class AttnFusion(nn.Module):
    """Attention-based modality fusion gate for shared representations."""
    def __init__(self, hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * hid, 64), nn.Tanh(), nn.Linear(64, 3)
        )

    def forward(self, s_mir, s_meth, s_mrna):
        s_cat = torch.cat([s_mir, s_meth, s_mrna], dim=1)
        alpha = torch.softmax(self.net(s_cat), dim=1)
        h = (alpha[:, 0:1] * s_mir +
             alpha[:, 1:2] * s_meth +
             alpha[:, 2:3] * s_mrna)
        return h, alpha


def cosine_orth_loss(s, p, eps=1e-8):
    """
    Orthogonality loss enforcing geometric decoupling between shared and private.
    Returns mean absolute cosine similarity.
    """
    p_norm = torch.norm(p, p=2, dim=1, keepdim=True)
    if torch.mean(p_norm) < eps:
        return torch.tensor(1.0, device=p.device)
    s_norm = F.normalize(s, p=2, dim=1, eps=eps)
    p_norm = F.normalize(p, p=2, dim=1, eps=eps)
    cos_sim = torch.sum(s_norm * p_norm, dim=1)
    return torch.mean(torch.abs(cos_sim))


def var_preservation(p, min_var=0.001, beta=10.0):
    """
    Information preservation loss preventing private representation collapse.
    Enforces per-dimension variance above threshold.
    """
    var = torch.var(p, dim=0).mean()
    return F.relu(min_var - var) * beta