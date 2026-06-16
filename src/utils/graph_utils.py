#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hypergraph construction utilities.
"""

import torch
import torch.nn.functional as F


def build_H(X, k=10, device='cpu'):
    """
    Construct hypergraph adjacency matrix via cosine similarity top-k.
    Returns numpy array for compatibility with data loading.
    """
    X_tensor = torch.as_tensor(X, device=device, dtype=torch.float32)
    n = X_tensor.size(0)
    if n == 0:
        return torch.zeros((0, 0), device=device)
    cos = F.normalize(X_tensor, p=2, dim=1)
    sim = cos @ cos.t()
    k_actual = min(k + 1, n)
    _, idx = torch.topk(sim, k=k_actual, dim=1)
    H = torch.zeros(n, n, device=device)
    src = torch.arange(n, device=device).view(-1, 1).expand_as(idx)
    H[src.reshape(-1), idx.reshape(-1)] = 1.
    H.fill_diagonal_(0)
    return H.cpu().numpy()