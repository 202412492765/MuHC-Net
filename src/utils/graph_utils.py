import torch
import torch.nn.functional as F
import numpy as np


def build_H(X, k=10, device='cpu'):
    X_tensor = torch.as_tensor(X, device=device)
    n = X_tensor.size(0)
    if n == 0:
        return np.zeros((0, 0))
    cos = F.normalize(X_tensor, p=2, dim=1)
    sim = cos @ cos.t()
    k_actual = min(k + 1, n)
    _, idx = torch.topk(sim, k=k_actual, dim=1)
    H = torch.zeros(n, n, device=device)
    src = torch.arange(n, device=device).view(-1, 1).expand_as(idx)
    H[src.reshape(-1), idx.reshape(-1)] = 1.
    H.fill_diagonal_(0)
    return H.cpu().numpy()