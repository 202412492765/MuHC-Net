#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training utilities: seeding, causal inference, metrics.
"""

import numpy as np
import torch
from lifelines.utils import concordance_index


def seed_everything(seed=42):
    """Fix random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def delta_ci(haz4, t, e):
    """
    Compute permutation-based causal effect strength per branch.
    Returns array of 4 delta values.
    """
    base_ci = concordance_index(t, -haz4.mean(1), e)
    delta = []
    for c in range(4):
        perm = haz4.copy()
        np.random.shuffle(perm[:, c])
        ci_perm = concordance_index(t, -perm.mean(1), e)
        delta.append(base_ci - ci_perm)
    return np.array(delta)