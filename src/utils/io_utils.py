#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I/O utilities for path management and model checkpointing.
"""

import json
from pathlib import Path


def ensure_dir(path):
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    """Save dictionary to JSON file."""
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)