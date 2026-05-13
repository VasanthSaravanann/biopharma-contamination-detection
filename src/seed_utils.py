"""Stabilize random seeds for reproducible experiments across all libraries.

Ensures that all RNG sources (numpy, torch, sklearn, etc.) are seeded
consistently to guarantee reproducible results in CI and local runs.
"""
import numpy as np
import random
import os


def set_random_seed(seed: int = 42):
    """Set seed across all RNG sources for reproducibility.
    
    Args:
        seed: Seed value (default 42)
    """
    # Python built-in
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # OS-level (for certain C++ libraries)
    os.environ['PYTHONHASHSEED'] = str(seed)
