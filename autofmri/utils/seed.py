"""Reproducibility utilities."""

import random

import numpy as np


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across numpy and (optionally) torch.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    np.random.seed(seed)
    random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def get_tsnr(X: np.ndarray) -> float:
    """Compute the mean temporal signal-to-noise ratio across voxels.

    Parameters
    ----------
    X : ndarray, shape (n_timepoints, ...)
        fMRI time-series data.

    Returns
    -------
    float
        Mean tSNR (NaN/Inf voxels excluded).
    """
    tsnr = np.mean(X, axis=0) / np.std(X, axis=0)
    tsnr = tsnr[~np.isnan(tsnr)]
    tsnr = tsnr[~np.isinf(tsnr)]
    return float(np.mean(tsnr))
