#!/usr/bin/env python3
"""
PCA that retains enough components to preserve a fraction of variance
"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on dataset X

    Parameters:
    X (numpy.ndarray): centered data (n, d)
    var (float): fraction of variance to retain

    Returns:
    W (numpy.ndarray): weight matrix (d, nd)
    """

    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2 or
            not isinstance(var, float) or
            var <= 0 or var > 1):
        return None

    try:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)

        # Use S directly (not S²) to compute variance ratio
        cumulative_ratio = np.cumsum(S) / np.sum(S)

        # Count components needed
        nd = int(np.sum(cumulative_ratio < var)) + 1

        W = Vt[:nd].T
        return W

    except Exception:
        return None
