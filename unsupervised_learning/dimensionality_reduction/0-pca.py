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

        explained_variance = S ** 2
        total_variance = np.sum(explained_variance)
        cumulative_ratio = np.cumsum(explained_variance) / total_variance

        # Count components where cumulative variance is LESS than var,
        # then add 1 to include the one that pushes it over
        nd = int(np.sum(cumulative_ratio < var)) + 1

        W = Vt[:nd].T
        return W

    except Exception:
        return None
