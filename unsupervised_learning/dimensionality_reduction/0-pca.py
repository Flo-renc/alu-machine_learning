#!/usr/bin/env python3
"""
PCA that keeps enough components to preserve a given variance
"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on dataset X

    Parameters:
    X (numpy.ndarray): centered data of shape (n, d)
    var (float): fraction of variance to retain

    Returns:
    W (numpy.ndarray): weight matrix of shape (d, nd)
    """

    # Validation
    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2 or
            not isinstance(var, float) or
            var <= 0 or
            var > 1):
        return None

    try:
        # SVD
        U, S, Vt = np.linalg.svd(X, full_matrices=False)

        # Compute variance ratios
        explained_variance = S ** 2
        total_variance = np.sum(explained_variance)

        cumulative_variance = (
            np.cumsum(explained_variance) / total_variance
        )

        # Find minimum number of components
        nd = np.searchsorted(cumulative_variance, var) + 1

        # Build weight matrix
        W = Vt[:nd].T

        return W

    except Exception:
        return None