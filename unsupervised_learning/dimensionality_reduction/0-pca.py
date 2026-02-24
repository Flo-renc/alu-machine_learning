#!/usr/bin/env python3
"""
Module that performs PCA
"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on dataset X
    """
    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2 or
            not isinstance(var, float) or
            var <= 0 or var > 1):
        return None

    try:
        # SVD decomposition
        U, S, Vt = np.linalg.svd(X, full_matrices=False)

        # Variance explained
        explained_variance = S ** 2
        total_variance = np.sum(explained_variance)

        cumulative_variance = (
            np.cumsum(explained_variance) / total_variance
        )

        # Find minimum dimensions needed
        nd = np.searchsorted(cumulative_variance, var) + 1

        # Weight matrix
        W = Vt[:nd].T

        return W

    except Exception:
        return None