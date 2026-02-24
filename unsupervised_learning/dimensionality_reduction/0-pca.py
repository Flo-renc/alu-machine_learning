#!/usr/bin/env python3
"""
Module that performs PCA dimensionality reduction
"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on dataset X
    """
    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2 or
            not isinstance(ndim, int) or
            ndim <= 0 or
            ndim > X.shape[1]):
        return None
    # SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Keep first ndim components
    W = Vt[:ndim].T

    # Project data
    T = X @ W

    return T

    