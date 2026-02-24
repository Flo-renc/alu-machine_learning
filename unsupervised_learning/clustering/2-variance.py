#!/usr/bin/env python3
"""
Module that calculates total intra-cluster variance
"""
import numpy as np


def variance(X, C):
    """
    Calculates total intra-cluster variance for a dataset

    Parameters:
    X (numpy.ndarray): dataset of shape (n, d)
    C (numpy.ndarray): centroids of shape (k, d)

    Returns:
    float: total variance
    or None on failure
    """
    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2 or
            not isinstance(C, np.ndarray) or
            len(C.shape) != 2 or
            X.shape[1] != C.shape[1]):
        return None

    # Compute squared distances
    distances = np.sum(
        (X[:, np.newaxis, :] - C) ** 2,
        axis=2
    )

    # Get minimum distance per point
    min_distances = np.min(distances, axis=1)

    # Total variance
    return np.sum(min_distances)