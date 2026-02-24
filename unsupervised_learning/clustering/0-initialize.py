#!/usr/bin/env python3
"""Module that initializes cluster centroids for k-means"""

import numpy as np

def initialize(X, k):
    """
    initialize cluster centriods for k-means
    
    :param X (numpy.nd): dataset of shape (n, d)
    :param k (int): number of clusters

    Returns:
    numpy.ndarray: initialized centroids of shape (k, d)
    or None on failure
    """
    
    if (not isinstance(X, np.array) or
            len(X.shape) != 2 or
            not isinstance(k, int) or
            k <= 0 or
            k > X.shape[0]):
        return None
    
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    centroids = np.random.uniform(
        low=min_vals,
        high=max_vals,
        size=(k, X.shape[1])
    )

    return centroids