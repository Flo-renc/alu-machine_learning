#!/usr/bin/env python3
"""
Module that determines the optimum number of clusters
"""
import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance

    Parameters:
    X (numpy.ndarray): dataset of shape (n, d)
    kmin (int): minimum number of clusters
    kmax (int): maximum number of clusters
    iterations (int): maximum K-means iterations

    Returns:
    results (list): K-means results for each k
    d_vars (list): variance differences from kmin
    or None, None on failure
    """

    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2 or
            not isinstance(kmin, int) or
            kmin <= 0 or
            not isinstance(iterations, int) or
            iterations <= 0):
        return None, None

    n = X.shape[0]

    if kmax is None:
        kmax = n

    if (not isinstance(kmax, int) or
            kmax <= kmin or
            kmax > n):
        return None, None

    results = []
    variances = []

    # LOOP 1
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)

        if C is None:
            return None, None

        results.append((C, clss))
        variances.append(variance(X, C))

    base_variance = variances[0]

    d_vars = [base_variance - v for v in variances]

    return results, d_vars