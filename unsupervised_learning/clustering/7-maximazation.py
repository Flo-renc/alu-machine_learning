#!/usr/bin/env python3
"""
Module that calculates the maximization step in EM
"""
import numpy as np


def maximization(X, g):
    """
    Performs maximization step in EM

    Returns:
    pi, m, S
    """

    if (not isinstance(X, np.ndarray) or
            not isinstance(g, np.ndarray)):
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n != n_g:
        return None, None, None

    sum_g = np.sum(g, axis=1)

    if np.any(sum_g == 0):
        return None, None, None

    # Update priors
    pi = sum_g / n

    # Update means
    m = (g @ X) / sum_g[:, np.newaxis]

    S = np.zeros((k, d, d))

    # 1 LOOP
    for i in range(k):
        diff = X - m[i]
        weighted = g[i][:, np.newaxis] * diff
        S[i] = (weighted.T @ diff) / sum_g[i]

    return pi, m, S