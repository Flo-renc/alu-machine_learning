#!/usr/bin/env python3
"""
Module that calculates the expectation step in EM
"""
import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Performs expectation step in EM

    Returns:
    g, l
    """

    if (not isinstance(X, np.ndarray) or
            not isinstance(pi, np.ndarray) or
            not isinstance(m, np.ndarray) or
            not isinstance(S, np.ndarray)):
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if (m.shape != (k, d) or
            S.shape != (k, d, d)):
        return None, None

    g = np.zeros((k, n))

    # 1 LOOP
    for i in range(k):
        P = pdf(X, m[i], S[i])
        if P is None:
            return None, None
        g[i] = pi[i] * P

    sum_g = np.sum(g, axis=0)

    if np.any(sum_g == 0):
        return None, None

    g /= sum_g

    l = np.sum(np.log(sum_g))

    return g, l