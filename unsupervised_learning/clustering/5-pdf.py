#!/usr/bin/env python3
"""
Module that calculates the PDF of a multivariate Gaussian distribution
"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution

    Parameters:
    X (numpy.ndarray): shape (n, d)
    m (numpy.ndarray): shape (d,)
    S (numpy.ndarray): shape (d, d)

    Returns:
    numpy.ndarray: shape (n,)
    or None on failure
    """

    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2 or
            not isinstance(m, np.ndarray) or
            len(m.shape) != 1 or
            not isinstance(S, np.ndarray) or
            len(S.shape) != 2 or
            X.shape[1] != m.shape[0] or
            S.shape[0] != S.shape[1] or
            S.shape[0] != m.shape[0]):
        return None

    n, d = X.shape

    try:
        inv_S = np.linalg.inv(S)
        det_S = np.linalg.det(S)
    except Exception:
        return None

    if det_S <= 0:
        return None

    diff = X - m

    exp_term = np.sum((diff @ inv_S) * diff, axis=1)

    norm_const = np.sqrt(((2 * np.pi) ** d) * det_S)

    P = (1 / norm_const) * np.exp(-0.5 * exp_term)

    P = np.maximum(P, 1e-300)

    return P