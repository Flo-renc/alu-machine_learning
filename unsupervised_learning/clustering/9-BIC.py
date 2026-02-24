#!/usr/bin/env python3
"""
Module that determines best k using BIC
"""
import numpy as np

expectation_maximization = \
    __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000,
        tol=1e-5, verbose=False):
    """
    Finds best k using BIC
    """

    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2 or
            not isinstance(kmin, int) or kmin <= 0):
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    if (not isinstance(kmax, int) or
            kmax <= kmin):
        return None, None, None, None

    ks = range(kmin, kmax + 1)

    l_vals = []
    b_vals = []
    results = []

    # ONE LOOP
    for k in ks:

        pi, m, S, _, l = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        if pi is None:
            return None, None, None, None

        # number of parameters
        p = (k * d) + (k * d * (d + 1) / 2) + (k - 1)

        bic = p * np.log(n) - 2 * l

        l_vals.append(l)
        b_vals.append(bic)
        results.append((pi, m, S))

    l_vals = np.array(l_vals)
    b_vals = np.array(b_vals)

    best_index = np.argmin(b_vals)
    best_k = ks[best_index]
    best_result = results[best_index]

    return best_k, best_result, l_vals, b_vals