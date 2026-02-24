#!/usr/bin/env python3
"""
Module that performs Expectation-Maximization for GMM
"""
import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000,
                             tol=1e-5, verbose=False):
    """
    Performs the EM algorithm
    """

    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2 or
            not isinstance(k, int) or k <= 0 or
            not isinstance(iterations, int) or iterations <= 0 or
            not isinstance(tol, float) or tol < 0 or
            not isinstance(verbose, bool)):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None:
        return None, None, None, None, None

    l_prev = 0

    # ONE LOOP
    for i in range(iterations):

        g, l = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None

        pi, m, S = maximization(X, g)

        if verbose and (i % 10 == 0 or i == iterations - 1):
            print("Log Likelihood after {} iterations: {:.5f}"
                  .format(i, l))

        if i > 0 and abs(l - l_prev) <= tol:
            break

        l_prev = l

    return pi, m, S, g, l