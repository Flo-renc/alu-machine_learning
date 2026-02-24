#!/usr/bin/env python3
"""
Module that calculates a GMM
"""
import sklearn.mixture


def gmm(X, k):
    """
    Calculates GMM clustering
    """
    if (not hasattr(X, "shape") or
            len(X.shape) != 2 or
            not isinstance(k, int) or k <= 0):
        return None, None, None, None, None

    try:
        model = sklearn.mixture.GaussianMixture(n_components=k)
        model.fit(X)

        pi = model.weights_
        m = model.means_
        S = model.covariances_
        clss = model.predict(X)
        bic = model.bic(X)

        return pi, m, S, clss, bic

    except Exception:
        return None, None, None, None, None