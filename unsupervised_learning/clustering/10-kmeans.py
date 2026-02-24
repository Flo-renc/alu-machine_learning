#!/usr/bin/env python3
"""
Module that performs K-means clustering
"""
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means clustering
    """
    if (not hasattr(X, "shape") or
            len(X.shape) != 2 or
            not isinstance(k, int) or k <= 0):
        return None, None

    try:
        model = sklearn.cluster.KMeans(n_clusters=k)
        model.fit(X)

        C = model.cluster_centers_
        clss = model.labels_

        return C, clss

    except Exception:
        return None, None