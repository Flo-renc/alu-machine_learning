#!/usr/bin/env python3
"""Module to calculate the mean and covariance of a dataset"""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a dataset
    :param X: numpy.ndarray of shape (n, d)
    :return: mean (1, d), covariance matrix (d, d)
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - mean
    cov = np.dot(X_centered.T, X_centered) / (n - 1)

    return mean, cov
