#!/usr/bin/env python3
"""Module to calculate a correlation matrix"""
import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix from a covariance matrix
    :param C: numpy.ndarray of shape (d, d)
    :return: correlation matrix of shape (d, d)
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    stddev = np.sqrt(np.diag(C))
    denom = np.outer(stddev, stddev)
    correlation_matrix = C / denom

    return correlation_matrix
