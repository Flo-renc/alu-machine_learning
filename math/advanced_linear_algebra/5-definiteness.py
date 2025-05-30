#!/usr/bin/env python3
"""
Determines the definiteness of a matrix using its eigenvalues.
"""
import numpy as np

def definiteness(matrix):
    """
    Classifies a matrix as positive definite, semi-definite, negative definite, etc.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    eigenvalues = np.linalg.eigvalsh(matrix)  # more stable for symmetric matrices

    if np.all(eigenvalues > 0):
        return "Positive definite"
    if np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    if np.all(eigenvalues < 0):
        return "Negative definite"
    if np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    return "Indefinite"
