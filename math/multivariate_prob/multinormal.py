#!/usr/bin/env python3
"""Defines the MultiNormal class representing a Multivariate Normal distribution"""
import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution
    """

    def __init__(self, data):
        """
        Initializes a MultiNormal object

        Parameters:
        - data (numpy.ndarray): shape (d, n) with n data points in d dimensions

        Raises:
        - TypeError: if data is not a 2D numpy.ndarray
        - ValueError: if n < 2
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        X_centered = data - self.mean
        self.cov = np.dot(X_centered, X_centered.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a data point x

        Parameters:
        - x (numpy.ndarray): shape (d, 1) data point

        Returns:
        - PDF value as a float

        Raises:
        - TypeError: if x is not a numpy.ndarray
        - ValueError: if x is not of shape (d, 1)
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d, _ = self.mean.shape
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        det = np.linalg.det(self.cov)
        if det == 0:
            raise ValueError("Covariance matrix can't be singular")

        inv = np.linalg.inv(self.cov)
        diff = x - self.mean

        exponent = -0.5 * np.dot(np.dot(diff.T, inv), diff)
        numerator = np.exp(exponent)
        denominator = np.sqrt(((2 * np.pi) ** d) * det)

        return float(numerator / denominator)
