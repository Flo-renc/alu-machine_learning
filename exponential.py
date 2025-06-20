#!/usr/bin/env python3
"""Exponential distribution class without imports"""


class Exponential:
    """Represents an exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Initialize the distribution"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(1 / (sum(data) / len(data)))

    def exp(self, x):
        """Approximates e^(-x) using Taylor series (up to 50 terms)"""
        result = 1
        term = 1
        for i in range(1, 51):
            term *= -x / i
            result += term
        return result

    def pdf(self, x):
        """Probability Density Function"""
        if x < 0:
            return 0
        return self.lambtha * self.exp(-self.lambtha * x)

    def cdf(self, x):
        """Cumulative Distribution Function"""
        if x < 0:
            return 0
        return 1 - self.exp(-self.lambtha * x)
