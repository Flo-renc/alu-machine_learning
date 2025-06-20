#!/usr/bin/env python3
"""Poisson distribution class without imports"""


class Poisson:
    """Represents a Poisson distribution"""

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
            self.lambtha = float(sum(data) / len(data))

    def factorial(self, k):
        """Computes factorial of k"""
        result = 1
        for i in range(1, k + 1):
            result *= i
        return result

    def exp(self, x):
        """Approximates e^(-x) using Taylor series (up to 50 terms)"""
        result = 1
        term = 1
        for i in range(1, 51):
            term *= -x / i
            result += term
        return result

    def pmf(self, k):
        """Probability Mass Function"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        return (self.lambtha ** k * self.exp(-self.lambtha)) / self.factorial(k)

    def cdf(self, k):
        """Cumulative Distribution Function"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        cumulative = 0
        for i in range(k + 1):
            cumulative += (self.lambtha ** i * self.exp(-self.lambtha)) / self.factorial(i)
        return cumulative
