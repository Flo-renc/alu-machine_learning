#!/usr/bin/env python3
"""Poisson distribution class"""
import math


class Poisson:
    """Class that represents a Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Initialize the Poisson distribution"""
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

    def pmf(self, k):
        """Probability Mass Function for Poisson distribution"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        # PMF formula: P(k) = (λ^k * e^-λ) / k!
        lambtha = self.lambtha
        return (lambtha ** k * math.exp(-lambtha)) / math.factorial(k)

    def cdf(self, k):
        """Cumulative Distribution Function for Poisson distribution"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        # CDF = sum of PMFs from 0 to k
        lambtha = self.lambtha
        cdf = 0
        for i in range(k + 1):
            cdf += (lambtha ** i * math.exp(-lambtha)) / math.factorial(i)
        return cdf
