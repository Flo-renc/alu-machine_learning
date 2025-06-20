#!/usr/bin/env python3
"""Normal distribution class without imports"""


class Normal:
    """Represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize the distribution"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = variance ** 0.5

    def erf(self, x):
        """Approximates error function erf(x) using Maclaurin series"""
        # erf(x) ≈ 2/sqrt(pi) * (x - x^3/3 + x^5/10 - x^7/42 + x^9/216 ...)
        pi = 3.141592653589793
        term = x
        total = x
        for i in range(1, 10):
            term *= -x * x / i
            total += term / (2 * i + 1)
        return (2 / (pi ** 0.5)) * total

    def z_score(self, x):
        """Converts x to z-score"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Converts z-score to x"""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Probability Density Function"""
        pi = 3.141592653589793
        e_term = -0.5 * ((x - self.mean) / self.stddev) ** 2
        return (1 / (self.stddev * (2 * pi) ** 0.5)) * self.exp(e_term)

    def cdf(self, x):
        """Cumulative Distribution Function"""
        z = (x - self.mean) / (self.stddev * 2 ** 0.5)
        return 0.5 * (1 + self.erf(z))

    def exp(self, x):
        """Approximates e^x using Taylor series"""
        result = 1
        term = 1
        for i in range(1, 50):
            term *= x / i
            result += term
        return result
