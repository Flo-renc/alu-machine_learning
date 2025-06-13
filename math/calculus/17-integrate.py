#!/usr/bin/env python3
"""
This module provides a function that calculates the integral
of a polynomial with an optional integration constant.
"""


def poly_integral(poly, C=0):
    """
    Calculates the indefinite integral of a polynomial.

    Args:
        poly (list): List of coefficients (index = power of x).
        C (int): Integration constant (default is 0).

    Returns:
        list: A list of coefficients representing the integral,
              or None if input is invalid.
    """
    if not isinstance(poly, list) or not isinstance(C, int):
        return None
    if not all(isinstance(c, (int, float)) for c in poly):
        return None

    integral = [C]
    for i in range(len(poly)):
        coeff = poly[i] / (i + 1)
        integral.append(int(coeff) if coeff.is_integer() else coeff)
    return integral
