#!/usr/bin/env python3
"""
This module provides a function that computes the derivative
of a polynomial represented as a list of coefficients.
"""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial.

    Args:
        poly (list): A list of coefficients. The index is the power of x.

    Returns:
        list: A list of coefficients representing the 
        derivative, or [0] if constant,
        or None if input is invalid.
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not all(isinstance(c, (int, float)) for c in poly):
        return None
    if len(poly) == 1:
        return [0]
    return [i * poly[i] for i in range(1, len(poly))]
