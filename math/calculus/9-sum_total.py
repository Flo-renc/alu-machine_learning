#!/usr/bin/env python3
"""
This module provides a function to compute the summation
of i squared from 1 to n using a mathematical formula.
"""


def summation_i_squared(n):
    """
    Calculate the sum of squares from 1 to n (inclusive).
    Args:
        n (int): The stopping condition.
    Returns:
        int: The result of the summation, or None if n is invalid.
    """
    if not isinstance(n, int) or n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
