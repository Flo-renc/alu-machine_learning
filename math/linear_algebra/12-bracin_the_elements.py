#!/usr/bin/env python3
"""
Performs element-wise operations on matrices using NumPy
"""

import numpy as np

def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication, and division.

    Returns:
        A tuple of (add, subtract, multiply, divide) results.
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)

