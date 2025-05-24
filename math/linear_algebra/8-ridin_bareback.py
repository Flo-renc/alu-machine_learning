#!/usr/bin/env python3
"""
Module that defines mat_mul to perform matrix multiplication.
"""


def mat_mul(mat1, mat2):
    """
    Multiplies two 2D matrices and returns the result as a new matrix.

    Args:
        mat1 (list of lists): First matrix.
        mat2 (list of lists): Second matrix.

    Returns:
        list of lists: Result of matrix multiplication or None if not possible.
    """
    if len(mat1[0]) != len(mat2):
        return None
    return [[sum(mat1[i][k] * mat2[k][j] for k in range(len(mat2)))
             for j in range(len(mat2[0]))] for i in range(len(mat1))]
