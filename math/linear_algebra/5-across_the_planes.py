#!/usr/bin/env python3
"""Module that adds two 2D matrices element-wise."""


def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise.
    """
    if len(mat1) != len(mat2):
        return None
    if any(len(r1) != len(r2) for r1, r2 in zip(mat1, mat2)):
        return None
    return [[a + b for a, b in zip(row1, row2)]
    for row1, row2 in 
        zip(mat1, mat2)]
