#!/usr/bin/env python3
"""Module that calculates the shape of a matrix"""


def matrix_shape(matrix):
    """
    Returns the shape of a matrix as a list of integers.
    Assumes all elements at each level have the same shape.

    Parameters:
        matrix (list): A nested list (matrix) of arbitrary depth

    Returns:
        list: Dimensions of the matrix
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) == 0:
            break
        matrix = matrix[0]
    return shape