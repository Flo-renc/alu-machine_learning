#!/usr/bin/env python3
"""
Module to compute the determinant of a matrix.
"""

def determinant(matrix):
    """
    Calculates the determinant of a square matrix.
    
    Args:
        matrix (list): A list of lists representing the matrix.
        
    Returns:
        int or float: Determinant of the matrix.
    
    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a square matrix.
    """
    if not isinstance(matrix, list) or any(not isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    n = len(matrix)

    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if n == 0:
        return 1
    if n == 1:
        return matrix[0][0]
    if n == 2:
        # base case for 2x2 matrix
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    # Recursive case for larger matrices
    det = 0
    for col in range(n):
        # build submatrix for minor
        minor = [row[:col] + row[col+1:] for row in matrix[1:]]
        cofactor = ((-1) ** col) * matrix[0][col] * determinant(minor)
        det += cofactor
    return det
