#!/usr/bin/env python3
"""
Module to compute the minor matrix of a given matrix.
"""

def determinant(matrix):
    """
    Helper function to calculate the determinant of a matrix.
    """
    if matrix == [[]]:
        return 1
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    det = 0
    for col in range(len(matrix)):
        submatrix = [row[:col] + row[col+1:] for row in matrix[1:]]
        det += ((-1) ** col) * matrix[0][col] * determinant(submatrix)
    return det

def minor(matrix):
    """
    Calculates the minor matrix of a given square matrix.

    Args:
        matrix (list): A list of lists representing the matrix.

    Returns:
        list of lists: Minor matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.
    """
    if not isinstance(matrix, list) or any(not isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    
    if matrix == [] or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    
    if matrix == [[ ]]:
        return [[1]]
    
    size = len(matrix)
    if size == 1:
        return [[1]]

    minors = []
    for i in range(size):
        row_minors = []
        for j in range(size):
            # Create submatrix excluding row i and column j
            sub = [
                [matrix[m][n] for n in range(size) if n != j]
                for m in range(size) if m != i
            ]
            row_minors.append(determinant(sub))
        minors.append(row_minors)

    return minors
