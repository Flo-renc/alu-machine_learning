#!/usr/bin/env python3
"""
Cofactor matrix calculation
"""

def determinant(matrix):
    """Recursively calculates the determinant of a square matrix"""
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    
    n = len(matrix)

    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    det = 0
    for col in range(n):
        # Create minor matrix
        minor = [row[:col] + row[col+1:] for row in matrix[1:]]
        # Alternate signs for cofactor expansion
        det += ((-1) ** col) * matrix[0][col] * determinant(minor)
    return det


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a square matrix
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    cofactor_matrix = []
    for i in range(n):
        row_cofactors = []
        for j in range(n):
            # Create the minor matrix
            minor = [row[:j] + row[j+1:] for k, row in enumerate(matrix) if k != i]
            # Cofactor is determinant of minor times sign
            sign = (-1) ** (i + j)
            cof = sign * determinant(minor)
            row_cofactors.append(cof)
        cofactor_matrix.append(row_cofactors)
    return cofactor_matrix
