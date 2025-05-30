#!/usr/bin/env python3
"""
Calculates the adjugate matrix of a given square matrix.
"""

def determinant(matrix):
    """Recursive function to calculate the determinant of a matrix"""
    if matrix == [[]]:
        return 1
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    det = 0
    for col in range(len(matrix)):
        sub = [row[:col] + row[col+1:] for row in matrix[1:]]
        det += ((-1) ** col) * matrix[0][col] * determinant(sub)
    return det

def minor(matrix):
    """Returns the minor matrix"""
    if not isinstance(matrix, list) or any(not isinstance(r, list) for r in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == [] or any(len(r) != len(matrix) for r in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1:
        return [[1]]
    
    minors = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix)):
            sub = [row[:j] + row[j+1:] for k, row in enumerate(matrix) if k != i]
            row.append(determinant(sub))
        minors.append(row)
    return minors

def adjugate(matrix):
    """Returns the adjugate matrix"""
    if not isinstance(matrix, list) or any(not isinstance(r, list) for r in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == [] or any(len(r) != len(matrix) for r in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    
    minors = minor(matrix)
    size = len(matrix)

    # Cofactor = alternate sign * minor
    for i in range(size):
        for j in range(size):
            minors[i][j] *= (-1) ** (i + j)
    
    # Transpose to get adjugate
    adj = [[minors[j][i] for j in range(size)] for i in range(size)]
    return adj
