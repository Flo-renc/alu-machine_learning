#!/usr/bin/env python3
"""Module that concatenates two 2D matrices along a specific axis."""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along the specified axis.
    """
    if axis == 0:
        if (len(mat1[0])
                != len(mat2[0])):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    elif axis == 1:
        # Check that number of rows match
        if len(mat1) != len(mat2):
            return None
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
    else:
        return None
