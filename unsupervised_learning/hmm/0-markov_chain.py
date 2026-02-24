#!/usr/bin/env python3
import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines probability of being in each state after t iterations
    """
    if (not isinstance(P, np.ndarray) or
            P.ndim != 2 or
            P.shape[0] != P.shape[1]):
        return None

    n = P.shape[0]

    if (not isinstance(s, np.ndarray) or
            s.shape != (1, n) or
            not isinstance(t, int) or
            t < 0):
        return None

    try:
        Pt = np.linalg.matrix_power(P, t)
        return np.matmul(s, Pt)
    except Exception:
        return None