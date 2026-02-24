#!/usr/bin/env python3
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely hidden state sequence
    """
    if (not isinstance(Observation, np.ndarray) or
            Observation.ndim != 1):
        return None, None

    if (not isinstance(Emission, np.ndarray) or
            Emission.ndim != 2):
        return None, None

    if (not isinstance(Transition, np.ndarray) or
            Transition.ndim != 2):
        return None, None

    if (not isinstance(Initial, np.ndarray) or
            Initial.shape[1] != 1):
        return None, None

    try:
        T = Observation.shape[0]
        N = Emission.shape[0]

        delta = np.zeros((N, T))
        psi = np.zeros((N, T), dtype=int)

        # Initialization
        delta[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

        # Recursion
        for t in range(1, T):
            for j in range(N):
                prob = delta[:, t - 1] * Transition[:, j]
                psi[j, t] = np.argmax(prob)
                delta[j, t] = np.max(prob) * Emission[j, Observation[t]]

        # Backtracking
        path = np.zeros(T, dtype=int)
        path[T - 1] = np.argmax(delta[:, T - 1])

        for t in reversed(range(T - 1)):
            path[t] = psi[path[t + 1], t + 1]

        P = np.max(delta[:, T - 1])

        return path.tolist(), P

    except Exception:
        return None, None