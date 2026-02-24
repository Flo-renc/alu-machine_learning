#!/usr/bin/env python3

def backward(Observation, Emission, Transition, Initial):
    """
    Performs backward algorithm
    """
    if (not isinstance(Observation, np.ndarray) or
            Observation.ndim != 1):
        return None, None

    try:
        T = Observation.shape[0]
        N = Emission.shape[0]

        B = np.zeros((N, T))

        # Initialization
        B[:, T - 1] = 1

        # Backward recursion
        for t in reversed(range(T - 1)):
            B[:, t] = np.sum(
                Transition *
                Emission[:, Observation[t + 1]] *
                B[:, t + 1],
                axis=1
            )

        P = np.sum(
            Initial[:, 0] *
            Emission[:, Observation[0]] *
            B[:, 0]
        )

        return P, B

    except Exception:
        return None, None