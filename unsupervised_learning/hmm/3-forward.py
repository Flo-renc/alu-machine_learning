def forward(Observation, Emission, Transition, Initial):
    """
    Performs forward algorithm for HMM
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
            Initial.ndim != 2):
        return None, None

    T = Observation.shape[0]
    N = Emission.shape[0]

    if (Transition.shape != (N, N) or
            Initial.shape != (N, 1)):
        return None, None

    try:
        F = np.zeros((N, T))

        # Initialization
        F[:, 0] = (Initial[:, 0] *
                   Emission[:, Observation[0]])

        # Forward recursion
        for t in range(1, T):
            F[:, t] = (
                np.matmul(F[:, t - 1], Transition)
                * Emission[:, Observation[t]]
            )

        P = np.sum(F[:, T - 1])

        return P, F
    except Exception:
        return None, None