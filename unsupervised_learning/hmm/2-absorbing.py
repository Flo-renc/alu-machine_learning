def absorbing(P):
    """
    Determines if a Markov chain is absorbing
    """
    if (not isinstance(P, np.ndarray) or
            P.ndim != 2 or
            P.shape[0] != P.shape[1]):
        return False

    n = P.shape[0]

    # Identify absorbing states
    absorbing_states = np.isclose(np.diag(P), 1)

    if not np.any(absorbing_states):
        return False

    try:
        # Check reachability
        P_power = np.linalg.matrix_power(P, n * n)
        reachable = P_power[:, absorbing_states]

        return np.all(np.sum(reachable, axis=1) > 0)
    except Exception:
        return False