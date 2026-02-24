def regular(P):
    """
    Determines steady state probabilities of a regular Markov chain
    """
    if (not isinstance(P, np.ndarray) or
            P.ndim != 2 or
            P.shape[0] != P.shape[1]):
        return None

    try:
        eigvals, eigvecs = np.linalg.eig(P.T)

        # Find eigenvalue closest to 1
        index = np.argmin(np.abs(eigvals - 1))
        steady = np.real(eigvecs[:, index])

        steady = steady / np.sum(steady)
        steady = steady.reshape(1, -1)

        if np.all(steady >= 0):
            return steady
        return None
    except Exception:
        return None