#!/usr/bin/env python3

def baum_welch(Observations, Transition, Emission,
               Initial, iterations=1000):
    """
    Performs Baum-Welch algorithm
    """
    if (not isinstance(Observations, np.ndarray) or
            Observations.ndim != 1):
        return None, None

    try:
        T = Observations.shape[0]
        M, N = Emission.shape

        for _ in range(iterations):

            # Forward
            P, F = forward(
                Observations, Emission,
                Transition, Initial
            )

            # Backward
            _, B = backward(
                Observations, Emission,
                Transition, Initial
            )

            gamma = (F * B) / P

            xi = np.zeros((M, M, T - 1))

            for t in range(T - 1):
                numerator = (
                    F[:, t][:, None] *
                    Transition *
                    Emission[:, Observations[t + 1]] *
                    B[:, t + 1]
                )
                xi[:, :, t] = numerator / np.sum(numerator)

            # Update Transition
            Transition = np.sum(xi, axis=2) / \
                np.sum(gamma[:, :-1], axis=1)[:, None]

            # Update Emission
            for k in range(N):
                mask = (Observations == k)
                Emission[:, k] = np.sum(
                    gamma[:, mask], axis=1
                )

            Emission = Emission / \
                np.sum(gamma, axis=1)[:, None]

        return Transition, Emission

    except Exception:
        return None, None