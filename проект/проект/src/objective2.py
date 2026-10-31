
import numpy as np

from src.filters import bilateral_filter
from src.metrics import mse

LO = np.array([0.3, 0.01])
HI = np.array([5.0, 1.0])


def make_objective2(noisy, clean):
    def f(theta):
        point = np.asarray(theta, dtype=float)

        clamped = np.clip(point, LO, HI)

        restored = bilateral_filter(
            noisy,
            clamped[0],
            clamped[1]
        )

        value = mse(restored, clean)

        gap = point - clamped
        penalty = 1000.0 * float(gap @ gap)

        return value + penalty

    return f
