
from src.filters import gaussian_blur
from src.metrics import mse

LO = 0.05
HI = 8.0

def make_objective(noisy, clean):
    def f(sigma):
        clamped = min(max(sigma, LO), HI)

        value = mse(
            gaussian_blur(noisy, clamped),
            clean
        )

        penalty = 1000.0 * (sigma - clamped) ** 2

        return value + penalty

    return f
