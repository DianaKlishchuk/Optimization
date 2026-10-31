
import numpy as np

from src.operators import grad_xy, divergence


def tv_energy(I, noisy, lam, eps=0.01):
    gx, gy = grad_xy(I)

    smooth = np.sqrt(gx * gx + gy * gy + eps * eps)

    return (
        0.5 * float(np.sum((I - noisy) ** 2))
        + lam * float(np.sum(smooth))
    )


def tv_gradient(I, noisy, lam, eps=0.01):
    gx, gy = grad_xy(I)

    denominator = np.sqrt(
        gx * gx + gy * gy + eps * eps
    )

    px = gx / denominator
    py = gy / denominator

    return (I - noisy) - lam * divergence(px, py)


def tv_nesterov(noisy, lam, eps=0.01,
                max_iter=3000, tol=1e-6):

    L = 1 + 8 * lam / eps
    h = 1 / L

    I = noisy.copy()
    I_prev = I.copy()

    for k in range(1, max_iter + 1):
        beta = (k - 1) / (k + 2)

        Y = I + beta * (I - I_prev)
        g = tv_gradient(Y, noisy, lam, eps)

        I_new = Y - h * g

        if np.linalg.norm(g) <= tol:
            return I_new, k

        I_prev, I = I, I_new

    return I, max_iter
