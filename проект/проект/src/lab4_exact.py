
import numpy as np
from scipy.fft import dctn, idctn


def tikhonov_exact(noisy, lam):
    h, w = noisy.shape

    i = np.arange(h)[:, None]
    j = np.arange(w)[None, :]

    mu = (
        2 - 2 * np.cos(np.pi * i / h)
        + 2 - 2 * np.cos(np.pi * j / w)
    )

    spectrum = dctn(noisy, type=2, norm="ortho")

    return idctn(
        spectrum / (1 + lam * mu),
        type=2,
        norm="ortho"
    )
