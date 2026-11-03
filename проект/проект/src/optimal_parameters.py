from functools import lru_cache
import numpy as np

from src.variant import SEED, NOISE_SIGMA, SIZE, LAMBDA0, EPS, SIMPLEX_EDGE
from src.image import clean_image, add_noise
from src.metrics import mse
from src.filters import gaussian_blur
from src.lab4_exact import tikhonov_exact
from src.lab1_svenn import svenn
from src.lab2_golden import golden_section
from src.lab3_nelder_mead import nelder_mead
from src.objective2 import make_objective2
from src.oracle import Oracle


@lru_cache(maxsize=1)
def optimal_lambda():
    clean = clean_image(SIZE)
    _, noisy = add_noise(clean, NOISE_SIGMA, SEED)
    oracle = Oracle(lambda lam: mse(tikhonov_exact(noisy, float(lam)), clean))
    bracket = svenn(oracle, LAMBDA0, 0.2, lo=1e-4, hi=50.0)
    result = golden_section(oracle, bracket["a"], bracket["b"], eps=1e-3)
    return float(result["x_star"])


@lru_cache(maxsize=1)
def optimal_gaussian_sigma():
    clean = clean_image(SIZE)
    _, noisy = add_noise(clean, NOISE_SIGMA, SEED)
    oracle = Oracle(lambda sigma: mse(clean, gaussian_blur(noisy, float(sigma))))
    result = golden_section(oracle, 0.05, 8.0, eps=EPS)
    return float(result["x_star"])


@lru_cache(maxsize=1)
def optimal_bilateral_theta():
    clean = clean_image(SIZE)
    _, noisy = add_noise(clean, NOISE_SIGMA, SEED)
    sigma = optimal_gaussian_sigma()
    theta0 = np.array([
        min(max(2.0 * sigma, 0.5), 4.0),
        min(max(0.15 + 0.1 * NOISE_SIGMA, 0.02), 0.9),
    ])
    oracle = Oracle(make_objective2(noisy, clean))
    result = nelder_mead(oracle, theta0, SIMPLEX_EDGE, tol=1e-7)
    return float(result["x"][0]), float(result["x"][1])
