
import numpy as np

from src.variant import SEED, NOISE_SIGMA, SIZE, SIMPLEX_EDGE
from src.image import clean_image, add_noise
from src.filters import gaussian_blur
from src.metrics import mse
from src.objective2 import make_objective2
from src.oracle import Oracle
from src.lab3_nelder_mead import nelder_mead
from src.lab2_golden import golden_section


clean = clean_image(SIZE)
raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)

gaussian_oracle = Oracle(
    lambda sigma: mse(clean, gaussian_blur(noisy, float(sigma)))
)

gaussian_result = golden_section(
    gaussian_oracle, 0.05, 8.0, eps=1e-4
)

best_gaussian_sigma = gaussian_result["x_star"]

theta0 = np.array([
    min(max(2.0 * best_gaussian_sigma, 0.5), 4.0),
    min(max(0.15 + 0.1 * NOISE_SIGMA, 0.02), 0.9)
])

f = make_objective2(noisy, clean)

edges = [
    ("Менший у 3 рази", SIMPLEX_EDGE / 3),
    ("Початковий", SIMPLEX_EDGE),
    ("Більший у 3 рази", SIMPLEX_EDGE * 3)
]

print("ДОСЛІД 1. РОЗМІР ПОЧАТКОВОГО СИМПЛЕКСА")
print(f"Початкова точка: {theta0}")
print()

for name, edge in edges:
    oracle = Oracle(f)

    result = nelder_mead(
        oracle,
        theta0,
        edge,
        tol=1e-7
    )

    print(f"{name}:")
    print(f"  Ребро = {edge:.6f}")
    print(f"  sigma_s* = {result['x'][0]:.6f}")
    print(f"  sigma_r* = {result['x'][1]:.6f}")
    print(f"  f* = {result['f']:.10f}")
    print(f"  Ітерацій = {result['iterations']}")
    print(f"  Nf = {result['n_f']}")
    print()
