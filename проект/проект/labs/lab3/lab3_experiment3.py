
import numpy as np

from src.variant import SEED, NOISE_SIGMA, SIZE, EPS
from src.image import clean_image, add_noise
from src.filters import gaussian_blur
from src.metrics import mse
from src.objective2 import make_objective2
from src.oracle import Oracle
from src.lab2_golden import golden_section
from src.lab3_powell import powell


clean = clean_image(SIZE)
raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)

gaussian_oracle = Oracle(
    lambda sigma: mse(clean, gaussian_blur(noisy, float(sigma)))
)

gaussian_result = golden_section(
    gaussian_oracle, 0.05, 8.0, eps=EPS
)

best_gaussian_sigma = gaussian_result["x_star"]

theta0 = np.array([
    min(max(2 * best_gaussian_sigma, 0.5), 4.0),
    min(max(0.15 + 0.1 * NOISE_SIGMA, 0.02), 0.9)
])

f = make_objective2(noisy, clean)

print("ДОСЛІД 3. КРИТЕРІЙ ЗУПИНКИ ПАУЕЛЛА")
print(f"Початкова точка: {theta0}")
print(f"Точність лінійного пошуку: {EPS}")
print()

for tol in [1e-4, 1e-7]:
    oracle = Oracle(f)

    result = powell(
        oracle,
        theta0,
        tol=tol,
        max_cycles=50
    )

    print(f"Критерій зупинки tol = {tol}")
    print(f"  sigma_s* = {result['x'][0]:.6f}")
    print(f"  sigma_r* = {result['x'][1]:.6f}")
    print(f"  f* = {result['f']:.10f}")
    print(f"  Циклів = {result['cycles']}")
    print(f"  Nf = {result['n_f']}")
    print()
