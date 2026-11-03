from src.optimal_parameters import optimal_lambda

import numpy as np

from src.variant import SEED, NOISE_SIGMA, SIZE
from src.image import clean_image, add_noise
from src.operators import energy_grad
from src.lab5_constrained import project, projected_gradient


clean = clean_image(SIZE)
raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)

lam = optimal_lambda()
L = 1 + 8 * lam
h = 1 / L
tol = 1e-6

correct = projected_gradient(
    raw, lam, L,
    accelerated=False,
    max_iter=200,
    tol=tol
)

I = project(raw.copy())

print("ДОСЛІД 1. НЕПРАВИЛЬНИЙ КРИТЕРІЙ ЗУПИНКИ")
print()
print(" k | ||grad||       | G")

converged_wrong = False

for k in range(201):
    g = energy_grad(I, raw, lam)

    grad_norm = float(np.linalg.norm(g))
    I_new = project(I - h * g)
    G = float(np.linalg.norm(I_new - I) / h)

    if k in [0, 1, 2, 5, 10, 20, 41, 50, 100, 150, 200]:
        print(f"{k:3d} | {grad_norm:.6e} | {G:.6e}")

    if grad_norm <= tol:
        converged_wrong = True
        break

    if k < 200:
        I = I_new

print()
print("Правильний критерій (градієнтне відображення):")
print(f'  Ітерацій = {correct["iterations"]}')
print(f'  G = {correct["G"]:.6e}')
print(f'  Збіжність = {"так" if correct["converged"] else "ні"}')

print()
print("Неправильний критерій (норма градієнта):")
print(f"  Ітерацій = {k}")
print(f"  ||grad|| = {grad_norm:.6e}")
print(f"  G = {G:.6e}")
print(f"  Збіжність = {'так' if converged_wrong else 'ні'}")
