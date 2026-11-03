import numpy as np

from src.variant import SEED, NOISE_SIGMA, SIZE, EPS
from src.image import clean_image, add_noise
from src.objective import make_objective
from src.oracle import Oracle
from src.lab1_svenn import svenn
from src.lab2_golden import golden_section
from src.lab2_dichotomy import dichotomy
from src.filters import gaussian_blur
from src.metrics import psnr, mse

clean = clean_image(SIZE)
_, noisy = add_noise(clean, NOISE_SIGMA, SEED)
f = make_objective(noisy, clean)
start_oracle = Oracle(f)
bracket = svenn(start_oracle, 0.2093, 0.2534)
a, b = bracket["a"], bracket["b"]

print("ДОСЛІД 1. ТОЧНІСТЬ epsilon = 1e-6")
for name, method in (("Дихотомія", dichotomy), ("Золотий переріз", golden_section)):
    oracle = Oracle(f)
    res = method(oracle, a, b, 1e-6)
    print(f"{name}: sigma*={res['x_star']:.9f}, n={res['iterations']}, NF={res['n_f']}, L={res['b']-res['a']:.9e}")

print("\nДОСЛІД 2. ДИХОТОМІЯ З delta = epsilon")
eps = EPS
delta = eps
left, right = a, b
oracle = Oracle(f)
for k in range(1, 101):
    x1 = (left + right - delta) / 2
    x2 = (left + right + delta) / 2
    if oracle.value(x1) <= oracle.value(x2):
        right = x2
    else:
        left = x1
    if right - left <= eps:
        break
print(f"Після {k} ітерацій: L = {right-left:.12e}; epsilon={eps:.12e}; завершено={right-left <= eps}")
print("При delta = epsilon звуження асимптотично наближається до epsilon; безпечний вибір delta < epsilon.")

print("\nДОСЛІД 3. РУЧНИЙ ПІДБІР sigma")
opt = golden_section(Oracle(f), a, b, EPS)["x_star"]
for sigma in (opt, 1.0):
    image = gaussian_blur(noisy, sigma)
    print(f"sigma={sigma:.6f}; MSE={mse(clean, image):.10f}; PSNR={psnr(clean, image):.4f} дБ")
