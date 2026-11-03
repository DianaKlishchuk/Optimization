
import numpy as np

from src.variant import (
    SEED, NOISE_SIGMA, SIZE, SIMPLEX_EDGE, EPS
)
from src.image import clean_image, add_noise
from src.filters import gaussian_blur, bilateral_filter
from src.metrics import mse, psnr
from src.objective2 import make_objective2
from src.oracle import Oracle
from src.lab3_nelder_mead import nelder_mead
from src.lab3_powell import powell
from src.lab3_plots import plot_contours

from src.lab2_golden import golden_section

clean = clean_image(SIZE)
raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)

print("ПЕРЕВІРКА БІЛАТЕРАЛЬНОГО ФІЛЬТРА")

bilateral = bilateral_filter(noisy, 1.0, 100.0)
gaussian = gaussian_blur(noisy, 1.0)

print(f"Розмір: {bilateral.shape}")
print(f"Мінімум: {bilateral.min():.6f}")
print(f"Максимум: {bilateral.max():.6f}")
print(f"MSE білатерального: {mse(clean, bilateral):.8f}")
print(f"PSNR білатерального: {psnr(clean, bilateral):.4f} дБ")
print(f"MSE між фільтрами: {mse(bilateral, gaussian):.8f}")

gaussian_oracle = Oracle(
    lambda sigma: mse(
        clean,
        gaussian_blur(noisy, float(sigma))
    )
)

gaussian_result = golden_section(
    gaussian_oracle,
    0.05,
    8.0,
    eps=EPS
)

best_gaussian_sigma = gaussian_result["x_star"]

print("\nОПТИМІЗАЦІЯ ГАУСОВОГО ФІЛЬТРА")
print(f"Найкраще sigma = {best_gaussian_sigma:.6f}")
print(f"MSE = {gaussian_result['f_star']:.10f}")
print(f"Nf = {gaussian_oracle.n_f}")

theta0 = np.array([
    min(max(2.0 * best_gaussian_sigma, 0.5), 4.0),
    min(max(0.15 + 0.1 * NOISE_SIGMA, 0.02), 0.9)
])

f = make_objective2(noisy, clean)
oracle = Oracle(f)

print("\nЦІЛЬОВА ФУНКЦІЯ ДВОХ ЗМІННИХ")
print(f"Початкова точка: {theta0}")
print(f"f(theta0) = {oracle.value(theta0):.8f}")
print(f"Кількість звернень Nf = {oracle.n_f}")

nm_oracle = Oracle(f)

nm_result = nelder_mead(
    nm_oracle,
    theta0,
    SIMPLEX_EDGE,
    tol=1e-7
)

print("\nМЕТОД НЕЛДЕРА–МІДА")
print(
    f"{'k':>3} {'Операція':<22} "
    f"{'sigma_s':>10} {'sigma_r':>10} "
    f"{'f':>12} {'Розмір':>10} {'Nf':>5}"
)

for r in nm_result["rows"]:
    print(
        f"{r['k']:>3} "
        f"{r['op']:<22} "
        f"{r['x'][0]:>10.6f} "
        f"{r['x'][1]:>10.6f} "
        f"{r['f']:>12.8f} "
        f"{r['size']:>10.7f} "
        f"{r['n_f']:>5}"
    )

print("\nРезультат Нелдера–Міда:")
print(f"sigma_s* = {nm_result['x'][0]:.6f}")
print(f"sigma_r* = {nm_result['x'][1]:.6f}")
print(f"f* = {nm_result['f']:.10f}")
print(f"Ітерацій = {nm_result['iterations']}")
print(f"Nf = {nm_result['n_f']}")

powell_oracle = Oracle(f)

powell_result = powell(
    powell_oracle,
    theta0,
    tol=EPS
)

print("\nМЕТОД ПАУЕЛЛА")
print(
    f"{'Цикл':>5} {'sigma_s':>10} {'sigma_r':>10} "
    f"{'f':>12} {'Зсув':>12} "
    f"{'alpha1':>12} {'alpha2':>12} {'Nf':>6}"
)

for r in powell_result["rows"]:
    print(
        f"{r['cycle']:>5} "
        f"{r['x'][0]:>10.6f} "
        f"{r['x'][1]:>10.6f} "
        f"{r['f']:>12.8f} "
        f"{r['shift']:>12.8f} "
        f"{r['alpha1']:>12.6f} "
        f"{r['alpha2']:>12.6f} "
        f"{r['n_f']:>6}"
    )

print("\nРезультат Пауелла:")
print(f"sigma_s* = {powell_result['x'][0]:.6f}")
print(f"sigma_r* = {powell_result['x'][1]:.6f}")
print(f"f* = {powell_result['f']:.10f}")
print(f"Циклів = {powell_result['cycles']}")
print(f"Nf = {powell_result['n_f']}")

print("\nПОРІВНЯННЯ МЕТОДІВ")
print(f"{'Метод':<18} {'f*':>14} {'Nf':>8}")
print(
    f"{'Нелдера–Міда':<18} "
    f"{nm_result['f']:>14.10f} "
    f"{nm_result['n_f']:>8}"
)
print(
    f"{'Пауелла':<18} "
    f"{powell_result['f']:>14.10f} "
    f"{powell_result['n_f']:>8}"
)

if nm_result["f"] <= powell_result["f"]:
    best_theta = nm_result["x"]
    best_method = "Нелдера–Міда"
else:
    best_theta = powell_result["x"]
    best_method = "Пауелла"

restored = bilateral_filter(
    noisy,
    float(best_theta[0]),
    float(best_theta[1])
)

print("\nВІДНОВЛЕНЕ ЗОБРАЖЕННЯ")
print(f"Кращий метод: {best_method}")
print(f"MSE зашумленого = {mse(clean, noisy):.10f}")
print(f"MSE відновленого = {mse(clean, restored):.10f}")
print(f"PSNR зашумленого = {psnr(clean, noisy):.4f} дБ")
print(f"PSNR відновленого = {psnr(clean, restored):.4f} дБ")

import os
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

plt.imsave(
    "outputs/lab3_restored.png",
    restored,
    cmap="gray",
    vmin=0,
    vmax=1
)

print("Збережено: outputs/lab3_restored.png")

nm_points = [theta0.copy()] + [
    row["x"] for row in nm_result["rows"]
]

powell_points = [
    theta0.copy()
] + [
    row["x"] for row in powell_result["rows"]
]

plot_contours(
    f,
    nm_points,
    powell_points,
    best_theta
)

print("Збережено графік траєкторій оптимізації.")
