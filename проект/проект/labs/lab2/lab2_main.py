
from src.variant import SEED, NOISE_SIGMA, EPS, SIZE
from src.image import clean_image, add_noise
from src.objective import make_objective
from src.oracle import Oracle
from src.lab2_dichotomy import dichotomy

clean = clean_image(SIZE)
raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)

f = make_objective(noisy, clean)
oracle = Oracle(f)

a0 = 0.2093
b0 = 0.9695

result = dichotomy(oracle, a0, b0, EPS)

print("МЕТОД ДИХОТОМІЇ")
print(f"{'k':>3} {'x1':>10} {'x2':>10} {'f1':>12} {'f2':>12} {'L':>10}")

for r in result["rows"]:
    print(
        f"{r['k']:>3} "
        f"{r['x1']:>10.6f} "
        f"{r['x2']:>10.6f} "
        f"{r['f1']:>12.8f} "
        f"{r['f2']:>12.8f} "
        f"{r['L']:>10.7f}"
    )

print("\nРезультат:")
print(f"σ* = {result['x_star']:.6f}")
print(f"f(σ*) = {result['f_star']:.8f}")
print(f"Ітерацій = {result['iterations']}")
print(f"Nf = {result['n_f']}")
print(f"Кінцевий відрізок = [{result['a']:.6f}; {result['b']:.6f}]")

from src.lab2_golden import golden_section

golden_oracle = Oracle(f)

golden_result = golden_section(
    golden_oracle, a0, b0, EPS
)

print("\nМЕТОД ЗОЛОТОГО ПЕРЕРІЗУ")

print(
    f"{'k':>3} {'x1':>10} {'x2':>10} "
    f"{'f1':>12} {'f2':>12} {'L':>10}"
)

for r in golden_result["rows"]:
    print(
        f"{r['k']:>3} "
        f"{r['x1']:>10.6f} "
        f"{r['x2']:>10.6f} "
        f"{r['f1']:>12.8f} "
        f"{r['f2']:>12.8f} "
        f"{r['L']:>10.7f}"
    )

print("\nРезультат золотого перерізу:")
print(f"σ* = {golden_result['x_star']:.6f}")
print(f"f(σ*) = {golden_result['f_star']:.8f}")
print(f"Ітерацій = {golden_result['iterations']}")
print(f"Nf = {golden_result['n_f']}")

print("\nПОРІВНЯННЯ МЕТОДІВ")
print(f"{'Метод':<20} {'σ*':>10} {'Ітерації':>10} {'Nf':>8}")

print(
    f"{'Дихотомія':<20} "
    f"{result['x_star']:>10.6f} "
    f"{result['iterations']:>10} "
    f"{result['n_f']:>8}"
)

print(
    f"{'Золотий переріз':<20} "
    f"{golden_result['x_star']:>10.6f} "
    f"{golden_result['iterations']:>10} "
    f"{golden_result['n_f']:>8}"
)


import os
import numpy as np
from PIL import Image

from src.filters import gaussian_blur
from src.metrics import mse, psnr

sigma_star = golden_result["x_star"]

restored = gaussian_blur(noisy, sigma_star)

mse_noisy = mse(clean, noisy)
psnr_noisy = psnr(clean, noisy)

mse_restored = mse(clean, restored)
psnr_restored = psnr(clean, restored)

print("\nЯКІСТЬ ВІДНОВЛЕННЯ")
print(f"Оптимальне σ* = {sigma_star:.6f}")
print(f"MSE до відновлення = {mse_noisy:.8f}")
print(f"MSE після відновлення = {mse_restored:.8f}")
print(f"PSNR до відновлення = {psnr_noisy:.4f} дБ")
print(f"PSNR після відновлення = {psnr_restored:.4f} дБ")
print(f"Приріст PSNR = {psnr_restored - psnr_noisy:.4f} дБ")

os.makedirs("outputs", exist_ok=True)

restored_uint8 = (
    np.clip(restored, 0, 1) * 255
).astype(np.uint8)

Image.fromarray(restored_uint8).save(
    "outputs/lab2_restored.png"
)

print("\nЗображення збережено: outputs/lab2_restored.png")


from src.lab2_plots import save_lab2_plots

save_lab2_plots(
    clean,
    noisy,
    restored,
    f,
    result,
    golden_result
)
