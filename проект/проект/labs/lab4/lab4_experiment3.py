from src.optimal_parameters import optimal_lambda

import os
import numpy as np
import matplotlib.pyplot as plt

from src.variant import SEED, NOISE_SIGMA, SIZE
from src.image import clean_image, add_noise
from src.metrics import psnr
from src.lab4_exact import tikhonov_exact

clean = clean_image(SIZE)
raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)

lambda_opt = optimal_lambda()

lambda_values = np.linspace(0.01, 1.0, 100)
psnr_values = []

print("ДОСЛІД 3. ВПЛИВ LAMBDA НА PSNR")
print()
print("lambda | PSNR (дБ)")

for lam in lambda_values:
    restored = tikhonov_exact(noisy, lam)
    value = psnr(clean, np.clip(restored, 0.0, 1.0))
    psnr_values.append(value)

    if len(psnr_values) % 10 == 0:
        print(f"{lam:.4f} | {value:.4f}")

restored_opt = tikhonov_exact(noisy, lambda_opt)
psnr_opt = psnr(clean, np.clip(restored_opt, 0.0, 1.0))

print()
print(f"Оптимальне lambda = {lambda_opt:.6f}")
print(f"PSNR при lambda* = {psnr_opt:.4f} дБ")

best_index = int(np.argmax(psnr_values))
print(f"Найкраще lambda на сітці = {lambda_values[best_index]:.6f}")
print(f"Найкраще PSNR на сітці = {psnr_values[best_index]:.4f} дБ")

os.makedirs("outputs", exist_ok=True)

plt.figure(figsize=(9, 5))
plt.plot(lambda_values, psnr_values, label="PSNR(lambda)")
plt.axvline(
    lambda_opt,
    linestyle="--",
    label=f"lambda* = {lambda_opt:.6f}"
)
plt.scatter([lambda_opt], [psnr_opt])

plt.xlabel("lambda")
plt.ylabel("PSNR, дБ")
plt.title("Лабораторна 4. Вплив lambda на PSNR")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig(
    "outputs/lab4_lambda_psnr.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

print()
print("Графік збережено: outputs/lab4_lambda_psnr.png")
