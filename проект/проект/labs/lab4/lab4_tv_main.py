from src.optimal_parameters import optimal_lambda

import os
import numpy as np
import matplotlib.pyplot as plt

from src.variant import SEED, NOISE_SIGMA, SIZE
from src.image import clean_image, add_noise
from src.metrics import mse, psnr
from src.lab4_exact import tikhonov_exact
from src.lab4_tv import tv_nesterov

clean = clean_image(SIZE)
raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)

lambda_tikhonov = optimal_lambda()
restored_tikhonov = tikhonov_exact(
    noisy, lambda_tikhonov
)

lambda_tv = 0.1
restored_tv, iterations = tv_nesterov(
    noisy,
    lambda_tv,
    eps=0.01,
    max_iter=3000
)

restored_tikhonov = np.clip(restored_tikhonov, 0, 1)
restored_tv = np.clip(restored_tv, 0, 1)

print("ПОРІВНЯННЯ МОДЕЛЕЙ")
print(f"TV: ітерацій = {iterations}")
print()

images = [
    (noisy, "Спостереження"),
    (restored_tikhonov, "Модель Тихонова"),
    (restored_tv, "Модель повної варіації"),
    (clean, "Еталон")
]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, (img, title) in zip(axes, images):
    value_mse = mse(clean, img)
    value_psnr = psnr(clean, img)

    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(
        f"{title}\nPSNR = {value_psnr:.2f} дБ\nMSE = {value_mse:.6f}",
        fontsize=10
    )
    ax.axis("off")

os.makedirs("outputs", exist_ok=True)
plt.tight_layout()

plt.savefig(
    "outputs/lab4_tv_comparison.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

print(f"Шумне: PSNR = {psnr(clean, noisy):.4f} дБ")
print(f"Тихонов: PSNR = {psnr(clean, restored_tikhonov):.4f} дБ")
print(f"TV: PSNR = {psnr(clean, restored_tv):.4f} дБ")
print()
print("Збережено: outputs/lab4_tv_comparison.png")
