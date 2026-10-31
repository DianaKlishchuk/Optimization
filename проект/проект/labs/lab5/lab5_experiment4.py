
import os
import numpy as np
import matplotlib.pyplot as plt

from src.variant import SEED, NOISE_SIGMA, SIZE
from src.image import clean_image, add_noise
from src.lab5_constrained import projected_gradient


clean = clean_image(SIZE)
raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)

lam = 0.325462
L = 1 + 8 * lam

result = projected_gradient(
    raw,
    lam,
    L,
    accelerated=False,
    max_iter=200,
    tol=1e-6
)

I = result["I"]

at_zero = I <= 1e-12
at_one = I >= 1.0 - 1e-12
free = ~(at_zero | at_one)

active_map = np.full(I.shape, 0.5)
active_map[at_zero] = 0.0
active_map[at_one] = 1.0

print("ДОСЛІД 4. КАРТА АКТИВНИХ ОБМЕЖЕНЬ")
print()
print(f"Пікселів на межі 0 = {np.count_nonzero(at_zero)}")
print(f"Пікселів на межі 1 = {np.count_nonzero(at_one)}")
print(f"Вільних пікселів = {np.count_nonzero(free)}")
print(f"Усього активних = {np.count_nonzero(at_zero | at_one)}")
print(f"Усього пікселів = {I.size}")

os.makedirs("outputs", exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

axes[0].imshow(clean, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Еталонне зображення")
axes[0].axis("off")

axes[1].imshow(I, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Розв'язок з обмеженнями")
axes[1].axis("off")

axes[2].imshow(
    active_map,
    cmap="gray",
    vmin=0,
    vmax=1,
    interpolation="nearest"
)
axes[2].set_title("Карта активних обмежень")
axes[2].axis("off")

plt.tight_layout()

plt.savefig(
    "outputs/lab5_active_constraints.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

print()
print("Збережено: outputs/lab5_active_constraints.png")
