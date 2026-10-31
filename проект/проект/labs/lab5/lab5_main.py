
import os
import numpy as np
import matplotlib.pyplot as plt

from src.variant import SEED, NOISE_SIGMA, SIZE
from src.image import clean_image, add_noise
from src.operators import energy, energy_grad
from src.lab4_exact import tikhonov_exact

clean = clean_image(SIZE)
raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)

lam = 0.325462

unconstrained = tikhonov_exact(raw, lam)


def show_violations(I, name):
    outside = (I < 0.0) | (I > 1.0)

    count = int(np.count_nonzero(outside))
    percent = count / I.size * 100

    max_violation = max(
        float(np.max(np.maximum(0.0, -I))),
        float(np.max(np.maximum(0.0, I - 1.0)))
    )

    print(name)
    print(f"  Поза [0; 1]: {count} пікселів ({percent:.2f}%)")
    print(f"  Мінімальне значення: {np.min(I):.6f}")
    print(f"  Максимальне значення: {np.max(I):.6f}")
    print(f"  Максимальне порушення: {max_violation:.8f}")
    print()


print("ЛАБОРАТОРНА РОБОТА №5")
print("СИРЕ СПОСТЕРЕЖЕННЯ")
print()

show_violations(raw, "Сире спостереження:")
show_violations(unconstrained, "Безумовний розв'язок:")

print(f"E безумовного розв'язку = {energy(unconstrained, raw, lam):.10f}")

os.makedirs("outputs", exist_ok=True)

raw_outside = np.count_nonzero((raw < 0) | (raw > 1))
unc_outside = np.count_nonzero(
    (unconstrained < 0) | (unconstrained > 1)
)

plt.figure(figsize=(10, 5.5))

plt.axvspan(
    0, 1,
    color="green",
    alpha=0.06,
    label="Допустима область Ω"
)

bins = np.linspace(
    min(raw.min(), unconstrained.min()) - 0.03,
    max(raw.max(), unconstrained.max()) + 0.03,
    120
)

plt.hist(
    raw.ravel(),
    bins=bins,
    histtype="step",
    linewidth=1.8,
    color="tab:blue",
    label=f"Спостереження (сире) (поза [0,1]: {raw_outside})"
)

plt.hist(
    unconstrained.ravel(),
    bins=bins,
    histtype="step",
    linewidth=1.8,
    color="tab:red",
    label=f"Безумовний розв'язок (поза [0,1]: {unc_outside})"
)

plt.axvline(0, color="green", linestyle="--", linewidth=1.2)
plt.axvline(1, color="green", linestyle="--", linewidth=1.2)

plt.yscale("log")
plt.ylim(bottom=1)

plt.xlabel("Значення пікселя")
plt.ylabel("Кількість пікселів")
plt.title("Дотримання фізичних меж пікселів")

plt.grid(alpha=0.15)
plt.legend(loc="lower center", fontsize=9)

plt.tight_layout()
plt.savefig(
    "outputs/lab5_histogram.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

print("Гістограму збережено: outputs/lab5_histogram.png")
