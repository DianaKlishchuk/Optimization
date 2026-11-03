
import numpy as np

from src.variant import SEED, NOISE_SIGMA, SIZE, LAMBDA0
from src.image import clean_image, add_noise
from src.metrics import mse
from src.operators import energy, energy_grad
from src.lab4_exact import tikhonov_exact
from src.oracle import Oracle
from src.lab1_svenn import svenn
from src.lab2_golden import golden_section


clean = clean_image(SIZE)
raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)


def objective(lam):
    lam = float(lam)
    restored = tikhonov_exact(noisy, lam)
    return mse(restored, clean)


oracle = Oracle(objective)

print("ЛАБОРАТОРНА РОБОТА №4")
print()

bracket = svenn(
    oracle,
    LAMBDA0,
    0.2,
    lo=0.0001,
    hi=50.0
)

print("Метод Свенна:")
print(f"  Інтервал: [{bracket['a']:.6f}; {bracket['b']:.6f}]")
print(f"  Nf = {oracle.n_f}")
print()

result = golden_section(
    oracle,
    bracket["a"],
    bracket["b"],
    eps=1e-3
)

lam = result["x_star"]

print("Золотий переріз:")
print(f"  lambda* = {lam:.6f}")
print(f"  MSE = {result['f_star']:.10f}")
print(f"  Nf разом = {oracle.n_f}")
print()

L = 1 + 8 * lam
mu = 1.0
kappa = L / mu
h = 1 / L

print("Константи:")
print(f"  L = {L:.6f}")
print(f"  mu = {mu:.6f}")
print(f"  kappa = {kappa:.6f}")
print(f"  h = {h:.6f}")
print(f"  Межа стійкості = {2 / L:.6f}")
print()

I_star = tikhonov_exact(noisy, lam)

E_star = energy(I_star, noisy, lam)
grad_norm = np.linalg.norm(
    energy_grad(I_star, noisy, lam)
)

print("Еталонний розв'язок:")
print(f"  E* = {E_star:.10f}")
print(f"  ||grad E(I*)|| = {grad_norm:.12e}")
print(f"  MSE = {mse(I_star, clean):.10f}")


from src.lab4_gradient import gradient_descent

import math

beta_hb = (
    (math.sqrt(L) - math.sqrt(mu))
    / (math.sqrt(L) + math.sqrt(mu))
) ** 2

h_hb = 1 / L

beta_nag = (
    (math.sqrt(L) - math.sqrt(mu))
    / (math.sqrt(L) + math.sqrt(mu))
)

print()
print("ГРАДІЄНТНІ МЕТОДИ")
print(f"beta_hb = {beta_hb:.6f}")
print(f"h_hb = {h_hb:.6f}")
print(f"beta_nag = {beta_nag:.6f}")
print()

methods = {
    "Градієнтний спуск": {
        "kind": "gd"
    },
    "Армійо": {
        "kind": "armijo"
    },
    "Важка кулька (опт.)": {
        "kind": "hb",
        "beta": beta_hb,
        "step": h_hb
    },
    "Важка кулька (0.9)": {
        "kind": "hb",
        "beta": 0.9
    },
    "Нестеров (опуклий)": {
        "kind": "nag"
    },
    "Нестеров (сильно опуклий)": {
        "kind": "nag",
        "beta": beta_nag
    }
}

results = {}

for name, params in methods.items():
    result = gradient_descent(
        noisy,
        lam,
        L,
        max_iter=200,
        tol=1e-6,
        x_ref=I_star,
        **params
    )

    results[name] = result

    print(name)
    print(f"  Ітерацій = {result['iterations']}")
    print(f"  Nf = {result['n_f']}")
    print(f"  Ngrad = {result['n_grad']}")
    print(f"  E = {result['E']:.10f}")
    print(f"  ||grad|| = {result['grad']:.6e}")
    print(f"  Збіжність = {'так' if result['converged'] else 'ні'}")
    print()

import os
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

plt.figure(figsize=(10, 6))

for name, res in results.items():
    values = [
        max(row["E"] - E_star, 1e-16)
        for row in res["rows"]
    ]
    plt.semilogy(range(len(values)), values, label=name)

plt.xlabel("Номер ітерації k")
plt.ylabel("E(I_k) - E*")
plt.title("Лабораторна 4. Збіжність за значенням функції")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(
    "outputs/lab4_convergence_f.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

plt.figure(figsize=(10, 6))

for name, res in results.items():
    values = [
        max(row["dist"], 1e-16)
        for row in res["rows"]
    ]
    plt.semilogy(range(len(values)), values, label=name)

plt.xlabel("Номер ітерації k")
plt.ylabel("||I_k - I*||")
plt.title("Лабораторна 4. Збіжність за відстанню")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(
    "outputs/lab4_convergence_dist.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

print("Графіки збережено:")
print("outputs/lab4_convergence_f.png")
print("outputs/lab4_convergence_dist.png")

print("\nПЕРЕВІРКА LAMBDA")

for value in [lam, 0.327995]:
    restored = tikhonov_exact(noisy, value)
    print(f"lambda = {value:.6f}, MSE = {mse(restored, clean):.12f}")

from src.filters import gaussian_blur, bilateral_filter
from src.metrics import mse, psnr

restored4 = np.clip(I_star, 0.0, 1.0)

from src.optimal_parameters import optimal_gaussian_sigma, optimal_bilateral_theta
sigma_gaussian = optimal_gaussian_sigma()
restored2 = gaussian_blur(noisy, sigma_gaussian)

sigma_s, sigma_r = optimal_bilateral_theta()
restored3 = bilateral_filter(noisy, sigma_s, sigma_r)

print()
print("ПОРІВНЯННЯ ЯКОСТІ")

print(f"Шумне зображення: PSNR = {psnr(clean, noisy):.4f} дБ")
print(f"Лабораторна 2 (гаусів фільтр): PSNR = {psnr(clean, restored2):.4f} дБ")
print(f"Лабораторна 3 (білатеральний фільтр): PSNR = {psnr(clean, restored3):.4f} дБ")
print(f"Лабораторна 4 (градієнтні методи): PSNR = {psnr(clean, restored4):.4f} дБ")

import matplotlib.pyplot as plt

images = [
    (clean, "Чисте зображення"),
    (noisy, "Шумне зображення"),
    (restored2, "Гаусів фільтр"),
    (restored3, "Білатеральний фільтр"),
    (restored4, "Градієнтна оптимізація")
]

fig, axes = plt.subplots(1, 5, figsize=(16, 4))

for ax, (img, title) in zip(axes, images):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig(
    "outputs/lab4_comparison.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

print("Зображення збережено: outputs/lab4_comparison.png")

