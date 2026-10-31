
import os
import numpy as np
import matplotlib.pyplot as plt

from src.variant import SEED, NOISE_SIGMA, SIZE
from src.image import clean_image, add_noise
from src.metrics import psnr
from src.lab5_constrained import (
    projected_gradient,
    penalty_method,
    violation
)

clean = clean_image(SIZE)
raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)

lam = 0.325462
L = 1 + 8 * lam

res_pg = projected_gradient(raw, lam, L, accelerated=False)
res_fista = projected_gradient(raw, lam, L, accelerated=True)
res_penalty = penalty_method(raw, lam, L)

results = [
    ("Проєкція градієнта", res_pg),
    ("FISTA", res_fista),
    ("Метод штрафів", res_penalty)
]

print("ЛАБОРАТОРНА 5. ПОРІВНЯННЯ МЕТОДІВ")
print()

for name, res in results:
    v = violation(res["I"])

    print(name)
    print(f'  E = {res["E"]:.10f}')
    print(f'  Nf = {res["n_f"]}')
    print(f'  Ngrad = {res["n_grad"]}')
    print(f'  Разом = {res["n_f"] + res["n_grad"]}')
    print(f'  Порушення = {v["max"]:.10e}')
    print(f'  Поза [0;1] = {v["count"]}')
    print(f'  PSNR = {psnr(clean, np.clip(res["I"], 0, 1)):.4f} дБ')
    print()

os.makedirs("outputs", exist_ok=True)

E_star = min(res_pg["E"], res_fista["E"])


plt.figure(figsize=(10, 5.5))

for name, res, color, style in [
    ("Проєкція градієнта", res_pg, "tab:blue", "-"),
    ("Проєкція градієнта з прискоренням (FISTA)",
     res_fista, "tab:red", "--")
]:
    values = [row["E"] for row in res["rows"]]

    plt.plot(
        range(len(values)),
        values,
        color=color,
        linestyle=style,
        linewidth=1.8,
        label=name
    )

plt.xlabel("Ітерація k")
plt.ylabel("E(I_k)")
plt.title("Збіжність проєкційних методів")
plt.grid(alpha=0.15)
plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/lab5_convergence.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()
