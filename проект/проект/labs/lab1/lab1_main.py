import os
import numpy as np
import matplotlib.pyplot as plt

from src.variant import SEED, NOISE_SIGMA, X0, DELTA0, SIZE
from src.image import clean_image, add_noise
from src.objective import make_objective
from src.oracle import Oracle
from src.lab1_svenn import svenn

clean = clean_image(SIZE)
raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)

f = make_objective(noisy, clean)
oracle = Oracle(f)

result = svenn(oracle, X0, DELTA0)

print(f"{'k':>2} {'x_k':>9} {'f(x_k)':>12} {'Delta_k':>9} {'p':>2}  Коментар")

for r in result["rows"]:
    print(
        f"{r['k']:>2} "
        f"{r['x']:>9.4f} "
        f"{r['f']:>12.8f} "
        f"{r['delta']:>9.4f} "
        f"{r['p']:>2}  "
        f"{r['comment']}"
    )

print(f"\nВідрізок = [{result['a']:.4f}, {result['b']:.4f}]")
print(f"Довжина = {result['b'] - result['a']:.4f}")
print(f"p = {result['p']}")
print(f"N_f = {result['n_f']}")

grid = np.linspace(0.05, 8.0, 200)
values = [oracle.peek(s) for s in grid]

plt.figure(figsize=(9, 5))

plt.plot(grid, values, label="f(σ) = MSE")

plt.scatter(
    [r["x"] for r in result["rows"]],
    [r["f"] for r in result["rows"]],
    color="red",
    zorder=5,
    label="Пробні точки Свенна"
)

plt.axvspan(
    result["a"],
    result["b"],
    alpha=0.2,
    color="green",
    label=f"Відрізок [{result['a']:.4f}; {result['b']:.4f}]"
)

plt.xscale("log")
plt.xlabel("σ — сила розмиття")
plt.ylabel("f(σ) = MSE")
plt.title("Лабораторна №1 — метод Свенна")
plt.legend()
plt.grid(alpha=0.3)

os.makedirs("outputs", exist_ok=True)

plt.savefig(
    "outputs/lab1_svenn.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("Графік збережено: outputs/lab1_svenn.png")
print("Кількість звернень до оракула:", oracle.n_f)


print("\nЕКСПЕРИМЕНТИ МЕТОДУ СВЕННА")

experiments = [
    ("Основний запуск", X0, DELTA0),
    ("Малий крок", X0, DELTA0 / 10),
    ("Великий крок", X0, DELTA0 * 10),
    ("Інша початкова точка", 3.0, DELTA0)
]

print(f"{'Експеримент':<25} {'x0':>8} {'delta0':>9} {'a':>9} {'b':>9} {'L':>9} {'p':>3} {'Nf':>4}")

for name, x0, delta0 in experiments:
    test_oracle = Oracle(f)
    test_result = svenn(test_oracle, x0, delta0)

    a = test_result["a"]
    b = test_result["b"]

    print(
        f"{name:<25} "
        f"{x0:>8.4f} "
        f"{delta0:>9.4f} "
        f"{a:>9.4f} "
        f"{b:>9.4f} "
        f"{b-a:>9.4f} "
        f"{test_result['p']:>3} "
        f"{test_result['n_f']:>4}"
    )
