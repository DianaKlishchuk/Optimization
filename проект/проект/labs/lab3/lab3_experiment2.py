
import numpy as np

from src.variant import SEED, NOISE_SIGMA, SIZE
from src.image import clean_image, add_noise
from src.filters import gaussian_blur
from src.metrics import mse
from src.objective2 import make_objective2, LO, HI
from src.oracle import Oracle
from src.lab1_svenn import svenn
from src.lab2_golden import golden_section


clean = clean_image(SIZE)
raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)

gaussian_oracle = Oracle(
    lambda sigma: mse(clean, gaussian_blur(noisy, float(sigma)))
)

gaussian_result = golden_section(
    gaussian_oracle, 0.05, 8.0, eps=1e-4
)

best_gaussian_sigma = gaussian_result["x_star"]

x = np.array([
    min(max(2 * best_gaussian_sigma, 0.5), 4.0),
    min(max(0.15 + 0.1 * NOISE_SIGMA, 0.02), 0.9)
])

f = make_objective2(noisy, clean)
oracle = Oracle(f)

f_x = oracle.value(x)

directions = [
    np.array([1.0, 0.0]),
    np.array([0.0, 1.0])
]

print("ДОСЛІД 2. ЦІНА ПЕРШОГО ЦИКЛУ ПАУЕЛЛА")
print(f"Початкова точка: {x}")
print(f"Початкове обчислення: 1 звернення")
print()

total_svenn = 0
total_golden = 0

for i, direction in enumerate(directions, start=1):

    line = Oracle(
        lambda a: oracle.value(
            np.clip(x + a * direction, LO, HI)
        )
    )

    before = oracle.n_f

    bracket = svenn(
        line, 0.0, 0.1, -1000.0, 1000.0
    )

    n_svenn = oracle.n_f - before

    before = oracle.n_f

    result = golden_section(
        line,
        bracket["a"],
        bracket["b"],
        eps=1e-4
    )

    n_golden = oracle.n_f - before

    alpha = result["x_star"]

    x = np.clip(x + alpha * direction, LO, HI)
    f_x = result["f_star"]

    total_svenn += n_svenn
    total_golden += n_golden

    print(f"Лінійний пошук {i}:")
    print(f"  Свенн: {n_svenn}")
    print(f"  Золотий переріз: {n_golden}")
    print(f"  Перевірка нової точки: 0")
    print(f"  Разом: {n_svenn + n_golden}")
    print()

print("ПІДСУМОК ПЕРШОГО ЦИКЛУ")
print(f"Свенн: {total_svenn}")
print(f"Золотий переріз: {total_golden}")
print("Окрема перевірка нових точок: 0")
print(f"Пошуки разом: {total_svenn + total_golden}")
print(f"З урахуванням початкової точки: {oracle.n_f}")
print(f"Точка після двох пошуків: {x}")
print(f"MSE: {f_x:.10f}")
