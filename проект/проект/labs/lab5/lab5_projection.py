from src.optimal_parameters import optimal_lambda

from src.variant import SEED, NOISE_SIGMA, SIZE
from src.image import clean_image, add_noise
from src.lab5_constrained import projected_gradient, violation


clean = clean_image(SIZE)
raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)

lam = optimal_lambda()
L = 1 + 8 * lam

print("ЛАБОРАТОРНА 5. ПРОЄКЦІЙНІ МЕТОДИ")
print(f"lambda = {lam:.6f}")
print(f"L = {L:.6f}")
print()

methods = [
    ("Проєкція градієнта", False),
    ("FISTA", True)
]

for name, accelerated in methods:
    result = projected_gradient(
        raw,
        lam,
        L,
        accelerated=accelerated,
        max_iter=200,
        tol=1e-6
    )

    print(name)
    print(" k | E(I)         | G           | Активних | Порушення")

    for row in result["rows"]:
        if row["k"] <= 5 or row["k"] == result["iterations"]:
            print(
                f'{row["k"]:2d} | '
                f'{row["E"]:12.8f} | '
                f'{row["G"]:.6e} | '
                f'{row["active"]:8d} | '
                f'{row["violation"]:.2e}'
            )

    v = violation(result["I"])

    print()
    print(f'Ітерацій = {result["iterations"]}')
    print(f'Nf = {result["n_f"]}')
    print(f'Ngrad = {result["n_grad"]}')
    print(f'Разом звернень = {result["n_f"] + result["n_grad"]}')
    print(f'E = {result["E"]:.10f}')
    print(f'G = {result["G"]:.6e}')
    print(f'Порушення = {v["max"]:.6e}')
    print(f'Пікселів поза [0;1] = {v["count"]}')
    print(f'Збіжність = {"так" if result["converged"] else "ні"}')
    print()
