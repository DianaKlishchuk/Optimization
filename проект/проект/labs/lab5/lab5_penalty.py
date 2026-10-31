
from src.variant import SEED, NOISE_SIGMA, SIZE
from src.image import clean_image, add_noise
from src.lab5_constrained import penalty_method, violation


clean = clean_image(SIZE)
raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)

lam = 0.325462
L = 1 + 8 * lam

print("ЛАБОРАТОРНА 5. МЕТОД ШТРАФІВ")
print(f"lambda = {lam:.6f}")
print(f"L = {L:.6f}")
print()

result = penalty_method(
    raw,
    lam,
    L,
    r0=1.0,
    growth=10.0,
    outer=7,
    inner=100,
    inner_tol=1e-6,
    tol=1e-8
)

print(
    " s | r         | E(I)         | "
    "P(I)        | rP          | Порушення   | Поза"
)

for row in result["rows"]:
    print(
        f'{row["s"]:2d} | '
        f'{row["r"]:9.0f} | '
        f'{row["E"]:12.8f} | '
        f'{row["P"]:.3e} | '
        f'{row["rP"]:.3e} | '
        f'{row["violation"]:.3e} | '
        f'{row["outside"]:4d}'
    )

v = violation(result["I"])

print()
print(f'Зовнішніх ітерацій = {len(result["rows"])}')
print(f'Nf = {result["n_f"]}')
print(f'Ngrad = {result["n_grad"]}')
print(f'Разом звернень = {result["n_f"] + result["n_grad"]}')
print(f'E = {result["E"]:.10f}')
print(f'Максимальне порушення = {v["max"]:.10e}')
print(f'Пікселів поза [0;1] = {v["count"]}')
