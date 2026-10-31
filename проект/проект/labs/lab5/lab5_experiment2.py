
from src.variant import SEED, NOISE_SIGMA, SIZE
from src.image import clean_image, add_noise
from src.lab5_constrained import penalty_method, violation


clean = clean_image(SIZE)
raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)

lam = 0.325462
L = 1 + 8 * lam

print("ДОСЛІД 2. ВПЛИВ МНОЖНИКА ЗРОСТАННЯ ШТРАФУ")
print()

for growth in [2.0, 10.0, 100.0]:
    result = penalty_method(
        raw,
        lam,
        L,
        r0=1.0,
        growth=growth,
        outer=20,
        inner=100,
        inner_tol=1e-6,
        tol=1e-8
    )

    v = violation(result["I"])
    last = result["rows"][-1]

    print(f"Множник growth = {growth:.0f}")
    print(f'  Зовнішніх ітерацій = {len(result["rows"])}')
    print(f'  Кінцеве r = {last["r"]:.6e}')
    print(f'  Nf = {result["n_f"]}')
    print(f'  Ngrad = {result["n_grad"]}')
    print(f'  Разом звернень = {result["n_f"] + result["n_grad"]}')
    print(f'  E = {result["E"]:.10f}')
    print(f'  Максимальне порушення = {v["max"]:.6e}')
    print(f'  Пікселів поза [0;1] = {v["count"]}')
    print(f'  rP = {last["rP"]:.6e}')
    print(f'  Критерій rP <= 1e-8 = {"так" if last["rP"] <= 1e-8 else "ні"}')
    print()
