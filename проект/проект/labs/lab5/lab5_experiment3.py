from src.optimal_parameters import optimal_lambda

from src.variant import SEED, NOISE_SIGMA, SIZE
from src.image import clean_image, add_noise
from src.lab5_constrained import penalty_method


clean = clean_image(SIZE)
raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)

lam = optimal_lambda()
L = 1 + 8 * lam

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

print("ДОСЛІД 3. ЦІНА ОБУМОВЛЕНОСТІ")
print()
print(f"L = {L:.6f}")
print()
print(" s | r          | L_r          | h = 1/L_r       | kappa")

for row in result["rows"]:
    s = row["s"]
    r = row["r"]

    L_r = L + 2 * r
    h = 1 / L_r
    kappa = L_r

    print(
        f"{s:2d} | "
        f"{r:10.0f} | "
        f"{L_r:12.6f} | "
        f"{h:.10e} | "
        f"{kappa:.6f}"
    )
