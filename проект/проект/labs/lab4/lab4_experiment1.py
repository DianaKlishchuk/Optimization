
import math

from src.variant import SEED, NOISE_SIGMA, SIZE
from src.image import clean_image, add_noise
from src.lab4_gradient import gradient_descent


clean = clean_image(SIZE)
raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)

lam = 10.0

L = 1 + 8 * lam
mu = 1.0
kappa = L / mu

beta_hb = (
    (math.sqrt(L) - math.sqrt(mu))
    / (math.sqrt(L) + math.sqrt(mu))
) ** 2

h_hb = 4 / (math.sqrt(L) + math.sqrt(mu)) ** 2

beta_nag = (
    (math.sqrt(L) - math.sqrt(mu))
    / (math.sqrt(L) + math.sqrt(mu))
)

print("ДОСЛІД 1. ВЕЛИКЕ ЧИСЛО ОБУМОВЛЕНОСТІ")
print(f"lambda = {lam}")
print(f"L = {L:.6f}")
print(f"kappa = {kappa:.6f}")
print()

methods = {
    "Градієнтний спуск": {
        "kind": "gd"
    },
    "Важка кулька (опт.)": {
        "kind": "hb",
        "beta": beta_hb,
        "step": h_hb
    },
    "Нестеров (сильно опуклий)": {
        "kind": "nag",
        "beta": beta_nag
    }
}

for name, params in methods.items():
    result = gradient_descent(
        noisy,
        lam,
        L,
        max_iter=3000,
        tol=1e-6,
        **params
    )

    print(name)
    print(f"  Ітерацій = {result['iterations']}")
    print(f"  Nf = {result['n_f']}")
    print(f"  Ngrad = {result['n_grad']}")
    print(f"  ||grad|| = {result['grad']:.6e}")
    print(f"  Збіжність = {'так' if result['converged'] else 'ні'}")
    print()
