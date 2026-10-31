
from src.variant import SEED, NOISE_SIGMA, SIZE
from src.image import clean_image, add_noise
from src.operators import energy, energy_grad
from src.lab4_exact import tikhonov_exact


clean = clean_image(SIZE)
raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)

lam = 0.325462
L = 1 + 8 * lam

h = 0.7
I = noisy.copy()

print("ДОСЛІД 2. ЗАВЕЛИКИЙ КРОК")
print(f"lambda = {lam}")
print(f"L = {L:.6f}")
print(f"Межа стійкості = {2 / L:.6f}")
print(f"Вибраний крок h = {h}")
print()

print("Ітерація | E(I)")

for k in range(51):
    if k % 10 == 0:
        print(f"{k:9d} | {energy(I, noisy, lam):.8f}")

    if k < 50:
        g = energy_grad(I, noisy, lam)
        I = I - h * g
