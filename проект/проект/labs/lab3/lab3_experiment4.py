
from src.variant import SEED, NOISE_SIGMA, SIZE, EPS
from src.image import clean_image, add_noise
from src.filters import gaussian_blur
from src.metrics import mse, psnr
from src.objective2 import make_objective2
from src.oracle import Oracle
from src.lab2_golden import golden_section


clean = clean_image(SIZE)
raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)

gaussian_oracle = Oracle(
    lambda sigma: mse(clean, gaussian_blur(noisy, float(sigma)))
)

gaussian_result = golden_section(
    gaussian_oracle, 0.05, 8.0, eps=EPS
)

fixed_sigma_s = gaussian_result["x_star"]

f = make_objective2(noisy, clean)

oracle = Oracle(
    lambda sigma_r: f([fixed_sigma_s, float(sigma_r)])
)

result = golden_section(
    oracle, 0.01, 1.0, eps=EPS
)

best_sigma_r = result["x_star"]
best_mse = result["f_star"]

from src.optimal_parameters import optimal_bilateral_theta

sigma_s, sigma_r = optimal_bilateral_theta()
mse_2d = f([sigma_s, sigma_r])

print("ДОСЛІД 4. ОДИН ПАРАМЕТР ПРОТИ ДВОХ")
print()

print("Одновимірна оптимізація:")
print(f"  Фіксоване sigma_s = {fixed_sigma_s:.6f}")
print(f"  Знайдене sigma_r = {best_sigma_r:.6f}")
print(f"  MSE = {best_mse:.10f}")
print(f"  PSNR = {psnr(clean, clean) if False else 10 * __import__('math').log10(1.0 / best_mse):.4f} дБ")
print(f"  Nf = {oracle.n_f}")
print()

print("Двовимірна оптимізація:")
from src.optimal_parameters import optimal_bilateral_theta
sigma_s, sigma_r = optimal_bilateral_theta()
print(f"  sigma_s = {sigma_s:.6f}")
print(f"  sigma_r = {sigma_r:.6f}")
print(f"  MSE = {mse_2d:.10f}")
print()

print("Порівняння:")
print(f"  Різниця MSE = {best_mse - mse_2d:.10f}")
print(f"  Погіршення = {(best_mse / mse_2d - 1) * 100:.2f}%")
