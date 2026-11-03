
import math
import numpy as np

def clean_image(size=128):
    """еталонне зображення"""

    img = np.zeros((size, size), dtype=float)

    for i in range(size):
        for j in range(size):
            u = j / (size - 1)
            v = i / (size - 1)

            # ф
            val = 0.25 + 0.35 * (0.6 * u + 0.4 * v)
            val += 0.08 * math.cos(
                2 * math.pi * math.sqrt(
                    (u - 0.5) ** 2 + (v - 0.5) ** 2
                )
            )

            # коло
            if (u - 0.32) ** 2 + (v - 0.34) ** 2 <= 0.16 ** 2:
                val = 0.92

            # квадрат
            if abs(u - 0.70) <= 0.15 and abs(v - 0.30) <= 0.15:
                val = 0.10

            # трик
            if 0.60 < v < 0.90 and abs(u - 0.30) < 0.9 * (v - 0.60):
                val = 0.68

            # шахівниця
            if 0.58 < u < 0.92 and 0.60 < v < 0.90:
                cell = max(3, size // 12)
                val = 0.85 if ((j // cell) + (i // cell)) % 2 == 0 else 0.20

            img[i][j] = min(max(val, 0.0), 1.0)

    # білі смуги 
    for idx in range(4):
        thickness = max(2, (4 - idx) * max(1, size // 64))
        row = int(0.04 * size) + idx * max(3, size // 22)

        if row + thickness < size:
            img[
                row:row + thickness,
                int(0.05 * size):int(0.95 * size)
            ] = 0.98

    return img

def add_noise(clean, sigma, seed):
    """гаусів шум"""
    from src.rng import Rng

    rng = Rng(seed)
    h, w = clean.shape
    raw = np.zeros((h, w), dtype=float)

    for i in range(h):
        for j in range(w):
            raw[i][j] = clean[i][j] + sigma * rng.normal()

    noisy = np.clip(raw, 0.0, 1.0)

    return raw, noisy
