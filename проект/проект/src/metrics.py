
import math
import numpy as np

def mse(a, b):
    return float(np.mean((a - b) ** 2))

def psnr(a, b):
    error = mse(a, b)
    if error == 0:
        return float("inf")
    return 10 * math.log10(1.0 / error)
