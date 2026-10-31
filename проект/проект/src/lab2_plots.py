
import os
import numpy as np
import matplotlib.pyplot as plt

from src.metrics import mse, psnr

def save_lab2_plots(
    clean, noisy, restored,
    f, dich_result, golden_result
):
    os.makedirs("outputs", exist_ok=True)

    sigma = np.linspace(0.05, 1.5, 300)
    values = [f(s) for s in sigma]

    plt.figure(figsize=(9, 5))
    plt.plot(sigma, values, label="f(σ) = MSE")

    plt.axvline(
        dich_result["x_star"],
        color="red",
        linestyle="--",
        label="Дихотомія"
    )

    plt.axvline(
        golden_result["x_star"],
        color="green",
        linestyle=":",
        label="Золотий переріз"
    )

    plt.xlabel("σ")
    plt.ylabel("MSE")
    plt.title("Лабораторна №2 — оптимізація параметра розмиття")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/lab2_objective.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))

    for result, label in [
        (dich_result, "Дихотомія"),
        (golden_result, "Золотий переріз")
    ]:
        lengths = [
            row["L"] for row in result["rows"]
        ]
        plt.plot(
            range(1, len(lengths) + 1),
            lengths,
            marker="o",
            markersize=3,
            label=label
        )

    plt.yscale("log")
    plt.xlabel("Номер ітерації")
    plt.ylabel("Довжина відрізка")
    plt.title("Скорочення відрізка невизначеності")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/lab2_convergence.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    images = [clean, noisy, restored]
    titles = [
        "Еталон",
        "Спостереження",
        f"Гаусів фільтр σ* = {golden_result['x_star']:.4f}"
    ]

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(
            f"{title}\n"
            f"PSNR {psnr(clean, image):.2f} дБ  ·  "
            f"MSE {mse(clean, image):.2e}",
            fontsize=10
        )
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(
        "outputs/lab2_comparison.png",
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(fig)

    print("\nГрафіки збережено:")
    print("outputs/lab2_objective.png")
    print("outputs/lab2_convergence.png")
    print("outputs/lab2_comparison.png")
