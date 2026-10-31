
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_contours(f, nm_points, powell_points, optimum):
    os.makedirs("outputs", exist_ok=True)

    sigma_s = np.linspace(0.3, 5.0, 40)
    sigma_r = np.linspace(0.01, 1.0, 40)

    X, Y = np.meshgrid(sigma_s, sigma_r)
    Z = np.zeros_like(X)

    print("Обчислення контурного графіка...")

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f([X[i, j], Y[i, j]])

    nm_points = np.asarray(nm_points, dtype=float)
    powell_points = np.asarray(powell_points, dtype=float)
    optimum = np.asarray(optimum, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))

    levels = np.quantile(
        Z,
        np.linspace(0.0, 0.6, 25)
    )

    contours = ax.contourf(
        X, Y, Z,
        levels=levels,
        alpha=0.75,
        extend="max"
    )

    fig.colorbar(
        contours,
        ax=ax,
        label="f(sigma_s, sigma_r) = MSE"
    )

    ax.plot(
        nm_points[:, 0],
        nm_points[:, 1],
        "o-",
        markersize=3,
        linewidth=1,
        label="Нелдер–Мід"
    )

    ax.plot(
        powell_points[:, 0],
        powell_points[:, 1],
        "s-",
        markersize=5,
        linewidth=1.5,
        label="Пауелл"
    )

    ax.scatter(
        optimum[0],
        optimum[1],
        marker="*",
        s=220,
        color="yellow",
        edgecolor="black",
        zorder=5,
        label="Оптимум"
    )

    ax.set_xlabel("sigma_s — просторовий масштаб")
    ax.set_ylabel("sigma_r — фотометричний поріг")
    ax.set_title(
        "Лабораторна 3: лінії рівня і траєкторії методів"
    )

    ax.set_xlim(0.3, 5.0)
    ax.set_ylim(0.01, 1.0)

    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()

    filename = "outputs/lab3_contours.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Графік збережено: {filename}")
