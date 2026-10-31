import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

X0 = np.array([5.0, -5.0])
EPS = 0.01
H_VALUES = (1.0, 0.7)
METRICS = ("sphere", "cube", "octa")
NAMES = {"sphere": "Сферична", "cube": "Кубічна", "octa": "Октаедрична"}
OUTPUT = Path(__file__).resolve().parent / "outputs"
OUTPUT.mkdir(exist_ok=True)


def F(x):
    x1, x2 = x
    return float(x1**2 + 5*x2**2 - 2*x1*x2 - 3*x1 + 11*x2)


def grad(x):
    x1, x2 = x
    return np.array([2*x1 - 2*x2 - 3, 10*x2 - 2*x1 + 11], dtype=float)


def direction(g, metric):
    if metric == "sphere":
        return -g / np.linalg.norm(g)
    if metric == "cube":
        return -np.sign(g)
    if metric == "octa":
        d = np.zeros_like(g)
        j = int(np.argmax(np.abs(g)))
        d[j] = -np.sign(g[j])
        return d
    raise ValueError(metric)


def solve(metric, h0, max_iter=5000):
    x = X0.copy()
    fx = F(x)
    h = float(h0)
    nf, ng = 1, 0
    rows = []
    points = [x.copy()]

    for k in range(max_iter + 1):
        g = grad(x)
        ng += 1
        norm = float(np.linalg.norm(g))
        if norm <= EPS:
            return dict(metric=metric, h0=h0, x=x, fx=fx, n=k, nf=nf, ng=ng,
                        rows=rows, points=np.array(points), norm=norm)
        if k == max_iter:
            break
        d = direction(g, metric)
        if np.dot(g, d) >= 0:
            raise RuntimeError("Напрямок не є напрямком спуску")
        halves = 0
        while True:
            xb = x + h*d
            fb = F(xb)
            nf += 1
            if fb < fx:
                break
            h /= 2
            halves += 1
            if h < 1e-10:
                raise RuntimeError("Крок став надто малим")
        rows.append(dict(k=k+1, x1=x[0], x2=x[1], F=fx, g1=g[0], g2=g[1],
                         grad_norm=norm, d1=d[0], d2=d[1], h=h,
                         halvings=halves, F_next=fb))
        x, fx = xb, fb
        points.append(x.copy())
    raise RuntimeError(f"Не досягнуто збіжності для {metric} з h0={h0}")


def save_csv(path, rows, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def plot_runs(runs, h0, x_exact):
    gx = np.linspace(-1.6, 5.6, 350)
    gy = np.linspace(-5.6, 0.8, 350)
    A, B = np.meshgrid(gx, gy)
    Z = A*A + 5*B*B - 2*A*B - 3*A + 11*B
    fig, ax = plt.subplots(figsize=(9, 7))
    levels = np.geomspace(0.03, max(70, Z.max()+6.25), 20) - 6.25
    cs = ax.contour(A, B, Z, levels=levels, cmap="Greys", alpha=0.65)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
    colors = {"sphere": "tab:blue", "cube": "tab:red", "octa": "tab:green"}
    for name in METRICS:
        pts = runs[name]["points"]
        ax.plot(pts[:,0], pts[:,1], "-o", color=colors[name], markersize=3,
                linewidth=1.5, label=f"{NAMES[name]} ({len(pts)-1} кроків)")
    ax.scatter([X0[0]], [X0[1]], marker="s", color="black", s=80, zorder=5, label="Початок")
    ax.scatter([x_exact[0]], [x_exact[1]], marker="*", color="gold", edgecolor="black",
               s=180, zorder=6, label="Точний мінімум")
    ax.set(xlabel="x₁", ylabel="x₂", title=f"ЛР 8. Лінії рівня й траєкторії, h₀ = {h0}")
    ax.grid(alpha=0.2)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(OUTPUT / f"trajectories_h{str(h0).replace('.', '_')}.png", dpi=180)
    plt.close(fig)


def main():
    H = np.array([[2.0, -2.0], [-2.0, 10.0]])
    b = np.array([3.0, -11.0])
    exact = np.linalg.solve(H, b)
    print("ЛАБОРАТОРНА №8")
    print(f"Точний мінімум: x* = ({exact[0]:.6f}; {exact[1]:.6f}), F* = {F(exact):.6f}")
    print("Перевірка градієнта при h = 1e-5:")
    for i in range(2):
        e = np.zeros(2)
        e[i] = 1
        numeric = (F(X0+1e-5*e)-F(X0-1e-5*e))/(2e-5)
        analytic = grad(X0)[i]
        print(f"  x{i+1}: аналітично {analytic:.9f}, чисельно {numeric:.9f}, похибка {abs(numeric-analytic):.3e}")
    print("\nПочаток: F(x0) =", F(X0), "градієнт =", grad(X0))
    summary = []
    for h0 in H_VALUES:
        runs = {}
        for metric in METRICS:
            res = solve(metric, h0)
            runs[metric] = res
            dist = float(np.linalg.norm(res["x"]-exact))
            summary.append(dict(metric=NAMES[metric], h0=h0, n=res["n"],
                                NF=res["nf"], Ngrad=res["ng"], total=res["nf"]+res["ng"],
                                x1=res["x"][0], x2=res["x"][1], F=res["fx"],
                                grad_norm=res["norm"], error=dist))
            name = f"iterations_{metric}_h{str(h0).replace('.', '_')}.csv"
            save_csv(OUTPUT / name, res["rows"],
                     ["k", "x1", "x2", "F", "g1", "g2", "grad_norm", "d1", "d2", "h", "halvings", "F_next"])
            print(f"\n{NAMES[metric]}, h0={h0}: n={res['n']}, NF={res['nf']}, Ngrad={res['ng']}, ",
                  f"x*=({res['x'][0]:.6f}, {res['x'][1]:.6f}), F={res['fx']:.9f}, error={dist:.6e}")
            print(f"{'k':>3} {'x1':>11} {'x2':>11} {'F':>13} {'g1':>10} {'g2':>10} {'||g||':>11} {'d1':>9} {'d2':>9} {'h':>10} {'др.':>4} {'F_next':>13}")
            for row in res["rows"]:
                print(f"{row['k']:3d} {row['x1']:11.6f} {row['x2']:11.6f} {row['F']:13.6f} {row['g1']:10.4f} {row['g2']:10.4f} {row['grad_norm']:11.6f} {row['d1']:9.5f} {row['d2']:9.5f} {row['h']:10.6f} {row['halvings']:4d} {row['F_next']:13.6f}")
        plot_runs(runs, h0, exact)
    save_csv(OUTPUT / "summary.csv", summary,
             ["metric", "h0", "n", "NF", "Ngrad", "total", "x1", "x2", "F", "grad_norm", "error"])
    print("\nЗведена таблиця")
    for item in summary:
        print(f"{item['metric']:13s} h0={item['h0']} n={item['n']:2d} NF={item['NF']:2d} Ngrad={item['Ngrad']:2d} error={item['error']:.6e}")
    print("\nCSV та графіки збережено до", OUTPUT)


if __name__ == "__main__":
    main()
