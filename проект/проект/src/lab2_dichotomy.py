
def dichotomy(oracle, a, b, eps):
    delta = eps / 3
    rows = []
    k = 0

    while b - a > eps:
        k += 1

        x1 = (a + b - delta) / 2
        x2 = (a + b + delta) / 2

        f1 = oracle.value(x1)
        f2 = oracle.value(x2)

        if f1 <= f2:
            b = x2
        else:
            a = x1

        rows.append({
            "k": k,
            "x1": x1,
            "x2": x2,
            "f1": f1,
            "f2": f2,
            "a": a,
            "b": b,
            "L": b - a
        })

    x_star = (a + b) / 2
    f_star = oracle.value(x_star)

    return {
        "x_star": x_star,
        "f_star": f_star,
        "a": a,
        "b": b,
        "iterations": k,
        "n_f": oracle.n_f,
        "rows": rows
    }
