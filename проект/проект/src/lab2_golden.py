
import math

TAU = (math.sqrt(5.0) - 1.0) / 2.0


def golden_section(oracle, a, b, eps=1e-5, max_iter=500):
    x1 = b - TAU * (b - a)
    x2 = a + TAU * (b - a)

    f1 = oracle.value(x1)
    f2 = oracle.value(x2)

    rows = []
    k = 0

    while (b - a) > eps and k < max_iter:
        if f1 <= f2:
            decision = "b = x2, успадкували x1"

            b = x2
            x2, f2 = x1, f1

            x1 = b - TAU * (b - a)
            f1 = oracle.value(x1)

        else:
            decision = "a = x1, успадкували x2"

            a = x1
            x1, f1 = x2, f2

            x2 = a + TAU * (b - a)
            f2 = oracle.value(x2)

        k += 1

        rows.append({
            "k": k,
            "a": a,
            "b": b,
            "x1": x1,
            "x2": x2,
            "f1": f1,
            "f2": f2,
            "L": b - a,
            "decision": decision
        })

    x_star = (a + b) / 2.0
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
