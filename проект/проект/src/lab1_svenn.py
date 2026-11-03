
def svenn(oracle, x0, delta0, lo=0.05, hi=8.0, max_iter=40):
    rows = []

    def ask(x, k, step, p, comment):
        value = oracle.value(x)
        rows.append({
            "k": k,
            "x": x,
            "f": value,
            "delta": step,
            "p": p,
            "comment": comment
        })
        return value

    f0 = ask(x0, 0, delta0, 0, "початкова точка")

    xr = min(x0 + delta0, hi)
    fr = ask(xr, 0, delta0, 0, "крок праворуч")

    xl = max(x0 - delta0, lo)
    fl = ask(xl, 0, -delta0, 1, "крок ліворуч")

    if fl >= f0 <= fr:
        return {
            "a": min(xl, xr),
            "b": max(xl, xr),
            "p": 0,
            "iterations": 0,
            "n_f": oracle.n_f,
            "rows": rows
        }

    if fr <= f0:
        p = 0
        step = delta0
        x_cur = xr
        f_cur = fr
    else:
        p = 1
        step = -delta0
        x_cur = xl
        f_cur = fl

    x_prev = x0
    x_new = x_cur
    k = 1

    while k <= max_iter:
        current_step = (2.0 ** k) * step
        x_new = min(max(x_cur + current_step, lo), hi)

        if x_new == x_cur:
            break

        f_new = ask(x_new, k, current_step, p, "розгін")

        if f_new >= f_cur:
            rows[-1]["comment"] = "функція зросла, зупинка"
            break

        x_prev = x_cur
        x_cur = x_new
        f_cur = f_new
        k += 1

    a = min(x_prev, x_new)
    b = max(x_prev, x_new)

    return {
        "a": max(a, lo),
        "b": min(b, hi),
        "p": p,
        "iterations": k,
        "n_f": oracle.n_f,
        "rows": rows
    }
