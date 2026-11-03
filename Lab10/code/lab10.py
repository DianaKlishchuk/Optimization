from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / 'lab10_results'
OUT.mkdir(parents=True, exist_ok=True)

class CountedFunction:
    def __init__(self, fun):
        self.fun, self.nf = fun, 0
        self.values = []
    def __call__(self, x):
        value = float(self.fun(np.asarray(x, float)))
        self.nf += 1
        self.values.append((tuple(np.asarray(x, float)), value))
        return value

def target(x):
    a, b = x
    return a*a + 5*b*b - 2*a*b - 3*a + 11*b

def nonsmooth(x):
    a, b = x
    return 2*abs(a-1) + 3*abs(b+2) + 0.5*(a-1)**2

def rosenbrock(x):
    a, b = x
    return (1-a)**2 + 100*(b-a*a)**2

TRUE_MIN = np.array([0.5, -1.0])
TRUE_F = target(TRUE_MIN)

def explore(F, y, Fy, delta):
    z, Fz = np.array(y, dtype=float, copy=True), float(Fy)
    probes = 0
    for i in range(len(z)):
        t = z.copy()
        t[i] += delta
        Ft = F(t); probes += 1
        if Ft < Fz:
            z, Fz = t, Ft
            continue
        t = z.copy()
        t[i] -= delta
        Ft = F(t); probes += 1
        if Ft < Fz:
            z, Fz = t, Ft
    return z, Fz, probes

def pattern_move(F, new_base, old_base):
    y = new_base + (new_base - old_base)
    return y, F(y)

def hooke_jeeves(fun, x0, delta0=1.0, eps=0.01, max_actions=100000):
    F = CountedFunction(fun)
    base = np.asarray(x0, dtype=float).copy()
    Fb = F(base)
    y, Fy, delta, from_pattern = base.copy(), Fb, float(delta0), False
    k, log, bases = 0, [], [base.copy()]
    def add(action, source, fsource, dest, fdest, probes):
        log.append(dict(k=k, action=action, from_x1=source[0], from_x2=source[1],
            F_from=fsource, delta=delta, to_x1=dest[0], to_x2=dest[1],
            F_to=fdest, probes=probes, NF=F.nf))

    for _ in range(max_actions):
        origin, Forigin = y.copy(), Fy
        z, Fz, probes = explore(F, y, Fy, delta)
        add('дослідний із зразка' if from_pattern else 'дослідний із бази',
            origin, Forigin, z, Fz, probes)
        if Fz < Fb:
            old = base.copy()
            base, Fb = z.copy(), Fz
            k += 1
            bases.append(base.copy())
            y, Fy = pattern_move(F, base, old)
            add('нова база → зразок', base, Fb, y, Fy, 1)
            from_pattern = True
        elif from_pattern:
            add('повернення в базу', y, Fy, base, Fb, 0)
            y, Fy, from_pattern = base.copy(), Fb, False
        else:
            olddelta = delta
            delta /= 2
            add('дроблення Δ: %.8g → %.8g' % (olddelta, delta), base, Fb, base, Fb, 0)
            if delta < eps:
                return dict(x=base, fx=Fb, bases=bases, n_bases=k, NF=F.nf,
                    log=log, delta=delta, history=F.values, converged=True)
    return dict(x=base, fx=Fb, bases=bases, n_bases=k, NF=F.nf,
                log=log, delta=delta, history=F.values, converged=False)

def save_csv(filename, data, columns=None):
    if not data: return
    cols = columns or list(data[0].keys())
    with (OUT / filename).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter=';')
        w.writeheader(); w.writerows(data)

def err(result, truth): return float(np.linalg.norm(result['x']-truth))

def plot_trajectories(function, runs, exact, filename, bounds, title):
    fig, ax = plt.subplots(figsize=(10, 7))
    xs = np.linspace(bounds[0],bounds[1],350)
    ys = np.linspace(bounds[2],bounds[3],350)
    X,Y=np.meshgrid(xs,ys)
    Z = function(np.stack([X,Y]))
    zmin = float(function(exact))
    levels = zmin + np.geomspace(0.02, max(0.03, float(np.nanmax(Z)-zmin)*0.7), 18)
    cs=ax.contour(X,Y,Z,levels=levels,linewidths=0.8,alpha=.6)
    for label,r in runs:
        p=np.array(r['bases'])
        ax.plot(p[:,0],p[:,1],marker='o',markersize=3,linewidth=1.5,label=f'{label}: баз {r["n_bases"]}, NF={r["NF"]}')
    ax.scatter([exact[0]],[exact[1]],marker='*',s=160,label='Аналітичний мінімум',zorder=5)
    ax.set(xlabel='x₁',ylabel='x₂',title=title,xlim=bounds[:2],ylim=bounds[2:])
    ax.grid(alpha=.2);ax.legend(fontsize=9);ax.set_aspect('equal', adjustable='box')
    fig.tight_layout();fig.savefig(OUT / filename,dpi=180);plt.close(fig)


def main():
    print('ЛР 10. Метод Хука–Дживса')
    print('F(x1,x2)=x1²+5x2²−2x1x2−3x1+11x2; старт (5;−5)')
    print('Аналітичний мінімум для контролю: x*=(0.5;−1), F*=',TRUE_F)
    main_runs=[]
    for delta in (1.0,0.7,0.5):
        r=hooke_jeeves(target,[5,-5],delta,0.01)
        main_runs.append((delta,r))
        save_csv(f'quadratic_delta_{str(delta).replace(".","_")}.csv',r['log'])
        print(f'Δ0={delta}: x={r["x"]}, F={r["fx"]:.10f}, баз={r["n_bases"]}, NF={r["NF"]}, похибка={err(r,TRUE_MIN):.8f}, Δ_кін={r["delta"]:.8f}')

    nons=hooke_jeeves(nonsmooth,[5,-5],1.0,0.01)
    save_csv('nonsmooth_G.csv',nons['log'])
    print(f'Негладка G: x={nons["x"]}, G={nons["fx"]:.9f}, баз={nons["n_bases"]}, NF={nons["NF"]}')
    rose_runs=[]
    for e in (0.01,0.0001):
        r=hooke_jeeves(rosenbrock,[-1.2,1.0],1.0,e)
        rose_runs.append((e,r))
        save_csv('rosenbrock_eps_'+('001' if e==.01 else '00001')+'.csv',r['log'])
        print(f'Розенброк ε={e}: x={r["x"]}, f={r["fx"]:.9g}, баз={r["n_bases"]}, NF={r["NF"]}, похибка={err(r,np.array([1.,1.])):.8f}')

    plot_trajectories(target, [(f'Δ₀={d}',r) for d,r in main_runs],TRUE_MIN,
        'quadratic_trajectories.png',[-1.5,7,-7,2], 'Хук–Дживс: власна квадратична функція')
    plot_trajectories(rosenbrock, [(f'ε={e}',r) for e,r in rose_runs],np.array([1.,1.]),
        'rosenbrock_trajectories.png',[-1.6,1.6,-0.6,1.9], 'Хук–Дживс: яр Розенброка')

    summary=[]
    for delta,r in main_runs:
        summary.append(dict(Функція='Квадратична',Метод=f'Хук–Дживс Δ0={delta}',
          n=r['n_bases'],NF=r['NF'],NgradF=0,Похибка=err(r,TRUE_MIN),F=r['fx']))
    for e,r in rose_runs:
        summary.append(dict(Функція='Розенброк',Метод=f'Хук–Дживс ε={e}',
          n=r['n_bases'],NF=r['NF'],NgradF=0,Похибка=err(r,np.array([1.,1.])),F=r['fx']))
    summary.append(dict(Функція='Негладка G',Метод='Хук–Дживс',n=nons['n_bases'],NF=nons['NF'],NgradF=0,
                       Похибка=err(nons,np.array([1.,-2.])),F=nons['fx']))
    for method,n,nf,ng in [('Сферична метрика',22,123,23),
                            ('Антиградієнт + точний пошук',6,232,7),
                            ('Флетчер–Рівс',2,78,3)]:
        summary.append(dict(Функція='Квадратична',Метод=method,n=n,NF=nf,NgradF=ng,Похибка='',F=''))
    save_csv('summary.csv',summary)
    print('Файли записано в',OUT)

if __name__=='__main__': main()
