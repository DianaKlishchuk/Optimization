import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / 'lab9_results'
OUT.mkdir(exist_ok=True)
EPS = 0.01
GOLDEN_EPS = 1e-6
DELTA0 = 0.01
MAX_ITER = 10000

class Problem:
    def __init__(self, name, f, grad):
        self.name, self.f, self.grad = name, f, grad
        self.nf = self.ng = 0
    def F(self, x):
        self.nf += 1
        return float(self.f(np.asarray(x, dtype=float)))
    def G(self, x):
        self.ng += 1
        return np.asarray(self.grad(np.asarray(x, dtype=float)), dtype=float)

H = np.array([[2., -2.], [-2., 10.]])
b = np.array([-3., 11.])
def quadratic(x):
    x1, x2 = x
    return x1*x1 + 5*x2*x2 - 2*x1*x2 - 3*x1 + 11*x2
def grad_quadratic(x):
    return H @ x + b

def rosenbrock(x):
    x1, x2 = x
    return (1-x1)**2 + 100*(x2-x1*x1)**2
def grad_rosenbrock(x):
    x1, x2 = x
    return np.array([-2*(1-x1)-400*x1*(x2-x1*x1), 200*(x2-x1*x1)])

def svenn(phi, delta=DELTA0, max_steps=70):
    h0 = 0.
    f0 = phi(h0)
    fp = phi(h0 + delta)
    fm = phi(h0 - delta)
    if fp >= f0 and fm >= f0:
        return (-delta, delta)
    direction = 1. if fp < fm else -1.
    h_prev = h0
    h_cur = h0 + direction*delta
    f_cur = fp if direction > 0 else fm
    step = delta
    for _ in range(max_steps):
        step *= 2
        h_next = h_cur + direction*step
        f_next = phi(h_next)
        if not np.isfinite(f_next) or f_next >= f_cur:
            return tuple(sorted((h_prev, h_next)))
        h_prev, h_cur, f_cur = h_cur, h_next, f_next
    raise RuntimeError('Свенн: не вдалося локалізувати мінімум')

def golden(phi, a, b, eps=GOLDEN_EPS):
    if not (a < b and eps > 0):
        raise ValueError('Некоректні межі або точність золотого перерізу')
    r = (np.sqrt(5)-1)/2
    c = b - r*(b-a)
    d = a + r*(b-a)
    fc, fd = phi(c), phi(d)
    while b-a > eps:
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b-r*(b-a)
            fc = phi(c)
        else:
            a, c, fc = c, d, fd
            d = a+r*(b-a)
            fd = phi(d)
    return (a+b)/2

def line_search(problem, x, direction):
    phi = lambda h: problem.F(x+h*direction)
    a, b = svenn(phi)
    return golden(phi, a, b)

def conjugate_gradient(problem, x0, eps=EPS, n=2, restart=True,
                       pure_gradient=False, max_iter=MAX_ITER):
    x = np.array(x0, dtype=float)
    g = problem.G(x)
    d = -g.copy()
    path = [x.copy()]
    rows = []
    checks = []
    for k in range(max_iter):
        value = problem.F(x)
        norm = np.linalg.norm(g)
        if norm <= eps:
            return {'x':x, 'iterations':k, 'rows':rows, 'path':np.array(path),
                    'checks':checks, 'nf':problem.nf, 'ng':problem.ng, 'converged':True}
        if g @ d >= 0:
            d = -g.copy()
        d_used = d.copy()
        h = line_search(problem, x, d_used)
        x_new = x + h*d_used
        g_new = problem.G(x_new)
        beta_raw = float((g_new @ g_new)/(g @ g))
        beta = 0. if pure_gradient or (restart and (k+1)%n == 0) else beta_raw
        d_new = -g_new + beta*d_used
        checks.append({'orthogonality':float(g_new @ d_used),
                       'conjugacy':float(d_new @ H @ d_used) if problem.name=='Квадратична' else float('nan'),
                       'descent':float(g_new @ d_new)})
        rows.append({'k':k, 'x1':x[0], 'x2':x[1], 'F':value,
                     'g1':g[0], 'g2':g[1], 'norm_g':norm,
                     'd1':d_used[0], 'd2':d_used[1], 'h':h, 'beta':beta,
                     'F_next':problem.F(x_new)})
        x, g, d = x_new, g_new, d_new
        path.append(x.copy())
    return {'x':x,'iterations':max_iter,'rows':rows,'path':np.array(path),
            'checks':checks,'nf':problem.nf,'ng':problem.ng,'converged':False}

def spherical_lr8(problem, x0, eps=EPS, max_iter=MAX_ITER):
    x = np.array(x0, dtype=float)
    g = problem.G(x)
    path = [x.copy()]
    rows = []
    for k in range(max_iter):
        value = problem.F(x)
        norm = np.linalg.norm(g)
        if norm <= eps:
            return {'x':x, 'iterations':k,'rows':rows,'path':np.array(path),
                    'nf':problem.nf,'ng':problem.ng,'converged':True}
        d = -g
        h = 1.
        while problem.F(x+h*d) > value + 1e-4*h*(g@d):
            h *= .5
            if h < 1e-15:
                raise RuntimeError('Крок став занадто малим')
        rows.append({'k':k,'x1':x[0],'x2':x[1], 'F':value,'g1':g[0],
                     'g2':g[1],'norm_g':norm,'d1':d[0],'d2':d[1],
                     'h':h,'beta':0.,'F_next':problem.F(x+h*d)})
        x = x+h*d
        g = problem.G(x)
        path.append(x.copy())
    return {'x':x,'iterations':max_iter,'rows':rows,'path':np.array(path),
            'nf':problem.nf,'ng':problem.ng,'converged':False}

def save_table(result, filename):
    if result['rows']:
        with (OUT/filename).open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(result['rows'][0]), delimiter=';')
            writer.writeheader()
            writer.writerows(result['rows'])

def print_rows(title, result):
    print('\n'+title)
    print(' k |          (x1; x2)        |        F        |       (g1; g2)       |  ||g||  |      (d1; d2)       |    h     | beta')
    for r in result['rows']:
        print(f"{r['k']:2d} | ({r['x1']:9.5f};{r['x2']:9.5f}) | {r['F']:13.7f} |"
              f" ({r['g1']:8.4f};{r['g2']:8.4f}) | {r['norm_g']:7.4f} |"
              f" ({r['d1']:9.4f};{r['d2']:9.4f}) | {r['h']:8.6f} | {r['beta']:.6f}")
    print(f"Фініш: x=({result['x'][0]:.8f}; {result['x'][1]:.8f}); "
          f"ітерацій={result['iterations']}; NF={result['nf']}; N∇F={result['ng']}; "
          f"успіх={result['converged']}")

def plot_quadratic(methods):
    x1 = np.linspace(-1.8, 6, 420)
    x2 = np.linspace(-5.8, 2.5, 420)
    X, Y = np.meshgrid(x1, x2)
    Z = X*X+5*Y*Y-2*X*Y-3*X+11*Y
    fig, ax = plt.subplots(figsize=(10,7))
    ax.contour(X,Y,Z, levels=[-6,-4,0,8,20,45,80,130,180], linewidths=1)
    for label, result, marker in methods:
        p = result['path']
        ax.plot(p[:,0], p[:,1], '-'+marker, markersize=4, linewidth=1.7,
                label=f'{label} ({result["iterations"]} ітер.)')
    ax.scatter([.5],[-1.],marker='*',s=160,c='black',label='Аналітичний мінімум')
    ax.set(xlabel='x₁',ylabel='x₂',title='ЛР 9: порівняння трьох траєкторій')
    ax.set_xlim(-1.8,6)
    ax.set_ylim(-5.8,2.5)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=.2)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(OUT/'quadratic_trajectories.png',dpi=180)
    plt.close(fig)

def plot_rosenbrock(methods):
    fig, ax = plt.subplots(figsize=(10,7))
    X,Y=np.meshgrid(np.linspace(-1.6,1.6,400),np.linspace(-.5,2,400))
    Z=(1-X)**2+100*(Y-X*X)**2
    ax.contour(X,Y,Z,levels=[.01,.1,1,4,10,50,150,400],linewidths=.85)
    for label,r,marker in methods:
        p=r['path']
        ax.plot(p[:,0],p[:,1],'-'+marker,markersize=2,linewidth=1.1,
                label=f'{label} ({r["iterations"]} ітер.)')
    ax.scatter([1],[1],s=140,marker='*',c='black',label='Мінімум (1;1)')
    ax.set(xlim=(-1.6,1.6),ylim=(-.5,2),xlabel='x₁',ylabel='x₂',title='Функція Розенброка')
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT/'rosenbrock_trajectories.png',dpi=180)
    plt.close(fig)

def main():
    x0=np.array([5.,-5.])
    x_star = np.linalg.solve(H,-b)
    print('F(x)=x1²+5x2²−2x1x2−3x1+11x2')
    print('H=',H,'\nx*=',x_star,'F*=',quadratic(x_star),'det(H)=',np.linalg.det(H))
    g0=grad_quadratic(x0)
    d0=-g0
    h_exact=-(g0@d0)/(d0@H@d0)
    temp=Problem('Квадратична',quadratic,grad_quadratic)
    h_numeric=line_search(temp,x0,d0)
    print(f'Контроль лінійного пошуку: h_аналітичне={h_exact:.10f}; '
          f'h_чисельне={h_numeric:.10f}; похибка={abs(h_exact-h_numeric):.3e}')
    assert abs(h_exact-h_numeric)<1e-6

    p1=Problem('Квадратична',quadratic,grad_quadratic)
    fr=conjugate_gradient(p1,x0)
    p2=Problem('Квадратична',quadratic,grad_quadratic)
    anti=conjugate_gradient(p2,x0,pure_gradient=True)
    p3=Problem('Квадратична',quadratic,grad_quadratic)
    spherical=spherical_lr8(p3,x0)
    for name,r,file in [('Флетчер–Рівс',fr,'fletcher_reeves.csv'),
                        ('Антиградієнт + точний пошук',anti,'antigradient.csv'),
                        ('Сферична метрика',spherical,'spherical.csv')]:
        print_rows(name,r)
        save_table(r,file)
    print('\nТри перевірки після першого кроку Флетчера–Рівса:')
    c=fr['checks'][0]
    print(f'<g(1),d(0)> = {c["orthogonality"]:.8e} (має бути ≈ 0)')
    print(f'd(1)^T H d(0) = {c["conjugacy"]:.8e} (має бути ≈ 0)')
    print(f'<g(1),d(1)> = {c["descent"]:.8f} (має бути < 0)')
    assert fr['converged'] and fr['iterations']==2, 'Очікувалося рівно 2 ітерації!'
    assert abs(c['orthogonality'])<5e-3
    assert abs(c['conjugacy'])<5e-2
    assert c['descent']<0
    print('Скалярні добутки сусідніх напрямків антиградієнта:')
    for i in range(min(5,len(anti['rows'])-1)):
        a=anti['rows'][i];bb=anti['rows'][i+1]
        print(f'k={i}: {np.dot([a["d1"],a["d2"]],[bb["d1"],bb["d2"]]):.3e}')
    plot_quadratic([('Сферична метрика',spherical,'o'),
                    ('Антиградієнт + точний',anti,'s'),
                    ('Флетчер–Рівс',fr,'^')])

    r1=Problem('Розенброк',rosenbrock,grad_rosenbrock)
    rb_restart=conjugate_gradient(r1,[-1.2,1],restart=True,max_iter=5000)
    r2=Problem('Розенброк',rosenbrock,grad_rosenbrock)
    rb_no_restart=conjugate_gradient(r2,[-1.2,1],restart=False,max_iter=5000)
    r3=Problem('Розенброк',rosenbrock,grad_rosenbrock)
    rb_spherical=spherical_lr8(r3,[-1.2,1],max_iter=5000)
    for name,r,file in [('Розенброк: із перезапуском',rb_restart,'rosenbrock_restart.csv'),
                        ('Розенброк: без перезапуску',rb_no_restart,'rosenbrock_no_restart.csv'),
                        ('Розенброк: сферична метрика',rb_spherical,'rosenbrock_spherical.csv')]:
        save_table(r,file)
        print(f'{name}: x={r["x"]}, F={rosenbrock(r["x"]):.9g}, '
              f'n={r["iterations"]}, NF={r["nf"]}, N∇F={r["ng"]}, '
              f'успіх={r["converged"]}')
    plot_rosenbrock([('Флетчер–Рівс, із перезапуском',rb_restart,'o'),
                     ('Флетчер–Рівс, без перезапуску',rb_no_restart,'s')])
    summary=[('Квадратична','Сферична метрика',spherical),
             ('Квадратична','Антиградієнт + точний пошук',anti),
             ('Квадратична','Флетчер–Рівс',fr),
             ('Розенброк','Сферична метрика',rb_spherical),
             ('Розенброк','Флетчер–Рівс + перезапуск',rb_restart),
             ('Розенброк','Флетчер–Рівс без перезапуску',rb_no_restart)]
    with (OUT/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f,delimiter=';')
        w.writerow(['Функція','Метод','n','NF','NgradF','NF+NgradF','Збіжність'])
        for fun,name,r in summary:
            w.writerow([fun,name,r['iterations'],r['nf'],r['ng'],r['nf']+r['ng'],r['converged']])
    print('\nЗВЕДЕНА ТАБЛИЦЯ')
    print(f'{"Функція":15} {"Метод":39} {"n":>6} {"NF":>7} {"NgradF":>7} {"Разом":>8}')
    for fun,name,r in summary:
        print(f'{fun:15} {name:39} {r["iterations"]:6d} {r["nf"]:7d} {r["ng"]:7d} {r["nf"]+r["ng"]:8d}')
    print('\nФайли таблиць і графіків:',OUT)

if __name__=='__main__':
    main()
