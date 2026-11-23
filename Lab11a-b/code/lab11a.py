RESULT_FOLDER='lab11_A_results'
import csv, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
X0=np.array([5.,-5.]); XFREE=np.array([.5,-1.]); H=np.array([[2.,-2.],[-2.,10.]])
OUT=Path(__file__).resolve().parent/RESULT_FOLDER

def F(x):
    u,v=x; return float(u*u+5*v*v-2*u*v-3*u+11*v)
def gradF(x):
    u,v=x; return np.array([2*u-2*v-3.,10*v-2*u+11.])
class Count:
    def __init__(self,fn): self.fn=fn; self.n=0
    def __call__(self,x): self.n+=1; return self.fn(x)
def affine(a,b,d):
    v=np.array([float(a),float(b)]); d=float(d)
    if np.linalg.norm(v)==0: raise ValueError('Нормаль обмеження має бути ненульовою')
    return (lambda x:float(v@x+d),lambda x:v.copy()),v,d

def penalty(cons, r, equalities=()):
    def phi(x): return F(x)+r*(sum(max(0.,g(x))**2 for g,gg in cons)+sum(h(x)**2 for h,gh in equalities))
    def grad(x):
        v=gradF(x)
        for g,gg in cons: v+=2*r*max(0.,g(x))*gg(x)
        for h,gh in equalities: v+=2*r*h(x)*gh(x)
        return v
    return phi,grad

def barrier(cons, mu):
    def phi(x):
        vals=[g(x) for g,gg in cons]
        if any(v>=0 for v in vals): return float('inf')
        return F(x)-mu*sum(math.log(-v) for v in vals)
    def grad(x):
        v=gradF(x)
        for g,gg in cons:
            val=g(x)
            if val>=0: raise ValueError('Бар’єр: точка не є строго допустимою')
            v-=mu/val*gg(x)
        return v
    return phi,grad

def svenn(phi, initial=.01, limit=65):
    f0=phi(0.); f1=phi(initial)
    if not math.isfinite(f0): raise ValueError('Старт лінійного пошуку недопустимий')
    if not math.isfinite(f1) or f1>=f0: return 0.,initial
    prev,cur,fc=0.,initial,f1; step=initial
    for _ in range(limit):
        step*=2; nxt=cur+step; fn=phi(nxt)
        if not math.isfinite(fn) or fn>=fc: return prev,nxt
        prev,cur,fc=cur,nxt,fn
    raise RuntimeError('Свенн не локалізував мінімум')

def golden(phi,a,b,eps=1e-6):
    t=(math.sqrt(5)-1)/2
    p=b-t*(b-a);q=a+t*(b-a);fp=phi(p);fq=phi(q)
    for _ in range(300):
        if b-a<=eps:break
        if fp<=fq:
            b,q,fq=q,p,fp;p=b-t*(b-a);fp=phi(p)
        else:
            a,p,fp=p,q,fq;q=a+t*(b-a);fq=phi(q)
    return (a+b)/2

def line_search(phi,x,d):
    f=lambda h:phi(x+h*d)
    a,b=svenn(f)
    return golden(f,a,b)

def fr(phi,grad,x0,tol=.01,maxit=1200):
    x=np.asarray(x0,float).copy();g=grad(x);d=-g;trace=[]
    for k in range(maxit+1):
        norm=float(np.linalg.norm(g))
        if norm<=tol: return x,k,trace,True
        if k==maxit: break
        if g@d>=-1e-14*(g@g):d=-g.copy()
        h=line_search(phi,x,d)
        xx=x+h*d
        gg=grad(xx)
        beta=float((gg@gg)/(g@g))
        if (k+1)%2==0:beta=0.
        trace.append(dict(k=k,x1=x[0],x2=x[1],Phi=phi(x),grad1=g[0],grad2=g[1],norm=norm,d1=d[0],d2=d[1],h=h,beta_next=beta))
        d=-gg+beta*d;x,g=xx,gg
    return x,maxit,trace,False

def spherical(phi,grad,x0,tol=.01,maxit=20000):
    x=np.asarray(x0,float).copy();g=grad(x)
    for k in range(maxit+1):
        if np.linalg.norm(g)<=tol:return x,k,True
        if k==maxit: break
        d=-g;fx=phi(x);h=1.
        for _ in range(100):
            if phi(x+h*d)<=fx+1e-4*h*(g@d):break
            h*=.5
        else: return x,k,False
        x=x+h*d;g=grad(x)
    return x,maxit,False

def run_outer(cons,x0=X0,c=10.,inner_eps=.01,outer_eps=.01, r0=1.,eq=(),maxouter=19, maxinner=1200):
    x=np.asarray(x0,float).copy();r=r0;rows=[];path=[x.copy()];totalF=totalG=0;first=[]
    for k in range(1,maxouter+1):
        p,g=penalty(cons,r,eq);pc,gc=Count(p),Count(g)
        old=x.copy();x,iterations,trace,ok=fr(pc,gc,x,inner_eps,maxinner)
        if k==1:first=trace
        totalF+=pc.n;totalG+=gc.n;path.append(x.copy())
        ss=[max(0.,g(x)) for g,gg in cons];hs=[h(x) for h,gh in eq];P=sum(s*s for s in ss)+sum(z*z for z in hs)
        row=dict(k=k,r=r,start1=old[0],start2=old[1],FR=iterations,NPhi=pc.n,NgradPhi=gc.n,
                 x1=x[0],x2=x[1],F=F(x),P=P,rP=r*P,violation=max(ss+[abs(z) for z in hs]+[0]),
                 lambdas=str([round(2*r*s,9) for s in ss]+[round(2*r*z,9) for z in hs]),
                 NPhi_total=totalF,NgradPhi_total=totalG,converged=ok)
        for j,s in enumerate(ss):row[f'g{j+1}']=cons[j][0](x);row[f'lambda{j+1}']=2*r*s
        for j,z in enumerate(hs):row[f'h{j+1}']=z;row[f'lambda_h{j+1}']=2*r*z
        rows.append(row)
        if not ok:break
        if r*P<=outer_eps:break
        r*=c
    return dict(x=x,rows=rows,path=np.array(path),first=first,ok=rows[-1]['converged'],NPhi=totalF,Ngrad=totalG)

def run_barrier(cons,inside,mu0=10.,c=10.,eps=.01,inner_eps=.01,maxouter=9):
    x=np.asarray(inside,float).copy()
    if any(g(x)>=0 for g,gg in cons):raise ValueError('Бар’єр: старт мусить бути строго всередині Ω')
    mu=mu0;rows=[];path=[x.copy()];nf=ng=0
    for k in range(1,maxouter+1):
        p,g=barrier(cons,mu);pc,gc=Count(p),Count(g)
        x,n,trace,ok=fr(pc,gc,x,inner_eps,maxit=1800)
        nf+=pc.n;ng+=gc.n;path.append(x.copy())
        vals=[gj(x) for gj,gg in cons]
        row=dict(k=k,mu=mu,x1=x[0],x2=x[1],F=F(x),g=vals[0],violation=max([0]+vals),
                 lambdas=str([-mu/v for v in vals]),FR=n,NPhi=pc.n,NgradPhi=gc.n,
                 NPhi_total=nf,NgradPhi_total=ng,converged=ok)
        for j,v in enumerate(vals):row[f'g{j+1}']=v;row[f'lambda{j+1}']=-mu/v
        rows.append(row)
        if not ok:break
        if mu*len(cons)<=eps*(1+1e-12):break
        mu/=c
    return dict(x=x,rows=rows,path=np.array(path),NPhi=nf,Ngrad=ng,ok=rows[-1]['converged'])

def exact_linear(cons):
    candidates=[];m=len(cons)
    for mask in range(1<<m):
        active=[j for j in range(m) if mask&(1<<j)]
        if len(active)>2:continue
        if not active:x=XFREE.copy();lam=np.zeros(m)
        else:
            A=np.array([cons[j][1](XFREE) for j in active]); b=np.array([cons[j][0](np.zeros(2)) for j in active])
            K=np.block([[H,A.T],[A,np.zeros((len(active),len(active)))]])
            rhs=np.concatenate([H@XFREE,-b])
            try:sol=np.linalg.solve(K,rhs)
            except np.linalg.LinAlgError:continue
            x=sol[:2];lam=np.zeros(m)
            for j,v in zip(active,sol[2:]):lam[j]=v
        if np.any(lam<-1e-7):continue
        if any(g(x)>1e-7 for g,gg in cons):continue
        candidates.append((F(x),x,lam))
    if not candidates:raise ValueError('Не вдалося розв’язати KKT; перевірте обмеження')
    return min(candidates,key=lambda item:item[0])

def circle_constraint(R):
    def g(x):return float(R*R-x@x)
    def gg(x):return -2*np.asarray(x,float)
    return (g,gg)

def write_csv(name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter=';');w.writeheader();w.writerows(rows)

def plot_main(cons, linear_solution, runs):
    xx=np.linspace(-1.5,8,300);yy=np.linspace(-7,5,300);X,Y=np.meshgrid(xx,yy)
    Z=X**2+5*Y**2-2*X*Y-3*X+11*Y
    c=cons[0];g=c[1](np.zeros(2));gv=g[0]*X+g[1]*Y+c[0](np.zeros(2))
    fig,ax=plt.subplots(figsize=(10,7));ax.contour(X,Y,Z,levels=F(XFREE)+np.geomspace(.1,500,20),alpha=.65)
    ax.contourf(X,Y,gv,levels=[-1e5,0],alpha=.12)
    ax.contour(X,Y,gv,levels=[0],linewidths=2)
    for name,r in runs:
        path=r['path'];ax.plot(path[:,0],path[:,1],'-o',ms=3,label=name)
    ax.scatter(*XFREE,marker='x',s=110,label='Безумовний мінімум')
    ax.scatter(*linear_solution,marker='*',s=150,label='Умовний мінімум')
    ax.legend(fontsize=8);ax.grid(alpha=.2);ax.set(xlabel='x₁',ylabel='x₂',title='ЛР11 ч.2: зовнішній штраф і внутрішній бар’єр')
    fig.tight_layout();fig.savefig(OUT/'external_vs_barrier.png',dpi=170);plt.close(fig)

def plot_study(runs):
    fig,ax=plt.subplots(figsize=(9,5))
    for name,run in runs:
        rs=[r['r'] for r in run['rows']];vs=[max(1e-15,r['violation']) for r in run['rows']]
        ax.loglog(rs,vs,'-o',label=name)
    ax.set(xlabel='r',ylabel='max(0,g)',title='Порушення залежно від параметра r')
    ax.grid(which='both',alpha=.25);ax.legend();fig.tight_layout();fig.savefig(OUT/'violation_vs_r.png',dpi=170);plt.close(fig)

def plot_circle(circle_runs, R, exact_candidates):
    XX=np.linspace(-4.2,4.2,320)
    YY=np.linspace(-4.2,4.2,320)
    X,Y=np.meshgrid(XX,YY)
    Z=X**2+5*Y**2-2*X*Y-3*X+11*Y
    fig,ax=plt.subplots(figsize=(8,7))
    ax.contour(X,Y,Z,levels=np.linspace(float(np.min(Z)),float(np.max(Z))*.5,22),alpha=.6)
    theta=np.linspace(0,2*np.pi,600)
    ax.plot(R*np.cos(theta),R*np.sin(theta),'k--',label='g₃=0: коло R=3')
    for name,r in circle_runs:
        points=r['path']
        ax.plot(points[:,0],points[:,1],'-o',ms=4,label=name)
    for idx,(_,x,val) in enumerate(exact_candidates):
        ax.scatter(*x,marker='*',s=130,label=f'Мінімум {idx+1}: F={val:.4f}')
    ax.set(xlabel='x₁',ylabel='x₂',title='Варіант 6: поза колом, два старти')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'circle_two_starts.png',dpi=170);plt.close(fig)

def circle_boundary_minima(R, resolution=30000):
    theta=np.linspace(-math.pi,math.pi,resolution+1)
    vals=np.array([F(R*np.array([math.cos(t),math.sin(t)])) for t in theta])
    found=[]
    for i in range(1,len(theta)-1):
        if vals[i]<=vals[i-1] and vals[i]<vals[i+1]:
            fn=lambda t:F(R*np.array([math.cos(t),math.sin(t)]))
            ang=golden(fn,theta[i-1],theta[i+1],eps=1e-11)
            x=R*np.array([math.cos(ang),math.sin(ang)])
            found.append((ang,x,F(x)))
    return found

def print_summary(label,result):
    end=result['rows'][-1]
    print(f'{label:23s} | зовн={len(result["rows"]):2d} | NΦ={result["NPhi"]:5d} | N∇Φ={result["Ngrad"]:4d} '
          f'| x=({result["x"][0]:.6f},{result["x"][1]:.6f}) | F={F(result["x"]):.6f} '
          f'| порушення={end["violation"]:.3e} | λ={end["lambdas"]} | ok={result["ok"]}')


def main():
    OUT.mkdir(exist_ok=True)
    cons1, normal, _ = affine(-1., 0., 3.)
    fstar, xstar, lam = exact_linear([cons1])
    print('ЛР 11А — Зовнішні штрафи, варіант 6')
    print('F=x1²+5x2²−2x1x2−3x1+11x2, x0=(5;−5), g1=3−x1≤0')
    print('Аналітично: x*=',xstar,' F*=',fstar,' lambda=',lam)
    run=run_outer([cons1],x0=X0,c=10.,inner_eps=.01,outer_eps=.01,r0=1.)
    print_summary('Зовнішній штраф',run)
    write_csv('outer_protocol.csv',run['rows'])
    write_csv('inner_r1_FR.csv',run['first'])
    a=np.linalg.solve(H,normal)
    q=float(normal@a)
    gfree=cons1[0](XFREE)
    analytic=[]
    for row in run['rows']:
        r=row['r']; exact=XFREE - (2*r*gfree/(1+2*r*q))*a
        analytic.append(dict(r=r,analytic_x1=exact[0],analytic_x2=exact[1],
                             numerical_x1=row['x1'],numerical_x2=row['x2'],
                             position_error=float(np.linalg.norm(exact-np.array([row['x1'],row['x2']]))),
                             g1=row['g1'],lambda_est=row['lambda1']))
    write_csv('analytic_check.csv',analytic)
    x=run['x']; xp=x-cons1[0](x)*normal/(normal@normal)
    print('Проєкція на x1=3: x_hat=',xp,' F=',F(xp),' g1=',cons1[0](xp))
    write_csv('projection.csv',[dict(before_x1=x[0],before_x2=x[1],before_F=F(x),before_g=cons1[0](x),
                                     after_x1=xp[0],after_x2=xp[1],after_F=F(xp),after_g=cons1[0](xp))])
    p,g=penalty([cons1],1e6);pc,gc=Count(p),Count(g)
    xd,n,trace,ok=fr(pc,gc,X0,tol=.01,maxit=1500)
    print('Одразу r=10^6: x=',xd,' FR=',n,' NΦ=',pc.n,' N∇Φ=',gc.n,' converged=',ok)
    write_csv('direct_r1e6_FR.csv',trace)
    X,Y=np.meshgrid(np.linspace(-1,7,320),np.linspace(-7,2,320))
    Z=X*X+5*Y*Y-2*X*Y-3*X+11*Y
    fig,ax=plt.subplots(figsize=(9,6))
    ax.contour(X,Y,Z,levels=F(XFREE)+np.geomspace(.2,200,18),alpha=.7)
    ax.axvspan(3,7,alpha=.12,label='Ω: x₁ ≥ 3')
    ax.axvline(3,linewidth=2,label='Межа x₁=3')
    path=run['path'];ax.plot(path[:,0],path[:,1],'-o',label='x(r)')
    ax.scatter(*XFREE,marker='x',s=110,label='Безумовний мінімум')
    ax.scatter(*xstar,marker='*',s=160,label='Умовний мінімум')
    ax.set(xlabel='x₁',ylabel='x₂',title='ЛР №11А — зовнішня штрафна функція')
    ax.grid(alpha=.2);ax.legend(fontsize=9);fig.tight_layout()
    fig.savefig(OUT/'contours.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    ax.loglog([row['r'] for row in run['rows']],[max(1e-15,row['violation']) for row in run['rows']],'-o')
    ax.grid(alpha=.3,which='both');ax.set(xlabel='r',ylabel='max(0, g₁)',title='Порушення обмеження')
    fig.tight_layout();fig.savefig(OUT/'violation_log.png',dpi=170);plt.close(fig)
    print('Таблиці й графіки:',OUT)
if __name__=='__main__':main()
