from pathlib import Path
import csv
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent/'lab12_results'
OUT.mkdir(exist_ok=True)
H=np.array([[2.,-2.],[-2.,10.]])
X0=np.array([5.,-5.]); X_INSIDE=np.array([4.,-9.]); X_CORNER=np.array([3.,-4.])
A1=np.array([[-1.,0.]])
b1=np.array([-3.])
A12=np.array([[-1.,0.],[0.,1.]])
b12=np.array([-3.,-1.])
A13=np.array([[-1.,0.],[0.,-1.]])
b13=np.array([-3.,4.])

def F(x):
    a,b=np.asarray(x)
    return float(a*a+5*b*b-2*a*b-3*a+11*b)

def gradF(x):
    a,b=np.asarray(x)
    return np.array([2*a-2*b-3,10*b-2*a+11],float)

class Counter:
    def __init__(self,func): self.func,self.n=func,0
    def __call__(self,x):
        self.n+=1
        return self.func(x)

def savecsv(name,rows):
    if not rows: return
    cols=list(dict.fromkeys(k for r in rows for k in r.keys()))
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=cols,delimiter=';')
        w.writeheader();w.writerows(rows)

def project_start(x,A,b,tol=1e-10):
    x=np.asarray(x,float).copy()
    logs=[]
    for it in range(100):
        g=A@x-b
        j=int(np.argmax(g))
        if g[j]<=tol: return x,logs
        old=x.copy()
        x=x-g[j]/(A[j]@A[j])*A[j]
        logs.append({'step':it,'wall':j+1,'old_x1':old[0],'old_x2':old[1],
                     'violation':g[j],'new_x1':x[0],'new_x2':x[1]})
    raise RuntimeError('Не вдалося отримати допустимий старт')

def golden(f,left,right,tol=1e-6):
    tau=(math.sqrt(5)-1)/2
    u=right-tau*(right-left);v=left+tau*(right-left)
    fu=f(u);fv=f(v)
    for _ in range(200):
        if right-left<=tol:break
        if fu<=fv:
            right,v,fv=v,u,fu
            u=right-tau*(right-left);fu=f(u)
        else:
            left,u,fu=u,v,fv
            v=left+tau*(right-left);fv=f(v)
    return .5*(left+right)

def bounded_line_search(Fcnt,grad,x,d,hmax,delta=.01,tol=1e-6):
    phi=lambda h: Fcnt(x+h*d) if h>=0 and h<=hmax else math.inf
    if hmax<=1e-14:return 0.,0,0,True
    left=0.;current=min(delta,hmax);f0=phi(0.);fc=phi(current)
    n_svenn=2
    if fc>=f0:
        right=current
    else:
        prev=0.;step=delta;right=current
        for _ in range(80):
            nxt=current+2*step
            if math.isfinite(hmax):nxt=min(nxt,hmax)
            fn=phi(nxt);n_svenn+=1
            right=nxt
            if fn>=fc or nxt>=hmax:break
            prev,current,fc=current,nxt,fn
            step*=2
        left=prev
    if right-left<=tol:
        candidate=(right+left)/2
    else:
        candidate=golden(phi,left,right,tol)
    if math.isfinite(hmax) and (phi(hmax)<=phi(candidate) or grad(x+hmax*d)@d<0):
        return float(hmax),n_svenn,0,True
    return float(candidate),n_svenn,0,False

def rosen(A,b,x0,eps=.01,maxiter=200):
    A=np.asarray(A,float);b=np.asarray(b,float)
    fc=Counter(F);gc=Counter(gradF)
    x,projection=project_start(x0,A,b)
    active=[j for j in range(len(b)) if abs(A[j]@x-b[j])<1e-9]
    traj=[x.copy()]
    logs=[]; events=[]
    for k in range(maxiter+1):
        gr=gc(x)
        if active:
            AI=A[active]
            normal_proj=AI.T@np.linalg.solve(AI@AI.T,AI)
            P=np.eye(2)-normal_proj
        else:P=np.eye(2)
        d=-P@gr
        n=float(np.linalg.norm(d))
        row={'k':k,'x1':x[0],'x2':x[1],'I':str([j+1 for j in active]),
             'd1':d[0],'d2':d[1],'norm_d':n,'hmax':'','h':'','F':fc(x),
             'NF':fc.n,'NgradF':gc.n,'event':''}
        if n<=eps:
            u= -np.linalg.solve(A[active]@A[active].T,A[active]@gr) if active else np.array([])
            if len(u)==0 or np.min(u)>=-1e-9:
                row['event']='ККТ: стоп';logs.append(row)
                lam=np.zeros(len(b));
                for idx,value in zip(active,u):lam[idx]=value
                return {'x':x,'F':F(x),'lambda':lam,'trajectory':np.array(traj),
                        'logs':logs,'projection':projection,'events':events,
                        'NF':fc.n,'NgradF':gc.n,'steps':len(traj)-1,
                        'violation':max(0.,float(np.max(A@x-b))), 'converged':True,
                        'KKT_residual':float(np.linalg.norm(gr+A.T@lam))}
            rem=int(np.argmin(u));bad=active.pop(rem)
            row['event']=f'Відпустили стіну {bad+1}: u={u[rem]:.8f}'
            events.append(row['event']);logs.append(row)
            continue
        hmax=math.inf;jhit=None
        for j in range(len(b)):
            if j in active:continue
            vd=float(A[j]@d)
            if vd>1e-12:
                h=float((b[j]-A[j]@x)/vd)
                if h<hmax: hmax=max(0.,h);jhit=j
        h,ns,ng,hit=bounded_line_search(fc,gc,x,d,hmax)
        row['hmax']=hmax;row['h']=h
        old=x.copy()
        x=x+h*d
        if hit and jhit is not None:
            x=x-(A[jhit]@x-b[jhit])/(A[jhit]@A[jhit])*A[jhit]
            active.append(jhit)
            row['event']=f'Паркан: стіна {jhit+1} стала активною'
            events.append(row['event'])
        if np.max(A@x-b)>1e-8:raise RuntimeError('Порушено допустимість')
        logs.append(row);traj.append(x.copy())
        if np.linalg.norm(x-old)<1e-12:raise RuntimeError('Нульовий крок: перевірити активну множину')
    raise RuntimeError('Ліміт ітерацій методу Розена')

def print_result(label,r):
    print(f'\n{label}: кроків={r["steps"]}, NF={r["NF"]}, N∇F={r["NgradF"]}')
    print(f'x=({r["x"][0]:.9f};{r["x"][1]:.9f}), F={r["F"]:.9f},'
          f' λ={np.round(r["lambda"],7)}, порушення={r["violation"]:.2g}')
    for row in r['logs']:
        print(f'  k={row["k"]}, x=({row["x1"]:.6f};{row["x2"]:.6f}),'
              f' I={row["I"]}, d=({row["d1"]:.6f};{row["d2"]:.6f}),'
              f' hmax={row["hmax"]}, h={row["h"]}, {row["event"]}')

def circle_run(start,h=.1,eps=.01,maxiter=2000):
    x=np.array(start,float)
    path=[x.copy()];cnt=Counter(gradF)
    for k in range(1,maxiter+1):
        y=x-h*cnt(x)
        norm=float(np.linalg.norm(y))
        if norm<3: y=y*(3/norm) if norm else np.array([3.,0.])
        err=float(np.linalg.norm(y-x))
        x=y
        path.append(x.copy())
        if err<=eps:break
    return {'x':x,'F':F(x),'k':k,'NgradF':cnt.n,'trajectory':np.array(path),
            'violation':max(0.,9-float(x@x)),'step_norm':err}

def compare_LR11():
    out=[]
    for label,path,defaults in [
        ('ЛР11А, 1 стіна',Path(__file__).resolve().parent/'lab11_A_results'/'outer_protocol.csv',
         {'iterations':4,'NF':402,'NgradF':17,'x1':2.9979996643,'x2':-.5003993872,'F':-1.25799814,'violation':.002000336,'lambda':'4.00067'}),
        ('ЛР11Б, 2 стіни',Path(__file__).resolve().parent/'lab11_B_results'/'two_constraints.csv',
         {'iterations':4,'NF':267,'NgradF':13,'x1':2.9975049810,'x2':-.9975149490,'F':-.0248508467,'violation':.002495019,'lambda':'(4.99004;4.97010)'})]:
        record=dict(defaults)
        if path.exists():
            with path.open(encoding='utf-8-sig',newline='') as f:
                data=list(csv.DictReader(f,delimiter=';'))
            if data:
                last=data[-1]
                record.update({'iterations':int(last['k']), 'NF':int(last['NPhi_total']),
                 'NgradF':int(last['NgradPhi_total']), 'x1':float(last['x1']),
                 'x2':float(last['x2']),'F':float(last['F']),
                 'violation':float(last['violation']),'lambda':last['lambdas']})
        out.append({'method':label,**record})
    return out

def plot_paths(single,twostart,inside,corner,penalty_rows):
    x=np.linspace(-1,7,320);y=np.linspace(-11,1.8,320)
    xx,yy=np.meshgrid(x,y)
    z=xx**2+5*yy**2-2*xx*yy-3*xx+11*yy
    fig,ax=plt.subplots(figsize=(10,7))
    levels=-6.25+np.geomspace(.2,350,20)
    ax.contour(xx,yy,z,levels=levels,alpha=.45,linewidths=.7)
    ax.axvspan(3,7,color='forestgreen',alpha=.07,label='Ω: x₁ ≥ 3')
    ax.axvline(3,linewidth=2,color='black',label='Стіна 1: x₁=3')
    ax.plot(*single['trajectory'].T,'o-',label='Розен: 1 стіна, старт (5;−5)')
    ax.scatter([3],[-.5],marker='*',s=120,zorder=5,label='x*=(3;−0,5)')
    if penalty_rows:
        pts=np.array([[float(r['x1']),float(r['x2'])] for r in penalty_rows])
        ax.plot(pts[:,0],pts[:,1],'x--',label='ЛР11А: x(r)')
    ax.set(xlabel='x₁',ylabel='x₂',title='Варіант 6 — одна стіна, Розен і ЛР11')
    ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(OUT/'one_wall_vs_LR11.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,7))
    ax.contour(xx,yy,z,levels=levels,alpha=.45,linewidths=.7)
    ax.contourf(xx,yy,((xx>=3)&(yy<=-1)).astype(float),levels=[.5,1.5],alpha=.1)
    ax.axvline(3,color='black',label='Стіна 1: x₁=3')
    ax.axhline(-1,color='gray',label='Стіна 2: x₂=−1')
    for label,r in [('Старт (5;−5)',twostart),('Старт (4;−9)',inside)]:
        path=r['trajectory'];ax.plot(path[:,0],path[:,1],'-o',label=label)
    ax.scatter([3],[-1],s=140,marker='*',label='x*=(3;−1)')
    p=Path(__file__).resolve().parent/'lab11_B_results'/'two_constraints.csv'
    if p.exists():
        with p.open(encoding='utf-8-sig',newline='') as f:records=list(csv.DictReader(f,delimiter=';'))
        pts=np.array([[float(r['x1']),float(r['x2'])] for r in records]);ax.plot(pts[:,0],pts[:,1],'x--',label='ЛР11Б: x(r)')
    ax.set(xlabel='x₁',ylabel='x₂',title='Варіант 6 — дві стіни й два старти')
    ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(OUT/'two_walls_two_starts.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6))
    ax.contour(xx,yy,z,levels=levels,alpha=.45,linewidths=.7)
    ax.axvline(3,color='black',label='Стіна 1: x₁=3')
    ax.axhline(-4,color='darkred',label='Стіна 3: x₂=−4')
    path=corner['trajectory'];ax.plot(path[:,0],path[:,1],'-o',label='Відпускання стіни 3')
    ax.set(xlabel='x₁',ylabel='x₂',title='Варіант 6 — відпускання зайвої стіни')
    ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'release_wall.png',dpi=180);plt.close(fig)

def plot_circle(runs):
    x=np.linspace(-5,5,320);y=np.linspace(-5,5,320)
    xx,yy=np.meshgrid(x,y)
    z=xx**2+5*yy**2-2*xx*yy-3*xx+11*yy
    fig,ax=plt.subplots(figsize=(8,7))
    ax.contour(xx,yy,z,levels=25,alpha=.45)
    t=np.linspace(0,2*np.pi,500)
    ax.plot(3*np.cos(t),3*np.sin(t),color='black',linewidth=2,label='Коло R=3')
    for label,r in runs.items():
        p=r['trajectory'];ax.plot(p[:,0],p[:,1],'-o',markersize=2,label=label)
        ax.scatter(r['x'][0],r['x'][1],s=85,marker='*')
    ax.set(xlabel='x₁',ylabel='x₂',title='Проєкція точки: поза колом, h=0,1',aspect='equal')
    ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'circle_projected_gradient.png',dpi=180);plt.close(fig)

def main():
    print('ЛР 12. Метод проєкції градієнта (Розен). Варіант 6')
    print('F=x1²+5x2²−2x1*x2−3x1+11x2, ε=0,01')
    r1=rosen(A1,b1,X0)
    r2=rosen(A12,b12,X0)
    r2ins=rosen(A12,b12,X_INSIDE)
    r3=rosen(A13,b13,X_CORNER)
    runs={'Прогін 1: x1≥3':r1,'Прогін 2: x1≥3,x2≤−1':r2,
          'Другий старт (4;−9)':r2ins,'Прогін 3: x1≥3,x2≥−4':r3}
    for label,r in runs.items():
        print_result(label,r)
        slug={'Прогін 1: x1≥3':'run1','Прогін 2: x1≥3,x2≤−1':'run2',
              'Другий старт (4;−9)':'run2_inside','Прогін 3: x1≥3,x2≥−4':'run3_release'}[label]
        savecsv(slug+'_protocol.csv',r['logs'])
        savecsv(slug+'_initial_projection.csv',r['projection'])
    prior=compare_LR11()
    summary=prior+[{'method':'Розен, 1 стіна','iterations':r1['steps'],'NF':r1['NF'],
                'NgradF':r1['NgradF'],'x1':r1['x'][0],'x2':r1['x'][1],
                'F':r1['F'],'violation':r1['violation'],'lambda':str(r1['lambda'])},
               {'method':'Розен, 2 стіни','iterations':r2['steps'],'NF':r2['NF'],
                'NgradF':r2['NgradF'],'x1':r2['x'][0],'x2':r2['x'][1],
                'F':r2['F'],'violation':r2['violation'],'lambda':str(r2['lambda'])},
               {'method':'Розен, 2 стіни, другий старт','iterations':r2ins['steps'],
                'NF':r2ins['NF'],'NgradF':r2ins['NgradF'],
                'x1':r2ins['x'][0],'x2':r2ins['x'][1],
                'F':r2ins['F'],'violation':r2ins['violation'],'lambda':str(r2ins['lambda'])},
               {'method':'Розен, відпускання стіни 3','iterations':r3['steps'],
                'NF':r3['NF'],'NgradF':r3['NgradF'],
                'x1':r3['x'][0],'x2':r3['x'][1],
                'F':r3['F'],'violation':r3['violation'],'lambda':str(r3['lambda'])}]
    savecsv('LR11_vs_LR12.csv',summary)
    out11=Path(__file__).resolve().parent/'lab11_A_results'/'outer_protocol.csv'
    rows=[]
    if out11.exists():
        with out11.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f,delimiter=';'))
    plot_paths(r1,r2,r2ins,r3,rows)
    circ={'Старт (5;−5)':circle_run((5.,-5.)),
          'Старт (−5;3)':circle_run((-5.,3.))}
    for label,r in circ.items():
        print(f'Коло, {label}: x={r["x"]}, F={r["F"]:.8f},'
              f' ітерацій={r["k"]}, N∇F={r["NgradF"]}, порушення={r["violation"]}')
    savecsv('circle_protocol.csv',[{'start':lab,'x1':r['x'][0],
      'x2':r['x'][1],'F':r['F'],'iterations':r['k'],
      'NgradF':r['NgradF'],'violation':r['violation']}
      for lab,r in circ.items()])
    plot_circle(circ)
    print('\nРезультати у:',OUT)

if __name__=='__main__': main()
