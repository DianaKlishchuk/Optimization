from fractions import Fraction as Q
from pathlib import Path
import csv
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent/'lab13_results'
OUT.mkdir(exist_ok=True)

def qstr(v): return str(v)
def dot(v,w): return sum((a*b for a,b in zip(v,w)),Q(0))

def vertices(A,b):
    import itertools
    walls=[(Q(1),Q(0),Q(0)),(Q(0),Q(1),Q(0))]
    walls += [(Q(a),Q(c),Q(t)) for (a,c),t in zip(A,b)]
    verts=set()
    for (a,c,t),(d,e,f) in itertools.combinations(walls,2):
        det=a*e-c*d
        if det==0: continue
        x,y=(t*e-c*f)/det,(a*f-t*d)/det
        if x>=0 and y>=0 and all(v*x+w*y<=z for v,w,z in walls[2:]): verts.add((x,y))
    return sorted(verts,key=lambda p:math.atan2(float(p[1])-sum(float(z[1]) for z in verts)/len(verts),float(p[0])-sum(float(z[0]) for z in verts)/len(verts)))

class Tableau:
    def __init__(self, rows, basis, names, costs):
        self.T=[[Q(t) for t in row] for row in rows]
        self.basis=list(basis); self.names=list(names)
        self.set_objective(costs)
    def set_objective(self,costs):
        self.costs=list(map(Q,costs))
        z=[-v for v in self.costs]+[Q(0)]
        for i,j in enumerate(self.basis):
            z=[z[k]+self.costs[j]*self.T[i][k] for k in range(len(z))]
        self.T=self.T[:len(self.basis)]+[z]
    def values(self):
        v=[Q(0)]*len(self.names)
        for i,j in enumerate(self.basis): v[j]=self.T[i][-1]
        return v
    def snapshot(self,phase,rule,k):
        return {'phase':phase,'rule':rule,'k':k,'basis':[self.names[j] for j in self.basis],
                'table':[[qstr(x) for x in r] for r in self.T],
                'variables':[qstr(x) for x in self.values()], 'objective':qstr(self.T[-1][-1])}
    def pivot(self,r,c):
        p=self.T[r][c]
        if p==0: raise ZeroDivisionError('Нульовий ведучий елемент')
        self.T[r]=[a/p for a in self.T[r]]
        for i in range(len(self.T)):
            if i!=r:
                factor=self.T[i][c]
                self.T[i]=[a-factor*v for a,v in zip(self.T[i],self.T[r])]
        self.basis[r]=c
        assert all(self.T[i][-1]>=0 for i in range(len(self.basis))), 'Порушена допустимість базису'
        for i,j in enumerate(self.basis):
            assert all(self.T[t][j]==(Q(1) if i==t else Q(0)) for t in range(len(self.basis)))
    def solve(self,rule='dantzig',phase='I',limit=100):
        records=[]; actions=[]; k=0
        while True:
            snap=self.snapshot(phase,rule,k); records.append(snap)
            print(f'\n{phase}, {rule}, таблиця {k}; базис {snap["basis"]}; ціль = {snap["objective"]}')
            print('      '+' | '.join(self.names+['b']))
            for name,row in zip(snap['basis']+['Z'],snap['table']):
                print(f'{name:>5}: '+' | '.join(row))
            neg=[j for j,v in enumerate(self.T[-1][:-1]) if v<0]
            if not neg: return records,actions,'optimal'
            col=min(neg,key=lambda j:(self.T[-1][j],j)) if rule=='dantzig' else min(neg)
            choices=[(self.T[i][-1]/self.T[i][col], self.basis[i], i) for i in range(len(self.basis)) if self.T[i][col]>0]
            if not choices:
                print('НЕОБМЕЖЕНІСТЬ: ведучий стовпець без додатних елементів')
                return records,actions,'unbounded'
            ratio,_,row=min(choices)
            action={'phase':phase,'rule':rule,'k':k,'enter':self.names[col],
                    'leave':self.names[self.basis[row]],'pivot':qstr(self.T[row][col]),'ratio':qstr(ratio)}
            actions.append(action)
            print('Вводимо',action['enter'],'виводимо',action['leave'],'відношення',action['ratio'],'ведучий',action['pivot'])
            self.pivot(row,col)
            k+=1
            if k>=limit: raise RuntimeError('Досягнуто ліміту перерахунків')

def standard_part1(rule):
    A=[[1,0],[0,1],[1,2]]; b=[7,5,15]; c=[8,6]; m=3;n=2
    rows=[[Q(x) for x in A[i]]+[Q(int(i==j)) for j in range(m)]+[Q(b[i])] for i in range(m)]
    obj=Tableau(rows,[2,3,4],['x1','x2','s1','s2','s3'],[8,6,0,0,0])
    snapshots,actions,status=obj.solve(rule,phase='Частина 1')
    return obj,snapshots,actions,status

def two_phase_part2():
    names=['x1','x2','s1','s2','s3','a1']
    rows=[[2,3,-1,0,0,1,18], [1,0,0,1,0,0,6], [0,1,0,0,1,0,4]]
    t=Tableau(rows,[5,3,4],names,[0,0,0,0,0,-1])
    r1,a1,status=t.solve('dantzig',phase='Фаза 1')
    assert status=='optimal'
    if t.T[-1][-1]!=0: raise ValueError('Область допустимих розв’язків порожня')
    art=5
    if art in t.basis:
        i=t.basis.index(art)
        candidates=[j for j in range(art) if t.T[i][j]!=0 and j not in t.basis]
        if candidates: t.pivot(i,min(candidates))
        else:
            if t.T[i][-1]!=0: raise ValueError('Штучна змінна лишилася додатною')
            t.T.pop(i);t.basis.pop(i)
    for row in t.T: row.pop(art)
    t.names.pop(art)
    t.set_objective([-1,-6,0,0,0])
    r2,a2,status=t.solve('dantzig',phase='Фаза 2')
    assert status=='optimal'
    v=t.values()
    assert v[0]==6 and v[1]==2 and -t.T[-1][-1]==18
    return t,r1,a1,r2,a2

def csv_rows(filename,items):
    if not items: return
    with (OUT/filename).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(items[0]),delimiter=';')
        w.writeheader();w.writerows(items)

def save_tables(name,records,actions,names):
    path=OUT/name
    with path.open('w',encoding='utf-8') as f:
        for r in records:
            f.write(f'Фаза: {r["phase"]}, правило: {r["rule"]}, таблиця {r["k"]}\n')
            f.write('Базис | '+' | '.join(names)+' | b\n')
            for label,row in zip(r['basis']+['Z'],r['table']):f.write(label+' | '+' | '.join(row)+'\n')
            f.write('Значення цільової функції таблиці = '+r['objective']+'\n')
            if r['k']<len(actions):
                a=actions[r['k']];f.write('Входить '+a['enter']+', виходить '+a['leave']+', ведучий '+a['pivot']+', відношення '+a['ratio']+'\n')
            f.write('\n')

def save_plot_1(recD,recB):
    v=vertices([[1,0],[0,1],[1,2]],[7,5,15]); poly=np.array([(float(a),float(b)) for a,b in v])
    fig,ax=plt.subplots(figsize=(9,7))
    ax.fill(poly[:,0],poly[:,1],alpha=.18,label='Допустима область Ω')
    ax.plot(np.r_[poly[:,0],poly[0,0]],np.r_[poly[:,1],poly[0,1]],lw=2)
    xx=np.linspace(0,8,250)
    for level in [0,30,56,70,80]: ax.plot(xx,(level-8*xx)/6,ls=':',lw=1,alpha=.6)
    for label,records in [('Данциг',recD),('Бленд',recB)]:
        pts=np.array([[float(Q(r['variables'][0])),float(Q(r['variables'][1]))] for r in records])
        ax.plot(pts[:,0],pts[:,1],'-o',lw=2,label=label)
    for x,y in v:ax.annotate(f'({x}; {y})\nZ={8*x+6*y}',(float(x),float(y)),xytext=(7,7),textcoords='offset points',fontsize=9)
    ax.scatter([7],[4],s=170,marker='*',label='Оптимум (7;4)',zorder=6)
    ax.set(xlim=(-.7,9),ylim=(-1,7),xlabel='x₁',ylabel='x₂',title='ЛР 13, варіант 6: вершини, Данциг і Бленд')
    ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'part1_simplex_paths.png',dpi=180);plt.close(fig)

def save_plot_2():
    v=vertices([[-2,-3],[1,0],[0,1]],[-18,6,4]);poly=np.array([(float(x),float(y)) for x,y in v])
    fig,ax=plt.subplots(figsize=(8,6))
    ax.fill(poly[:,0],poly[:,1],alpha=.22,label='Допустима область Ω')
    ax.plot(np.r_[poly[:,0],poly[0,0]],np.r_[poly[:,1],poly[0,1]],lw=2)
    xx=np.linspace(0,7.2,250)
    for level in [18,27,30,36]:ax.plot(xx,(level-xx)/6,ls=':',alpha=.5,label=f'W = {level}')
    for x,y in v:ax.annotate(f'({x}; {y}) W={x+6*y}',(float(x),float(y)),xytext=(5,7),textcoords='offset points')
    ax.scatter([6],[2],s=150,marker='*',label='Мінімум (6;2)',zorder=5)
    ax.set(xlim=(-.5,7.5),ylim=(-.5,5.5),xlabel='x₁',ylabel='x₂',title='Частина 2: двофазний симплекс, W → min')
    ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'part2_feasible_region.png',dpi=180);plt.close(fig)


def rosen_linear_optional():
    A=np.array([[1.,0.],[0.,1.],[1.,2.],[-1.,0.],[0.,-1.]])
    b=np.array([7.,5.,15.,0.,0.])
    x=np.array([0.,0.]); grad=np.array([-8.,-6.]); active=[3,4]
    result=[]; count=0
    for k in range(30):
        AI=A[active] if active else np.empty((0,2))
        if len(active):
            P=np.eye(2)-AI.T@np.linalg.solve(AI@AI.T,AI)
        else:P=np.eye(2)
        d=-P@grad;count+=1
        if np.linalg.norm(d)<1e-10:
            u=-np.linalg.solve(AI@AI.T,AI@grad) if len(active) else np.array([])
            result.append({'k':k,'x1':x[0],'x2':x[1],'Z':8*x[0]+6*x[1],
                           'active':str(active),'d':str(tuple(np.round(d,8))),
                           'hmax':'','lambda':str(tuple(np.round(u,8))),'action':'stop' if np.all(u>=-1e-9) else 'release'})
            if not len(u) or np.all(u>=-1e-9):break
            active.pop(int(np.argmin(u)))
            continue
        hmax=float('inf');hit=None
        for j in range(len(A)):
            if j in active:continue
            ad=float(A[j]@d)
            if ad>1e-10:
                h=(b[j]-A[j]@x)/ad
                if h<hmax: hmax,hit=h,j
        if not math.isfinite(hmax):raise RuntimeError('Розен: лінійна задача необмежена')
        result.append({'k':k,'x1':x[0],'x2':x[1],'Z':8*x[0]+6*x[1],
                       'active':str(active),'d':str(tuple(np.round(d,8))),
                       'hmax':hmax,'lambda':'','action':'step to wall '+str(hit)})
        x=x+hmax*d
        if hit is not None and hit not in active:active.append(hit)
    assert np.allclose(x,[7.,4.]),x
    return result

def main():
    print('ЛР №13 · варіант 6 · точні дроби')
    D,drec,dact,ds=standard_part1('dantzig')
    B,brec,bact,bs=standard_part1('bland')
    assert ds=='optimal' and bs=='optimal'
    assert D.values()[:2]==[7,4] and B.values()[:2]==[7,4]
    assert D.T[-1][-1]==B.T[-1][-1]==80
    dual=D.T[-1][2:5]
    assert dual==[Q(5),Q(0),Q(3)]
    assert dot(dual,[7,5,15])==80
    assert [dual[0]+dual[2],dual[1]+2*dual[2]]==[8,6]
    print('\nЧастина 1: x* = (7;4), Z*=80; тіньові ціни:',tuple(map(str,dual)))
    print('Двоїста: min 7y1+5y2+15y3; y1+y3>=8; y2+2y3>=6; y>=0')
    print('W_dual = ',dot(dual,[7,5,15]))
    for name,rec,act in [('dantzig',drec,dact),('bland',brec,bact)]:
        save_tables('part1_'+name+'_tables.txt',rec,act,D.names)
        csv_rows('part1_'+name+'_pivots.csv',act)
    t,r1,a1,r2,a2=two_phase_part2()
    save_tables('part2_phase1_tables.txt',r1,a1,['x1','x2','s1','s2','s3','a1'])
    save_tables('part2_phase2_tables.txt',r2,a2,['x1','x2','s1','s2','s3'])
    csv_rows('part2_phase1_pivots.csv',a1)
    csv_rows('part2_phase2_pivots.csv',a2)
    print('\nЧастина 2: x*=(6;2), W*=18, фаза 1 переходів',len(a1),'фаза 2',len(a2))
    v1=vertices([[1,0],[0,1],[1,2]],[7,5,15])
    v2=vertices([[-2,-3],[1,0],[0,1]],[-18,6,4])
    csv_rows('part1_vertices.csv',[{'x1':str(x),'x2':str(y),'Z':str(8*x+6*y)} for x,y in v1])
    csv_rows('part2_vertices.csv',[{'x1':str(x),'x2':str(y),'W':str(x+6*y)} for x,y in v2])
    save_plot_1(drec,brec);save_plot_2()
    rose=rosen_linear_optional();csv_rows('optional_rosen.csv',rose)
    print('Розен: кінцева точка (7;4), записи протоколу',len(rose))
    print('\nЗбережено файли в:',OUT)
    print('Таблиць: Данциг',len(drec),'Бленд',len(brec),'фаза 1',len(r1),'фаза 2',len(r2))

if __name__=='__main__':main()
