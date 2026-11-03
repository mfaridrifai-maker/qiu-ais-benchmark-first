"""95% confidence intervals for the main statistics (Wilson for proportions; percentile bootstrap otherwise).
Note: Cramer's V uses the grouped categories in the shared file; the manuscript used the original finer categories, so V values can differ slightly.
Usage: python confidence_intervals.py"""
import numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
rng=np.random.default_rng(42); B=1000
df=pd.read_csv('qiu_ais_survey_anonymized.csv'); items=list(df.columns[1:23])
LAB=["EF","EF","EF","EC","EC","EC","ST","CC","EF","EC","ST","ST","CC","EF","EC","EC","ST","CC","EF","ST","EC","CC"]
X=pd.DataFrame({k:df[[i for i,l in zip(items,LAB) if l==k]].mean(axis=1) for k in ["EF","EC","ST","CC"]}); S=X.mean(axis=1).values
Z=((X-X.min())/(X.max()-X.min())).values
def lab_of(Z,S):
    l=KMeans(3,n_init=10,random_state=42).fit_predict(Z); o=np.argsort([S[l==i].mean() for i in range(3)]); r={x:i for i,x in enumerate(o)}; return np.array([r[i] for i in l])
lab=lab_of(Z,S); n=314
wil=lambda k,n_:stats.binomtest(int(round(k)),n_).proportion_ci(method='wilson')
print('A. group proportions (Wilson)')
for g,nm in enumerate(['Less','Moderately','Highly']):
    k=(lab==g).sum(); c=wil(k,n); print(nm,k,'%.1f%% (%.1f-%.1f)'%(k/n*100,c.low*100,c.high*100))
# B. silhouette bootstrap with refit
print('B. silhouette bootstrap (refit each resample)')
sk=[];st=[];d=[]
for b in range(B):
    i=rng.integers(0,n,n); Xs=X.values[i]; Ss=Xs.mean(1); Zs=(Xs-Xs.min(0))/(Xs.max(0)-Xs.min(0))
    l=KMeans(3,n_init=5,random_state=b).fit_predict(Zs); s1=silhouette_score(Zs,l)
    q=(stats.rankdata(Ss,method='ordinal')-1)*3//n; s2=silhouette_score(Zs,q)
    sk.append(s1);st.append(s2);d.append(s1-s2)
p=lambda a:np.percentile(a,[2.5,97.5]).round(3)
print('K-Means Silh',round(silhouette_score(Z,lab),3),p(sk),'| tertile',round(silhouette_score(Z,(stats.rankdata(S,method='ordinal')-1)*3//n),3),p(st),'| diff CI',p(d))
# C/D module differences & Wilcoxon effect size
MOD={'Course Registration':[f'REG{i}' for i in range(1,9)],'Grade Viewing':[f'GRD{i}' for i in range(1,6)],'Class Schedule':[f'SCH{i}' for i in range(1,6)],'Payment':[f'PAY{i}' for i in range(1,5)]}
M=pd.DataFrame({k:df[v].mean(axis=1) for k,v in MOD.items()})
def rb(x,y):
    dd=x-y; dd=dd[dd!=0]; r=stats.rankdata(np.abs(dd)); wp=r[dd>0].sum(); wm=r[dd<0].sum(); return (wp-wm)/(wp+wm)
print('C/D module Grade-Payment, bootstrap 95% CI (paired); Wilcoxon rank-biserial r and r=Z/sqrt(N)')
for g,nm in enumerate(['Less','Moderately','Highly']):
    x=M.loc[lab==g,'Grade Viewing'].values; y=M.loc[lab==g,'Payment'].values; m=len(x)
    md=(x-y); bs=[];br=[]
    for b in range(5000):
        i=rng.integers(0,m,m); bs.append(md[i].mean()); br.append(rb(x[i],y[i]))
    w=stats.wilcoxon(x,y); zz=stats.norm.isf(w.pvalue/2); print(nm,m,'diff %.2f'%md.mean(),np.percentile(bs,[2.5,97.5]).round(2),'p=%.1e'%w.pvalue,'rb=%.2f'%rb(x,y),np.percentile(br,[2.5,97.5]).round(2),'r=Z/sqrt(N)=%.2f'%(zz/np.sqrt(m)))
# gap difference between groups: (Payment-Grade) Less minus Highly
dl=(M.loc[lab==0,'Payment']-M.loc[lab==0,'Grade Viewing']).values; dh=(M.loc[lab==2,'Payment']-M.loc[lab==2,'Grade Viewing']).values
bs=[rng.choice(dl,len(dl)).mean()-rng.choice(dh,len(dh)).mean() for _ in range(5000)]; print('gap widening Less minus Highly: %.2f'%(dl.mean()-dh.mean()),np.percentile(bs,[2.5,97.5]).round(2))
# module means per group CIs
print('module means by group, bootstrap 95% CI')
for m_ in MOD:
    row=[]
    for g in range(3):
        v=M.loc[lab==g,m_].values; bs=[rng.choice(v,len(v)).mean() for _ in range(2000)]; row.append('%.2f [%.2f, %.2f]'%(v.mean(),*np.percentile(bs,[2.5,97.5])))
    print(m_,row)
# E Cramer's V with finer categories (clean file)
ext={'study program (grouped)':df.program_group.values,'semester (grouped)':df.semester_group.values,'gender':df.gender.values,'usage frequency':df.use_frequency.values,'problem: confusing':df.problem_confusing.values,'problem: errors/crashes':df.problem_errors_crash.values}
def V(a,b):
    t=pd.crosstab(a,b); chi2=stats.chi2_contingency(t,correction=False)[0]; N=t.values.sum(); r,k=t.shape
    phi2=max(0,chi2/N-(k-1)*(r-1)/(N-1)); rc=r-(r-1)**2/(N-1); kc=k-(k-1)**2/(N-1); return np.sqrt(phi2/max(1e-12,min(kc-1,rc-1)))
print('E Cramers V bias-corrected, bootstrap 95% CI')
for nm,a in ext.items():
    v=V(a,lab); bs=[]
    for b in range(2000):
        i=rng.integers(0,n,n); bs.append(V(a[i],lab[i]))
    print(nm,'V=%.2f'%v,np.percentile(bs,[2.5,97.5]).round(2),'p=%.3f'%stats.chi2_contingency(pd.crosstab(a,lab))[1])
# F sim rates Wilson
print('F sim rates Wilson (rate -> CI)')
T4={'U':(94.0,79.7),'G':(99.7,98.0),'M0.2':(87.3,50.7),'M0.4':(76.3,25.7),'M0.6':(11.0,4.0),'M0.8':(0.0,0.0),'M1.4':(0.0,0.0)}
for k,(a,b) in T4.items():
    r=[]
    for v in (a,b):
        c=wil(v*3,300); r.append('%.1f (%.1f-%.1f)'%(v,c.low*100,c.high*100))
    print(k,r)
T5={'U':[6.7,0.0,5.0,1.7,0.0],'G':[0.0,0.0,50.0,100.0,85.0],'M0.4':[26.7,0.0,11.7,1.7,0.0],'M0.6':[90.0,6.7,1.7,1.7,0.0],'M0.8':[100.0,100.0,0.0,1.7,0.0]}
for k,v in T5.items():
    print(k,['%.1f (%.1f-%.1f)'%(x,wil(x*0.6,60).low*100,wil(x*0.6,60).high*100) for x in v])
for nm,v,nn in [('N150 U',89.0,300),('N150 G',99.0,300),('N600 U',97.7,300),('N600 G',100,300)]:
    c=wil(v*3,nn); print(nm,v,'(%.1f-%.1f)'%(c.low*100,c.high*100))
