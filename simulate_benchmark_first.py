#!/usr/bin/env python3
"""Simulation study for the benchmark-first decision rule (companion to the QiU manuscript).
For synthetic 22-item Likert data (items written for EF/EC/ST/CC as in the real instrument) the rule
compares K-Means (k=3) on the four min-max composites with K-Means (k=3) on the overall score alone.
ARI >= tau -> "threshold-only"; otherwise -> "multidimensional structure".
Usage: python simulate_benchmark_first.py [--reps 300]
"""
import argparse, itertools, numpy as np, pandas as pd, warnings
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI
warnings.filterwarnings("ignore")
ap=argparse.ArgumentParser(); ap.add_argument("--reps",type=int,default=300); a=ap.parse_args()
SIZES={"EF":6,"EC":7,"ST":5,"CC":4}; CH=list(SIZES)
def items_from_latent(L,rng,noise=0.45):
    """L: n x 4 latent characteristic scores (1..5 scale) -> n x 22 integer Likert items"""
    cols=[]
    for j,c in enumerate(CH):
        for _ in range(SIZES[c]):
            cols.append(L[:,j]+rng.normal(0,noise,len(L)))
    return np.clip(np.rint(np.column_stack(cols)),1,5)
def composites(X):
    idx=np.cumsum([0]+[SIZES[c] for c in CH]); return np.column_stack([X[:,idx[i]:idx[i+1]].mean(1) for i in range(4)])
def gen_U(n,rng):               # one dominant dimension, unimodal, left-skewed
    th=4.25+0.55*rng.standard_normal(n)-0.25*np.abs(rng.standard_normal(n))
    return th[:,None]+rng.normal(0,0.10,(n,4))
def gen_G(n,rng):               # three groups that differ only in overall level (one dimension)
    g=rng.choice(3,n,p=[.17,.42,.41]); mu=np.array([3.1,4.0,4.85]); sd=np.array([.40,.30,.15])
    th=mu[g]+sd[g]*rng.standard_normal(n); return th[:,None]+rng.normal(0,0.10,(n,4))
PAT=np.array([[+.5,+.5,-.5,-.5],[-.5,-.5,+.5,+.5],[0,0,0,0]])
def gen_M(n,rng,s):             # three types with different profiles across characteristics
    g=rng.choice(3,n); common=rng.normal(0,0.45,n)
    L=4.1+s*PAT[g]+common[:,None]+rng.normal(0,0.20,(n,4)); return L
def km(Z,seed): return KMeans(3,init="k-means++",n_init=5,random_state=seed).fit_predict(Z)
def rule(X,seed):
    C=composites(X); ov=C.mean(1)
    Z=(C-C.min(0))/(C.max(0)-C.min(0)+1e-12)
    l4=km(Z,seed); l1=km(ov[:,None],seed)
    r=np.corrcoef(C.T); mr=r[np.triu_indices(4,1)].mean()
    return ARI(l4,l1),mr
def run(gen,n,reps,seed0,**kw):
    out=[]
    for r in range(reps):
        rng=np.random.default_rng(seed0+r); X=items_from_latent(np.clip(gen(n,rng,**kw),1,5),rng)
        out.append(rule(X,seed0+r))
    return np.array(out)
rows=[]; N=314
for name,gen,kw in [("U: one dimension, unimodal",gen_U,{}),("G: groups on one dimension",gen_G,{})]:
    o=run(gen,N,a.reps,1000,**kw)
    rows.append(dict(scenario=name,sep=np.nan,n=N,meanARI=o[:,0].mean(),sdARI=o[:,0].std(),meanR=o[:,1].mean(),
        **{f"P(thr>={t})":(o[:,0]>=t).mean() for t in (0.8,0.9,0.95)}))
for s in (0.0,0.2,0.4,0.6,0.8,1.0,1.4):
    o=run(gen_M,N,a.reps,2000,s=s)
    rows.append(dict(scenario="M: three profile types",sep=s,n=N,meanARI=o[:,0].mean(),sdARI=o[:,0].std(),meanR=o[:,1].mean(),
        **{f"P(thr>={t})":(o[:,0]>=t).mean() for t in (0.8,0.9,0.95)}))
for n in (150,600):
    for name,gen,kw in [("U: one dimension, unimodal",gen_U,{}),("G: groups on one dimension",gen_G,{}),("M: three profile types",gen_M,{"s":0.8})]:
        o=run(gen,n,a.reps,3000,**kw)
        rows.append(dict(scenario=name,sep=kw.get("s",np.nan),n=n,meanARI=o[:,0].mean(),sdARI=o[:,0].std(),meanR=o[:,1].mean(),
            **{f"P(thr>={t})":(o[:,0]>=t).mean() for t in (0.8,0.9,0.95)}))
df=pd.DataFrame(rows); pd.set_option("display.width",200); print(df.round(3).to_string(index=False)); df.round(4).to_csv("simulation_results.csv",index=False)
