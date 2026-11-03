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
def rule(X,seed,mode='char'):
    C=composites(X); ov=C.mean(1) if mode=='char' else X.mean(1)
    Z=(C-C.min(0))/(C.max(0)-C.min(0)+1e-12)
    l4=km(Z,seed); l1=km(ov[:,None],seed)
    r=np.corrcoef(C.T); mr=r[np.triu_indices(4,1)].mean()
    return ARI(l4,l1),mr

from joblib import Parallel,delayed
def one(gen,n,seed0,r,kw):
    rng=np.random.default_rng(seed0+r); X=items_from_latent(np.clip(gen(n,rng,**kw),1,5),rng)
    return rule(X,seed0+r,'char')[0],rule(X,seed0+r,'item')[0]
rows=[]
for name,gen,kw,seed0 in [('U',gen_U,{},1000),('G',gen_G,{},1000),('M0.4',gen_M,{'s':0.4},2000),('M0.6',gen_M,{'s':0.6},2000),('M0.8',gen_M,{'s':0.8},2000)]:
    o=np.array(Parallel(n_jobs=4)(delayed(one)(gen,314,seed0,r,kw) for r in range(300)))
    print(name,'thr@0.8 char %.1f%% item %.1f%%'%((o[:,0]>=.8).mean()*100,(o[:,1]>=.8).mean()*100),'meanARI char %.3f item %.3f'%(o[:,0].mean(),o[:,1].mean()),'thr@0.9 char %.1f item %.1f'%((o[:,0]>=.9).mean()*100,(o[:,1]>=.9).mean()*100),flush=True)
