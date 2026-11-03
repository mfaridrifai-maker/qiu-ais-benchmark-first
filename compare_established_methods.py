#!/usr/bin/env python3
"""Compares the benchmark-first ARI rule with established methods on the same simulated data and on the real survey.
Question asked of every method: does the four-characteristic data contain structure BEYOND the overall level?
Truth: scenarios U and G = no (one dimension); scenario M (profile types) = yes.
Methods: (1) ARI rule (tau=0.80); (2) Horn parallel analysis on the 4 composites (>1 component => structure);
(3) gap statistic (k_hat>1 => clusters); (4) Gaussian-mixture BIC (k_hat>1 => clusters);
(5) Silhouette heuristic (3-cluster Silhouette >= 0.50 => clusters).
Usage: python compare_established_methods.py [--reps 100] [--data qiu_ais_survey_anonymized.csv]"""
from joblib import Parallel, delayed
import argparse, numpy as np, pandas as pd, warnings
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score as ARI, silhouette_score
warnings.filterwarnings("ignore")
ap=argparse.ArgumentParser(); ap.add_argument("--reps",type=int,default=100); ap.add_argument("--data",default=None); a=ap.parse_args()
SIZES={"EF":6,"EC":7,"ST":5,"CC":4}; CH=list(SIZES)
def items_from_latent(L,rng,noise=0.45):
    cols=[]
    for j,c in enumerate(CH):
        for _ in range(SIZES[c]): cols.append(L[:,j]+rng.normal(0,noise,len(L)))
    return np.clip(np.rint(np.column_stack(cols)),1,5)
def composites(X):
    idx=np.cumsum([0]+[SIZES[c] for c in CH]); return np.column_stack([X[:,idx[i]:idx[i+1]].mean(1) for i in range(4)])
def gen_U(n,rng):
    th=4.25+0.55*rng.standard_normal(n)-0.25*np.abs(rng.standard_normal(n)); return th[:,None]+rng.normal(0,0.10,(n,4))
def gen_G(n,rng):
    g=rng.choice(3,n,p=[.17,.42,.41]); mu=np.array([3.1,4.0,4.85]); sd=np.array([.40,.30,.15])
    th=mu[g]+sd[g]*rng.standard_normal(n); return th[:,None]+rng.normal(0,0.10,(n,4))
PAT=np.array([[+.5,+.5,-.5,-.5],[-.5,-.5,+.5,+.5],[0,0,0,0]])
def gen_M(n,rng,s):
    g=rng.choice(3,n); common=rng.normal(0,0.45,n)
    return 4.1+s*PAT[g]+common[:,None]+rng.normal(0,0.20,(n,4))
def km(Z,k,seed,n_init=5): return KMeans(k,init="k-means++",n_init=n_init,random_state=seed).fit(Z)
def minmax(C): return (C-C.min(0))/(C.max(0)-C.min(0)+1e-12)
def m_rule(C,seed):
    ov=C.mean(1); return ARI(km(minmax(C),3,seed).labels_,km(ov[:,None],3,seed).labels_)>=0.80   # True = thresholds only
def m_parallel(C,seed,B=30):
    ev=np.sort(np.linalg.eigvalsh(np.corrcoef(C.T)))[::-1]; rng=np.random.default_rng(seed)
    sim=np.array([np.sort(np.linalg.eigvalsh(np.corrcoef(rng.standard_normal(C.shape).T)))[::-1] for _ in range(B)])
    thr=np.percentile(sim,95,axis=0); return int((ev>thr).sum())>1                                  # True = structure beyond 1 dimension
def m_gap(C,seed,B=10,kmax=6):
    Z=minmax(C); rng=np.random.default_rng(seed); W=[]
    Zc=Z-Z.mean(0); _,_,Vt=np.linalg.svd(Zc,full_matrices=False); Y=Zc@Vt.T; lo,hi=Y.min(0),Y.max(0)   # PCA-aligned reference (Tibshirani 2001, method b)
    for k in range(1,kmax+1): W.append(np.log(km(Z,k,seed,3).inertia_+1e-12))
    Wb=np.zeros((B,kmax))
    for b in range(B):
        R=rng.uniform(lo,hi,Z.shape)@Vt
        for k in range(1,kmax+1): Wb[b,k-1]=np.log(km(R,k,seed,2).inertia_+1e-12)
    gap=Wb.mean(0)-np.array(W); s=Wb.std(0)*np.sqrt(1+1/B)
    for k in range(1,kmax):
        if gap[k-1]>=gap[k]-s[k]: return k>1                                                        # True = clusters
    return kmax>1
def m_gmm(C,seed):
    Z=minmax(C); bic=[GaussianMixture(k,covariance_type="full",n_init=2,random_state=seed).fit(Z).bic(Z) for k in range(1,6)]
    return int(np.argmin(bic))+1>1                                                                   # True = clusters
def m_sil(C,seed):
    Z=minmax(C); return silhouette_score(Z,km(Z,3,seed).labels_)>=0.50                              # True = clusters
def decide(C,seed):
    return dict(rule_structure=not m_rule(C,seed),parallel=m_parallel(C,seed),gap=m_gap(C,seed),gmm=m_gmm(C,seed),silhouette=m_sil(C,seed))
rows=[]; N=314
for name,gen,kw in [("U: one dimension",gen_U,{}),("G: groups on one dimension",gen_G,{})]+[("M: profile types, gap=%.1f"%s,gen_M,{"s":s}) for s in (0.2,0.4,0.6,0.8,1.0)]:
    def one(r,gen=gen,kw=kw):
        rng=np.random.default_rng(5000+r); X=items_from_latent(np.clip(gen(N,rng,**kw),1,5),rng); return decide(composites(X),5000+r)
    res=Parallel(n_jobs=4)(delayed(one)(r) for r in range(a.reps)); print(name,flush=True)
    df=pd.DataFrame(res).mean(); df["scenario"]=name; rows.append(df)
out=pd.DataFrame(rows).set_index("scenario")[["rule_structure","parallel","gap","gmm","silhouette"]]
out.columns=["ARI rule","Parallel analysis","Gap statistic","GMM BIC","Silhouette>=0.5"]
print("Share of replications flagged as 'structure beyond the overall level' (ideal: ~0 for U and G, ~1 for M)"); print(out.round(3).to_string())
out.round(3).to_csv("method_comparison_results.csv")
if a.data:
    df=pd.read_csv(a.data); items=[c for c in df.columns if c[:3] in ("REG","GRD","SCH","PAY") and c[3:].isdigit()]
    CHAR={"EF":["REG1","REG2","REG3","GRD1","SCH1","PAY1"],"EC":["REG4","REG5","REG6","GRD2","SCH2","SCH3","PAY3"],"ST":["REG7","GRD3","GRD4","SCH4","PAY2"],"CC":["REG8","GRD5","SCH5","PAY4"]}
    C=np.column_stack([df[v].astype(float).mean(axis=1) for v in CHAR.values()])
    print("\nReal survey (N=%d):"%len(df)); print(decide(C,42))
