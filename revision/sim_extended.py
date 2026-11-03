#!/usr/bin/env python3
"""Extended simulation for calibrating and validating the benchmark-first ARI threshold (tau).
Item-to-characteristic and item-to-module mapping follow the real questionnaire (data_dictionary.csv).
Data-generating mechanism (per respondent i):
  severity theta_i  : class A  unimodal, left-skewed  theta = base + s_theta*z - kappa*|z'|
                      class B  kG level-only groups (means linspace(3.0,4.85,kG), SDs linspace(.40,.15,kG)), no profile structure
                      classes C, D  unimodal severity as in A
  profile structure : class C  three profile types, opposite types differ by gap s (>= 0.6): PAT*s with PAT rows (+.5,+.5,-.5,-.5), (-.5,-.5,+.5,+.5), 0
                      class D  weak or continuous structure: gap s in {0.2, 0.4} or a continuous residual factor f_i ~ N(0, rho) times (+.5,+.5,-.5,-.5)
  characteristic latent L_ij = theta_i + profile_ij + N(0, sigma_c * w_j)     (w = 1 or (0.6,1,1.4,1) for heterogeneous variances), clipped to [1,5]
  item x_ik = L_i,char(k) + N(0, sigma_item) + u_i,module(k),  u ~ N(0, sigma_mod) (correlated errors within module), rounded and clipped to {1..5}
Rule: K-Means (k-means++, n_init=5) on min-max scaled composites (scaling re-estimated in every replication) vs K-Means on the mean of the composites; ARI.
Residual-profile test (conditional on the overall score): lambda_1 of the covariance of composite residuals (composite minus respondent mean) against a
parametric-bootstrap null from a one-factor (level-only) Gaussian model fitted to the composites (B = 99; alpha = .05).  Composite parallel analysis: second eigenvalue of the correlation matrix against the 95th percentile of 50 random normal datasets.
Usage: python sim_extended.py [--reps 60] [--configs 240] [--out sim_extended_raw.csv]"""
import argparse, numpy as np, pandas as pd, warnings
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.decomposition import FactorAnalysis
from sklearn.metrics import adjusted_rand_score as ARI
warnings.filterwarnings("ignore")
ap=argparse.ArgumentParser(); ap.add_argument("--reps",type=int,default=60); ap.add_argument("--configs",type=int,default=240)
ap.add_argument("--out",default="sim_extended_raw.csv"); ap.add_argument("--start",type=int,default=0); ap.add_argument("--end",type=int,default=10**9); ap.add_argument("--jobs",type=int,default=4); a=ap.parse_args()
CHARS={"EF":["REG1","REG2","REG3","GRD1","SCH1","PAY1"],"EC":["REG4","REG5","REG6","GRD2","SCH2","SCH3","PAY3"],"ST":["REG7","GRD3","GRD4","SCH4","PAY2"],"CC":["REG8","GRD5","SCH5","PAY4"]}
ITEMS=[f"REG{i}" for i in range(1,9)]+[f"GRD{i}" for i in range(1,6)]+[f"SCH{i}" for i in range(1,6)]+[f"PAY{i}" for i in range(1,5)]
CIDX=np.array([[j for j,(k,v) in enumerate(CHARS.items()) if it in v][0] for it in ITEMS])
MIDX=np.array([["REG","GRD","SCH","PAY"].index(it[:3]) for it in ITEMS])
PAT=np.array([[+.5,+.5,-.5,-.5],[-.5,-.5,+.5,+.5],[0,0,0,0]])
KGP={2:[.3,.7],3:[.17,.42,.41],4:[.10,.20,.30,.40]}
def make_configs(N,seed=20261104):
    rng=np.random.default_rng(seed); out=[]
    cls=rng.permutation(np.repeat(list("ABCD"),[N//6,N//4,N//3+N-(N//6+N//4+N//3+N//4),N//4]))[:N]
    for cid,c in enumerate(cls):
        cfg=dict(cid=cid,cls=c,n=int(rng.choice([150,314,600])),sigma_item=float(rng.choice([.35,.45,.60])),
                 s_theta=float(rng.choice([.40,.55,.80])),kappa=float(rng.choice([0,.25,.5])),base=float(rng.choice([4.0,4.25,4.6])),
                 sigma_c=float(rng.choice([.10,.20,.30])),het=int(rng.integers(0,2)),sigma_mod=float(rng.choice([0,.30])),
                 props=str(rng.choice(["equal","skewed","survey"])),kG=int(rng.choice([2,3,4])),gap=0.0,rho=0.0)
        if c=="C": cfg["gap"]=float(rng.choice([.6,.8,1.0,1.4]))
        if c=="D":
            if rng.random()<.5: cfg["gap"]=float(rng.choice([.2,.4]))
            else: cfg["rho"]=float(rng.choice([.10,.20,.30]))
        out.append(cfg)
    return out
def props_of(cfg,k):
    return {"equal":[1/k]*k,"skewed":([.6,.3,.1] if k==3 else [.5,.5] if k==2 else [.1,.2,.3,.4]),"survey":KGP[k]}[cfg["props"]]
def generate(cfg,rng):
    n=cfg["n"]
    if cfg["cls"]=="B":
        k=cfg["kG"]; g=rng.choice(k,n,p=KGP[k]); mu=np.linspace(3.0,4.85,k); sd=np.linspace(.40,.15,k); th=mu[g]+sd[g]*rng.standard_normal(n)
    else:
        th=cfg["base"]+cfg["s_theta"]*rng.standard_normal(n)-cfg["kappa"]*np.abs(rng.standard_normal(n))
    prof=np.zeros((n,4))
    if cfg["cls"]=="C" or (cfg["cls"]=="D" and cfg["gap"]>0):
        g=rng.choice(3,n,p=props_of(cfg,3)); prof=cfg["gap"]*PAT[g]
    elif cfg["cls"]=="D":
        prof=(cfg["rho"]*rng.standard_normal(n))[:,None]*PAT[0][None,:]
    w=np.array([.6,1,1.4,1]) if cfg["het"] else np.ones(4)
    L=np.clip(th[:,None]+prof+rng.normal(0,1,(n,4))*cfg["sigma_c"]*w,1,5)
    X=L[:,CIDX]+rng.normal(0,cfg["sigma_item"],(n,22))
    if cfg["sigma_mod"]>0: X=X+rng.normal(0,cfg["sigma_mod"],(n,4))[:,MIDX]
    return np.clip(np.rint(X),1,5)
def comps(X): return np.column_stack([X[:,CIDX==j].mean(1) for j in range(4)])
def km(Z,k,seed): return KMeans(k,init="k-means++",n_init=5,random_state=seed).fit_predict(Z)
def changes(l,sc): o=np.argsort(sc,kind="stable"); s=l[o]; return int((s[1:]!=s[:-1]).sum())
def resid_test(C,rng,B=99):
    """Conditional residual-profile test. Null = one-factor (level-only) Gaussian model fitted to the composites
    (parametric bootstrap with the fitted loadings and unique variances; Hennig & Lin style tailored null model).
    Statistic = largest eigenvalue of the covariance of residuals (composite minus respondent mean)."""
    n=len(C); fa=FactorAnalysis(1).fit(C); lam=fa.components_[0]; psi=fa.noise_variance_
    R=C-C.mean(1,keepdims=True); stat=np.linalg.eigvalsh(np.cov(R.T))[-1]; ge=0
    for _ in range(B):
        f=rng.standard_normal(n); Cs=fa.mean_+f[:,None]*lam+rng.standard_normal((n,4))*np.sqrt(psi)
        Rs=Cs-Cs.mean(1,keepdims=True); ge+=np.linalg.eigvalsh(np.cov(Rs.T))[-1]>=stat
    return (1+ge)/(B+1)<=0.05, stat/np.trace(np.cov(C.T))
def pa_test(C,rng,B=50):
    ev=np.sort(np.linalg.eigvalsh(np.corrcoef(C.T)))[::-1]
    sim=np.array([np.sort(np.linalg.eigvalsh(np.corrcoef(rng.standard_normal(C.shape).T)))[::-1] for _ in range(B)])
    return bool(ev[1]>np.percentile(sim[:,1],95))
def one(cfg,r):
    seed=100000*cfg["cid"]+r; rng=np.random.default_rng(seed); X=generate(cfg,rng); C=comps(X); ov=C.mean(1)
    Z=(C-C.min(0))/(C.max(0)-C.min(0)+1e-12); rows=[]
    rp,share=resid_test(C,rng); pa=pa_test(C,rng)
    for kr in (2,3,4):
        l4=km(Z,kr,seed); l1=km(ov[:,None],kr,seed)
        rows.append(dict(cid=cfg["cid"],rep=r,krule=kr,ari=ARI(l4,l1),mono_changes=changes(l4,ov),resid_flag=rp,resid_share=share,pa_flag=pa))
    return rows
if __name__=="__main__":
    cfgs=make_configs(a.configs); pd.DataFrame(cfgs).to_csv("sim_extended_configs.csv",index=False); cfgs=cfgs[a.start:a.end]
    res=Parallel(n_jobs=a.jobs)(delayed(one)(c,r) for c in cfgs for r in range(a.reps))
    pd.DataFrame([x for rr in res for x in rr]).to_csv(a.out,index=False); print("done",len(res)*3,"rows")
