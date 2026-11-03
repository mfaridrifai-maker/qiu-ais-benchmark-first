"""Adds the conditional residual-profile comparator to the Table 5 scenarios (same data: seeds 5000 + r, 60 replications).
Null: one-factor Gaussian parametric bootstrap on the composites (B = 199), flag if p <= 0.05."""
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.decomposition import FactorAnalysis
from joblib import Parallel, delayed
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
def resid_flag(C,rng,B=199):
    n=len(C); fa=FactorAnalysis(1).fit(C); lam=fa.components_[0]; psi=fa.noise_variance_
    R=C-C.mean(1,keepdims=True); stat=np.linalg.eigvalsh(np.cov(R.T))[-1]; ge=0
    for _ in range(B):
        Cs=fa.mean_+rng.standard_normal(n)[:,None]*lam+rng.standard_normal((n,4))*np.sqrt(psi)
        Rs=Cs-Cs.mean(1,keepdims=True); ge+=np.linalg.eigvalsh(np.cov(Rs.T))[-1]>=stat
    return (1+ge)/(B+1)<=0.05
N=314
for name,gen,kw in [("U",gen_U,{}),("G",gen_G,{}),("M 0.4",gen_M,{"s":.4}),("M 0.6",gen_M,{"s":.6}),("M 0.8",gen_M,{"s":.8})]:
    def one(r):
        rng=np.random.default_rng(5000+r); X=items_from_latent(np.clip(gen(N,rng,**kw),1,5),rng)
        return resid_flag(composites(X),np.random.default_rng(900000+r))
    res=Parallel(n_jobs=4)(delayed(one)(r) for r in range(60)); print(name,round(100*np.mean(res),1))
