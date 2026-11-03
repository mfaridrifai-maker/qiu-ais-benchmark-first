"""Uniform bootstrap stability (R1-3). For each method: resample respondents with replacement (B=200, RandomState(42)),
re-estimate min-max scaling on the resample, refit the method (k=3) on the resample, assign ALL original respondents
(K-Means: nearest centre; GMM: predict; fuzzy c-means: highest membership; hierarchical: nearest resample-cluster centroid),
ARI against the full-sample partition of the same method. Fuzzy c-means is a plain numpy implementation (m=2)."""
import numpy as np, pandas as pd, sys
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score as ARI
df=pd.read_csv(sys.argv[1]); B=int(sys.argv[2]) if len(sys.argv)>2 else 200
CHAR={"EF":["REG1","REG2","REG3","GRD1","SCH1","PAY1"],"EC":["REG4","REG5","REG6","GRD2","SCH2","SCH3","PAY3"],"ST":["REG7","GRD3","GRD4","SCH4","PAY2"],"CC":["REG8","GRD5","SCH5","PAY4"]}
C=np.column_stack([df[v].astype(float).mean(axis=1) for v in CHAR.values()])
def mm(A,ref): return (A-ref.min(0))/(ref.max(0)-ref.min(0)+1e-12)
def fcm(X,k=3,m=2,it=200,seed=0):
    r=np.random.RandomState(seed); U=r.dirichlet(np.ones(k),len(X))
    for _ in range(it):
        W=U**m; V=(W.T@X)/W.sum(0)[:,None]
        D=np.linalg.norm(X[:,None]-V[None],axis=2)+1e-12
        Un=1/(D**(2/(m-1))*np.sum(D**(-2/(m-1)),axis=1,keepdims=True)); 
        if np.abs(Un-U).max()<1e-6: U=Un;break
        U=Un
    return V
def assign_c(X,V): return np.linalg.norm(X[:,None]-V[None],axis=2).argmin(1)
def fit(method,Xtr,seed):
    if method=="KMeans":
        m=KMeans(3,init="k-means++",n_init=5,random_state=seed).fit(Xtr); return lambda X:m.predict(X)
    if method=="FuzzyCMeans":
        V=fcm(Xtr,seed=seed); return lambda X:assign_c(X,V)
    if method=="Hierarchical":
        l=AgglomerativeClustering(3,linkage="ward").fit_predict(Xtr); V=np.array([Xtr[l==j].mean(0) for j in range(3)]); return lambda X:assign_c(X,V)
    if method=="GMM":
        g=GaussianMixture(3,covariance_type="full",n_init=2,random_state=seed).fit(Xtr); return lambda X:g.predict(X)
rng=np.random.RandomState(42); n=len(C); res={}
Z0=mm(C,C)
ref={m:fit(m,Z0,0)(Z0) for m in ["KMeans","FuzzyCMeans","Hierarchical","GMM"]}
ref["KMeans"]=KMeans(3,init="k-means++",n_init=10,random_state=42).fit_predict(Z0)   # reported full-sample solution (n_init=10, random_state=42; lowest inertia). Other methods: reference = same fit/assignment rule as in the resamples.
for m in ref: res[m]=[]
for b in range(B):
    idx=rng.randint(0,n,n); Cb=C[idx]; Zb=mm(Cb,Cb); Zo=mm(C,Cb)   # scaling re-estimated on the resample
    for m in ref:
        try: res[m].append(ARI(ref[m],fit(m,Zb,b)(Zo)))
        except Exception: pass
for m,v in res.items():
    v=np.array(v); print(f"{m}: mean ARI {v.mean():.3f}  SD {v.std():.3f}  95% interval [{np.percentile(v,2.5):.3f}, {np.percentile(v,97.5):.3f}]  n={len(v)}")
