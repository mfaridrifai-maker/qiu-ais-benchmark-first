"""K-Means bootstrap stability for k = 2, 3, 4 under the uniform protocol (see stability_uniform.py)."""
import numpy as np, pandas as pd, sys
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI
df=pd.read_csv(sys.argv[1])
CHAR={"EF":["REG1","REG2","REG3","GRD1","SCH1","PAY1"],"EC":["REG4","REG5","REG6","GRD2","SCH2","SCH3","PAY3"],"ST":["REG7","GRD3","GRD4","SCH4","PAY2"],"CC":["REG8","GRD5","SCH5","PAY4"]}
C=np.column_stack([df[v].astype(float).mean(axis=1) for v in CHAR.values()])
mm=lambda A,r:(A-r.min(0))/(r.max(0)-r.min(0)+1e-12)
Z0=mm(C,C); n=len(C)
for k in (2,3,4):
    ref=KMeans(k,init="k-means++",n_init=5,random_state=0).fit(Z0).predict(Z0)
    rng=np.random.RandomState(42); v=[]
    for b in range(200):
        idx=rng.randint(0,n,n); Cb=C[idx]
        m=KMeans(k,init="k-means++",n_init=5,random_state=b).fit(mm(Cb,Cb)); v.append(ARI(ref,m.predict(mm(C,Cb))))
    v=np.array(v); print(f"k={k}: mean ARI {v.mean():.3f} SD {v.std():.3f} 95% interval [{np.percentile(v,2.5):.3f},{np.percentile(v,97.5):.3f}]")
