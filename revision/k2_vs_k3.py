"""2- vs 3-group reporting comparison on the real data (R2-4)."""
import numpy as np, pandas as pd, sys
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI, silhouette_score
df=pd.read_csv(sys.argv[1])
CHAR={"EF":["REG1","REG2","REG3","GRD1","SCH1","PAY1"],"EC":["REG4","REG5","REG6","GRD2","SCH2","SCH3","PAY3"],"ST":["REG7","GRD3","GRD4","SCH4","PAY2"],"CC":["REG8","GRD5","SCH5","PAY4"]}
C=np.column_stack([df[v].astype(float).mean(axis=1) for v in CHAR.values()]); ov=C.mean(1)
Z=(C-C.min(0))/(C.max(0)-C.min(0))
for k in (2,3,4):
    a=KMeans(k,init="k-means++",n_init=10,random_state=42).fit(Z); b=KMeans(k,init="k-means++",n_init=10,random_state=42).fit(ov[:,None])
    la,lb=a.labels_,b.labels_
    order=np.argsort([ov[lb==j].mean() for j in range(k)]); rank=np.empty(k,int); rank[order]=range(k)
    cuts=[(ov[lb==order[i]].max()+ov[lb==order[i+1]].min())/2 for i in range(k-1)]
    print(f"k={k}: ARI(composites vs overall-score)={ARI(la,lb):.3f}; silhouette(composites)={silhouette_score(Z,la):.3f}; silhouette(score)={silhouette_score(ov[:,None],lb):.3f}")
    print("   sizes (score-only, ascending):",[round(float((lb==order[i]).mean()*100),1) for i in range(k)],"| score cut-offs:",[round(c,2) for c in cuts])
    print("   crosstab composites(rows) x score-only(cols, ascending):\n",pd.crosstab(la,rank[lb]).to_string().replace("\n","\n   "))
