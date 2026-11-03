#!/usr/bin/env python3
"""Sensitivity of the benchmark-first rule to the definition of the overall QiU score (real data).
Definition A (main analysis): unweighted mean of the four characteristic composites (EF, EC, ST, CC).
Definition B: mean of all 22 items. Usage: python overall_definition_sensitivity.py qiu_ais_survey_anonymized.csv
For the simulation counterpart see simulate_benchmark_first.py (rule() takes the overall score from the composites)."""
import sys, numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari
df=pd.read_csv(sys.argv[1] if len(sys.argv)>1 else "qiu_ais_survey_anonymized.csv")
items=list(df.columns[1:23])
LAB=["EF","EF","EF","EC","EC","EC","ST","CC","EF","EC","ST","ST","CC","EF","EC","EC","ST","CC","EF","ST","EC","CC"]
X=pd.DataFrame({k:df[[i for i,l in zip(items,LAB) if l==k]].mean(axis=1) for k in ["EF","EC","ST","CC"]})
A=X.mean(axis=1); B=df[items].mean(axis=1)
Z=(X-X.min())/(X.max()-X.min()); lc=KMeans(3,n_init=10,random_state=42).fit_predict(Z)
def grp(S):
    l=KMeans(3,n_init=10,random_state=42).fit_predict(S.values.reshape(-1,1)); o=np.argsort([S.values[l==i].mean() for i in range(3)])
    r={x:i for i,x in enumerate(o)}; return np.array([r[i] for i in l])
lA,lB=grp(A),grp(B)
print("mean A %.3f  B %.3f  Pearson r %.4f"%(A.mean(),B.mean(),np.corrcoef(A,B)[0,1]))
for n,S,l in (("A (4 composites)",A,lA),("B (22 items)",B,lB)):
    srt=[S.values[l==q] for q in range(3)]; th=[(srt[0].max()+srt[1].min())/2,(srt[1].max()+srt[2].min())/2]
    print(n,"shares %s"%[round((l==q).mean()*100,1) for q in range(3)],"thresholds %.2f/%.2f"%tuple(th),"ARI(4-composite K-Means vs overall-only)=%.3f"%ari(lc,l),"share<3.5=%.1f%%"%((S<3.5).mean()*100))
print("ARI between the two overall-score partitions: %.3f"%ari(lA,lB))
