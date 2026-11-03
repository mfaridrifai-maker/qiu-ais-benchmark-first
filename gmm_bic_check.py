#!/usr/bin/env python3
"""GMM check: BIC over k = 1..10 (full covariance), the k = 3 mixture used as comparator, and its Silhouette/ARI vs K-Means.
Usage: python gmm_bic_check.py qiu_ais_survey_anonymized.csv"""
import sys,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from sklearn.mixture import GaussianMixture as GM
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari, silhouette_samples as ss, silhouette_score
df=pd.read_csv(sys.argv[1] if len(sys.argv)>1 else 'qiu_ais_survey_anonymized.csv'); items=list(df.columns[1:23])
LAB=["EF","EF","EF","EC","EC","EC","ST","CC","EF","EC","ST","ST","CC","EF","EC","EC","ST","CC","EF","ST","EC","CC"]
X=pd.DataFrame({k:df[[i for i,l in zip(items,LAB) if l==k]].mean(axis=1) for k in ["EF","EC","ST","CC"]}); Z=((X-X.min())/(X.max()-X.min())).values
b=[GM(k,covariance_type='full',n_init=10,random_state=42).fit(Z).bic(Z) for k in range(1,11)]; print('BIC k=1..10',np.round(b,0),'min at k =',int(np.argmin(b))+1)
l=GM(3,covariance_type='full',n_init=10,random_state=42).fit_predict(Z); km=KMeans(3,n_init=10,random_state=42).fit_predict(Z)
all5=(df[items]==5).all(axis=1).values; s=ss(Z,l)
print('k=3 mixture: Silhouette %.3f, ARI vs K-Means %.3f, sizes %s'%(silhouette_score(Z,l),ari(l,km),np.bincount(l)))
print('respondents with all 22 items = 5:',all5.sum(),'by component',np.bincount(l[all5],minlength=3),'| Silhouette by component',[round(s[l==i].mean(),3) for i in range(3)])
m=~all5; l2=GM(3,covariance_type='full',n_init=10,random_state=42).fit_predict(Z[m]); km2=KMeans(3,n_init=10,random_state=42).fit_predict(Z[m])
print('without all-5 respondents: Silhouette %.3f, ARI %.3f'%(silhouette_score(Z[m],l2),ari(l2,km2)))
