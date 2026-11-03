import numpy as np,pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari, silhouette_score
df=pd.read_csv('qiu_ais_survey_anonymized.csv')
items=list(df.columns[1:23])
LAB=["EF","EF","EF","EC","EC","EC","ST","CC","EF","EC","ST","ST","CC","EF","EC","EC","ST","CC","EF","ST","EC","CC"]
X=pd.DataFrame({k:df[[i for i,l in zip(items,LAB) if l==k]].mean(axis=1) for k in ["EF","EC","ST","CC"]}); S=X.mean(axis=1)
print('observed ranges',(X.max()-X.min()).round(2).to_dict(),'SDs',X.std().round(3).to_dict(),'IQR',(X.quantile(.75)-X.quantile(.25)).round(2).to_dict())
def sc(name):
    if name=='minmax': return (X-X.min())/(X.max()-X.min())
    if name=='raw': return X.copy()
    if name=='z': return (X-X.mean())/X.std()
    if name=='robust': return (X-X.median())/(X.quantile(.75)-X.quantile(.25))
def order(l,S):
    o=np.argsort([S.values[l==i].mean() for i in range(3)]); r={x:i for i,x in enumerate(o)}; return np.array([r[i] for i in l])
lo=order(KMeans(3,n_init=10,random_state=42).fit_predict(S.values.reshape(-1,1)),S)
main=order(KMeans(3,n_init=10,random_state=42).fit_predict(sc('minmax')),S)
for n in ['minmax','raw','z','robust']:
    Z=sc(n); l=order(KMeans(3,n_init=10,random_state=42).fit_predict(Z),S)
    srt=[S.values[l==q] for q in range(3)]; th=[(srt[0].max()+srt[1].min())/2,(srt[1].max()+srt[2].min())/2]
    print(f'{n:8s} shares {[round((l==q).mean()*100,1) for q in range(3)]} thr {th[0]:.2f}/{th[1]:.2f} ARI vs overall-only {ari(l,lo):.3f} vs minmax-groups {ari(l,main):.3f} Silh {silhouette_score(Z,l):.3f}')
