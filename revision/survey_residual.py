"""(The valid conditional residual test is in survey_residual_test.py.) Residual-profile description of the survey composites (conditional on the overall score), partition cross-tabs, monotonicity."""
import sys
import numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI, silhouette_score
df=pd.read_csv(sys.argv[1] if len(sys.argv)>1 else '../qiu_ais_survey_anonymized.csv'); items=list(df.columns[1:23])
CH={"EF":["REG1","REG2","REG3","GRD1","SCH1","PAY1"],"EC":["REG4","REG5","REG6","GRD2","SCH2","SCH3","PAY3"],"ST":["REG7","GRD3","GRD4","SCH4","PAY2"],"CC":["REG8","GRD5","SCH5","PAY4"]}
MOD={'Course Registration':[c for c in items if c.startswith('REG')],'Grade Viewing':[c for c in items if c.startswith('GRD')],'Class Schedule':[c for c in items if c.startswith('SCH')],'Payment':[c for c in items if c.startswith('PAY')]}
C=np.column_stack([df[v].mean(1) for v in CH.values()]); S=C.mean(1)
def lab3(Z,sc,seed=42):
    l=KMeans(3,n_init=10,random_state=seed).fit_predict(Z); o=np.argsort([sc[l==i].mean() for i in range(3)]); m={x:i for i,x in enumerate(o)}; return np.array([m[i] for i in l])
mm=lambda A:(A-A.min(0))/(A.max(0)-A.min(0))
l4=lab3(mm(C),S); l1=lab3(S[:,None],S)
print('ARI composites vs score-only',round(ARI(l4,l1),4))
# residuals
R=C-C.mean(1,keepdims=True)
tot=C.var(0,ddof=1).sum(); res=R.var(0,ddof=1).sum()
print('variance of composites (sum)',round(tot,4),'| residual variance after removing each respondent mean',round(res,4),'| share',round(res/tot*100,2),'%')
ev=np.sort(np.linalg.eigvalsh(np.cov(R.T)))[::-1]; print('eigenvalues of residual covariance',ev.round(4),'| share of total composite variance by 1st residual comp',round(ev[0]/tot*100,2),'%')
# loading pattern of 1st residual component
w,V=np.linalg.eigh(np.cov(R.T)); print('1st residual component loading (EF,EC,ST,CC):',V[:,-1].round(3))
# crosstabs & monotonicity
def mono(l,sc):
    o=np.argsort(sc,kind='stable'); seq=l[o]; return int((seq[1:]!=seq[:-1]).sum())
print('label changes along sorted overall score: composites',mono(l4,S),'| score-only',mono(l1,S))
M=np.column_stack([df[v].mean(1) for v in MOD.values()]); lm=lab3(mm(M),S)
print('modules vs score-only ARI',round(ARI(lm,l1),3)); print(pd.crosstab(pd.Series(lm,name='modules'),pd.Series(l1,name='score-only')))
print('misassigned',(lm!=l1).sum(),'of 314 =',round((lm!=l1).mean()*100,1),'% ; label changes along score',mono(lm,S))
S22=df[items].mean(1).values; l22=lab3(S22[:,None],S22); print('22-item overall-only vs 4-composite KM ARI',round(ARI(l22,l4),4)); print(pd.crosstab(pd.Series(l4,name='4-composite'),pd.Series(l22,name='22-item')))
print('misassigned',(l22!=l4).sum(),'; mono changes',mono(l22,S22),mono(l4,S22))
tert=(pd.Series(S).rank(method='first').values-1)*3//314; tert=tert.astype(int); print('tertiles vs KM ARI',round(ARI(tert,l4),3)); print(pd.crosstab(pd.Series(tert,name='tertile'),pd.Series(l4,name='KMeans')))
