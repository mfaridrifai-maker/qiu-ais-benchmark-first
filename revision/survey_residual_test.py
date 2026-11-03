"""Conditional residual-profile test on the survey (R1-2, R1-7). Null: one-factor Gaussian parametric bootstrap on the four composites (2,000 data sets, default_rng(3)).
Usage: python survey_residual_test.py ../qiu_ais_survey_anonymized.csv"""
import numpy as np, pandas as pd, sys
from sklearn.decomposition import FactorAnalysis
df=pd.read_csv(sys.argv[1] if len(sys.argv)>1 else '../qiu_ais_survey_anonymized.csv')
CHAR={"EF":["REG1","REG2","REG3","GRD1","SCH1","PAY1"],"EC":["REG4","REG5","REG6","GRD2","SCH2","SCH3","PAY3"],"ST":["REG7","GRD3","GRD4","SCH4","PAY2"],"CC":["REG8","GRD5","SCH5","PAY4"]}
C=np.column_stack([df[v].astype(float).mean(axis=1) for v in CHAR.values()]); n=len(C)
fa=FactorAnalysis(1).fit(C); lam=fa.components_[0]; psi=fa.noise_variance_
R=C-C.mean(1,keepdims=True); stat=np.linalg.eigvalsh(np.cov(R.T))[-1]
rng=np.random.default_rng(3); null=[]
for _ in range(2000):
    Cs=fa.mean_+rng.standard_normal(n)[:,None]*lam+rng.standard_normal((n,4))*np.sqrt(psi)
    Rs=Cs-Cs.mean(1,keepdims=True); null.append(np.linalg.eigvalsh(np.cov(Rs.T))[-1])
null=np.array(null)
print(f"lambda1 observed {stat:.4f}; null 95th percentile {np.percentile(null,95):.4f}; p = {(1+(null>=stat).sum())/2001:.3f}")
print("one-factor loadings (EF,EC,ST,CC):",lam.round(3),"unique variances:",psi.round(3))
