import numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
df=pd.read_csv('qiu_ais_survey_anonymized.csv')
items=list(df.columns[1:23]); X=df[items].astype(float)
LAB=["EF","EF","EF","EC","EC","EC","ST","CC","EF","EC","ST","ST","CC","EF","EC","EC","ST","CC","EF","ST","EC","CC"]
MOD=['REG']*8+['GRD']*6+['SCH']*4+['PAY']*4
print(items)
chi,p=calculate_bartlett_sphericity(X); kmo_all,kmo=calculate_kmo(X); print('KMO',round(kmo,3),'Bartlett',round(chi,2),p)
R=np.corrcoef(X.T); ev=np.sort(np.linalg.eigvalsh(R))[::-1]; print('eig',ev[:6].round(3),'prop',(ev[:3]/22*100).round(1))
# parallel analysis (Pearson, 1000 sims, 95th percentile; also mean)
rng=np.random.default_rng(1); sims=np.array([np.sort(np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((314,22)).T)))[::-1] for _ in range(1000)])
p95=np.percentile(sims,95,axis=0); pm=sims.mean(0); print('PA95',p95[:4].round(3),'PAmean',pm[:4].round(3),'n factors (95th):',int((ev>p95).sum()),'(mean):',int((ev>pm).sum()))
# PA on reduced correlation (PAF eigenvalues) - use factor_analyzer
# Ordinal-aware PA: resample from observed marginals (permutation of columns)
perm=[]
for _ in range(1000):
    Xp=np.column_stack([rng.permutation(X.values[:,j]) for j in range(22)]); perm.append(np.sort(np.linalg.eigvalsh(np.corrcoef(Xp.T)))[::-1])
perm=np.array(perm); pp95=np.percentile(perm,95,axis=0); print('PA perm 95',pp95[:4].round(3),'n factors',int((ev>pp95).sum()))
# MAP test (Velicer)
def map_test(R):
    w,v=np.linalg.eigh(R); idx=np.argsort(w)[::-1]; w,v=w[idx],v[:,idx]; out=[]
    for m in range(0,10):
        if m==0: C=R.copy()
        else:
            A=v[:,:m]*np.sqrt(w[:m]); C=R-A@A.T
        d=np.sqrt(np.diag(C)); Cp=C/np.outer(d,d); out.append(((Cp**2).sum()-22)/(22*21))
    return out
mp=map_test(R); print('MAP',np.round(mp,4),'min at',int(np.argmin(mp)))
# EFA: extraction principal axis, 1 factor and 2 factors oblimin
for nf,rot in [(1,None),(2,'oblimin'),(2,'varimax')]:
    fa=FactorAnalyzer(n_factors=nf,method='principal',rotation=rot,rotation_kwargs={}); 
    fa=FactorAnalyzer(n_factors=nf,method='principal',rotation=rot); fa.fit(X)
    L=pd.DataFrame(fa.loadings_,index=items); h=fa.get_communalities()
    print('\n',nf,'factors',rot,'communalities range',h.min().round(2),h.max().round(2),'mean',h.mean().round(3))
    print('var explained (SS loadings, prop, cum):',[np.round(v,3) for v in fa.get_factor_variance()])
    if rot=='oblimin': print('factor corr',fa.phi_.round(3).tolist())
    L['h2']=h; L['mod']=MOD; L['char']=LAB
    L.to_csv(f'load_{nf}_{rot}.csv')
    print(L.round(2).to_string())
