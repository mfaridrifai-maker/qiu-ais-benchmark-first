import numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from factor_analyzer import FactorAnalyzer
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
df=pd.read_csv('qiu_ais_survey_anonymized.csv')
items=list(df.columns[1:23]); X=df[items].astype(float)
MOD=['REG']*8+['GRD']*5+['SCH']*5+['PAY']*4
res={}
for nf,rot in [(1,None),(2,'oblimin'),(3,'oblimin')]:
    fa=FactorAnalyzer(n_factors=nf,method='minres',rotation=rot); fa.fit(X)
    L=pd.DataFrame(fa.loadings_,index=items); h=pd.Series(fa.get_communalities(),index=items)
    print(nf,rot,'h2 range %.2f-%.2f mean %.3f'%(h.min(),h.max(),h.mean()),'SS',np.round(fa.get_factor_variance()[0],2),'cum prop',np.round(fa.get_factor_variance()[2],3), 'phi',(fa.phi_.round(2).tolist() if rot else None))
    res[nf]=(L,h)
    if nf==1: print('loadings range',L[0].abs().min().round(2),L[0].abs().max().round(2))
L1,h1=res[1]; L2,h2=res[2]; L3,h3=res[3]
sign=np.sign(L1[0].mean()); L1=L1*sign
t=pd.DataFrame({'item':items,'module':MOD,'L1':L1[0].round(2),'h2_1f':h1.round(2),'F1':L2[0].round(2),'F2':L2[1].round(2),'h2_2f':h2.round(2)})
print(t.to_string(index=False)); t.to_csv('efa_table.csv',index=False)
# which factor dominates per module
print(t.groupby('module')[['F1','F2']].mean().round(2))
# cross-loading items (|both|>=.30)
print('cross-loading >=.30 both:',t[(t.F1.abs()>=.3)&(t.F2.abs()>=.3)].item.tolist())
print('3f phi',FactorAnalyzer(n_factors=3,method='minres',rotation='oblimin').fit(X).phi_.round(2).tolist())
print(L3.round(2).assign(mod=MOD).to_string())
# scree + PA
R=np.corrcoef(X.values.T); ev=np.sort(np.linalg.eigvalsh(R))[::-1]
rng=np.random.default_rng(1); sims=np.array([np.sort(np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((314,22)).T)))[::-1] for _ in range(1000)])
p95=np.percentile(sims,95,axis=0)
fig,ax=plt.subplots(figsize=(3.4,2.5),dpi=300)
k=np.arange(1,11); BLUE='#5B9BD5'; RED='#E15759'
ax.plot(k,ev[:10],'s-',color=BLUE,ms=3.5,lw=1.2,label='Observed eigenvalues')
ax.plot(k,p95[:10],'o--',color=RED,ms=3,lw=1,label='Parallel analysis (95th)')
ax.axhline(1,color='gray',lw=0.6,ls=':'); ax.text(10,0.84,'Kaiser = 1',ha='right',fontsize=7)
for i in range(3): ax.annotate('%.2f'%ev[i],(k[i],ev[i]),textcoords='offset points',xytext=((5,4) if i==0 else (3,-10)),fontsize=6)
ax.set_xlabel('Factor number',fontsize=8,fontweight='bold'); ax.set_ylabel('Eigenvalue',fontsize=8,fontweight='bold'); ax.set_yscale('log'); ax.set_ylim(0.7,18); ax.set_yticks([1,2,5,10]); ax.set_yticklabels(['1','2','5','10']); ax.set_xticks(k)
ax.tick_params(labelsize=7); ax.legend(fontsize=7,frameon=False)
plt.tight_layout(); plt.savefig('scree.png'); print('saved')
