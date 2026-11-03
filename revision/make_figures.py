"""Redraws Figs. 1-5 for the IJIES format: single-column width (3.4 in), fonts >= 10 pt, no outer frame, one panel per figure, no (a)/(b)."""
import numpy as np, pandas as pd, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.8,'savefig.dpi':300})
data,out=sys.argv[1],sys.argv[2]
df=pd.read_csv(data)
CHAR={"EF":["REG1","REG2","REG3","GRD1","SCH1","PAY1"],"EC":["REG4","REG5","REG6","GRD2","SCH2","SCH3","PAY3"],"ST":["REG7","GRD3","GRD4","SCH4","PAY2"],"CC":["REG8","GRD5","SCH5","PAY4"]}
items=[i for v in CHAR.values() for i in v]; X=df[items].astype(float)
C=np.column_stack([df[v].astype(float).mean(axis=1) for v in CHAR.values()]); ov=C.mean(1); Z=(C-C.min(0))/(C.max(0)-C.min(0))
BLUE='#5B9BD5'; RED='#E15759'; GREEN='#6AAB48'; AMBER='#F5B800'
W,H=3.4,2.25
# Fig 1 scree
R=np.corrcoef(X.values.T); ev=np.sort(np.linalg.eigvalsh(R))[::-1]
rng=np.random.default_rng(1); sims=np.array([np.sort(np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((314,22)).T)))[::-1] for _ in range(1000)]); p95=np.percentile(sims,95,axis=0)
fig,ax=plt.subplots(figsize=(W,H)); k=np.arange(1,11)
ax.plot(k,ev[:10],'s-',color=BLUE,ms=4,lw=1.4,label='Observed'); ax.plot(k,p95[:10],'o--',color=RED,ms=3.5,lw=1.2,label='Parallel analysis (95th)')
ax.axhline(1,color='gray',lw=0.7,ls=':'); ax.set_yscale('log'); ax.set_ylim(0.3,20); ax.set_yticks([0.5,1,2,5,10]); ax.set_yticklabels(['0.5','1','2','5','10']); ax.minorticks_off(); ax.set_xticks(k)
ax.set_xlabel('Factor number'); ax.set_ylabel('Eigenvalue'); ax.legend(frameon=False,loc='upper right',handlelength=1.8)
fig.tight_layout(); fig.savefig(f'{out}/fig1_scree.png'); plt.close(fig)
# Fig 2, 3: elbow and silhouette
ks=np.arange(2,9); wc=[]; sl=[]
for kk in ks:
    m=KMeans(kk,init='k-means++',n_init=10,random_state=42).fit(Z); wc.append(m.inertia_); sl.append(silhouette_score(Z,m.labels_))
print('WCSS',np.round(wc,2)); print('Sil',np.round(sl,4))
fig,ax=plt.subplots(figsize=(W,H)); ax.plot(ks,wc,'o-',color=BLUE,ms=5,lw=1.6); ax.axvline(3,color=RED,ls='--',lw=1.3)
ax.text(3.15,max(wc)*0.97,'Selected k = 3',color=RED,fontsize=10,va='top'); ax.set_xlabel('Number of clusters (k)'); ax.set_ylabel('Within-cluster sum of squares'); ax.set_xticks(ks); ax.set_ylim(0,20)
fig.tight_layout(); fig.savefig(f'{out}/fig2_elbow.png'); plt.close(fig)
fig,ax=plt.subplots(figsize=(W,H)); ax.plot(ks,sl,'s-',color=BLUE,ms=5,lw=1.6); ax.axvline(3,color=RED,ls='--',lw=1.3)
ax.text(3.15,0.4415,'Selected k = 3',color=RED,fontsize=10,va='bottom'); ax.set_xlabel('Number of clusters (k)'); ax.set_ylabel('Silhouette score'); ax.set_xticks(ks); ax.set_ylim(0.43,0.56)
fig.tight_layout(); fig.savefig(f'{out}/fig3_silhouette.png'); plt.close(fig)
# groups
km=KMeans(3,init='k-means++',n_init=10,random_state=42).fit(Z); lab=km.labels_
order=np.argsort([ov[lab==j].mean() for j in range(3)]); rank=np.empty(3,int); rank[order]=range(3)  # 0 = Less, 1 = Moderately, 2 = Highly
g=rank[lab]; names=['Less Satisfied','Moderately Satisfied','Highly Satisfied']; cols=[RED,AMBER,GREEN]
print('sizes',[int((g==i).sum()) for i in range(3)])
# Fig 4 histogram
bins=np.arange(1.25,5.26,0.25)
fig,ax=plt.subplots(figsize=(W,H+0.2))
ax.hist([ov[g==i] for i in range(3)],bins=bins,stacked=True,color=cols,edgecolor='white',linewidth=0.6,label=[f'{names[i]} (n = {(g==i).sum()})' for i in range(3)])
for c_ in (3.6,4.4): ax.vlines(c_,0,80,color='#333333',ls='--',lw=1.2)
ax.set_xlabel('Overall QiU score'); ax.set_ylabel('Number of students'); ax.set_xlim(1.2,5.2); ax.set_ylim(0,125); ax.set_xticks([1.5,2.5,3.5,4.5]); ax.set_yticks([0,20,40,60,80])
h,l=ax.get_legend_handles_labels(); ax.legend(h[::-1],l[::-1],frameon=False,loc='upper left',fontsize=10,handlelength=1.0,borderaxespad=0.1)
fig.tight_layout(); fig.savefig(f'{out}/fig4_histogram.png'); plt.close(fig)
# Fig 5 PCA
pc=PCA(2).fit(Z); S=pc.transform(Z); ev_=pc.explained_variance_ratio_*100
fig,ax=plt.subplots(figsize=(W,H+0.2))
for i in (0,1,2): ax.scatter(S[g==i,0],S[g==i,1],s=14,color=cols[i],edgecolor='white',linewidth=0.3,label=names[i],alpha=0.95)
ax.set_xlabel(f'PC1 ({ev_[0]:.2f}% of variance)'); ax.set_ylabel(f'PC2 ({ev_[1]:.2f}%)'); ax.set_ylim(-0.3,0.78); ax.legend(frameon=False,loc='upper left',fontsize=10,handletextpad=0.2,borderaxespad=0.1,markerscale=1.2)
fig.tight_layout(); fig.savefig(f'{out}/fig5_pca.png'); plt.close(fig)
print('PCA',ev_.round(2))
