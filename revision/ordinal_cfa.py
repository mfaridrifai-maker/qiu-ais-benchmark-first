"""Ordinal (polychoric) CFA of the 22 items (R1-4). Polychoric matrix: semopy.polycorr; estimation: DWLS/ULS on the polychoric matrix.
Compares the 4-characteristic model with a 1-factor model and reports loadings, latent correlations (admissibility), residual diagnostics."""
import numpy as np, pandas as pd, sys, itertools, warnings; warnings.filterwarnings("ignore")
from semopy import Model, calc_stats
from scipy.stats import norm
from scipy.optimize import minimize_scalar
GX,GW=np.polynomial.legendre.leggauss(24)
def bvn(h,k,r):
    """P(X<=h,Y<=k), corr r: Phi(h)Phi(k)+int_0^r phi2(h,k;t)dt (Gauss-Legendre); h,k arrays, +-inf allowed."""
    h=np.asarray(h,float); k=np.asarray(k,float); out=norm.cdf(h)*norm.cdf(k)
    fin=np.isfinite(h)&np.isfinite(k)
    t=r/2*(GX+1); w=GW*r/2
    hh=h[fin][:,None]; kk=k[fin][:,None]
    ph=np.exp(-(hh**2-2*t*hh*kk+kk**2)/(2*(1-t**2)))/(2*np.pi*np.sqrt(1-t**2))
    out[fin]+= (ph*w).sum(1); return out
def polychoric_corr(x,y):
    x=np.asarray(x); y=np.asarray(y)
    cx=np.unique(x); cy=np.unique(y)
    T=np.array([[np.sum((x==a)&(y==b)) for b in cy] for a in cx],float); N=T.sum()
    tx=norm.ppf(np.clip(np.cumsum(T.sum(1))/N,1e-9,1-1e-9)); tx[-1]=np.inf; tx=np.r_[-np.inf,tx]
    ty=norm.ppf(np.clip(np.cumsum(T.sum(0))/N,1e-9,1-1e-9)); ty[-1]=np.inf; ty=np.r_[-np.inf,ty]
    H0,K0=np.meshgrid(tx[:-1],ty[:-1],indexing="ij"); H1,K1=np.meshgrid(tx[1:],ty[1:],indexing="ij")
    def nll(r):
        p=bvn(H1.ravel(),K1.ravel(),r)-bvn(H0.ravel(),K1.ravel(),r)-bvn(H1.ravel(),K0.ravel(),r)+bvn(H0.ravel(),K0.ravel(),r)
        return -np.sum(T.ravel()*np.log(np.clip(p,1e-12,None)))
    return minimize_scalar(nll,bounds=(-0.995,0.995),method="bounded").x
df=pd.read_csv(sys.argv[1])
CHAR={"EF":["REG1","REG2","REG3","GRD1","SCH1","PAY1"],"EC":["REG4","REG5","REG6","GRD2","SCH2","SCH3","PAY3"],"ST":["REG7","GRD3","GRD4","SCH4","PAY2"],"CC":["REG8","GRD5","SCH5","PAY4"]}
items=[i for v in CHAR.values() for i in v]; X=df[items].astype(float)
print("category counts per item (1..5), min/max:",X.apply(lambda c:c.value_counts().reindex([1,2,3,4,5],fill_value=0).tolist()).T.min().min(),"smallest cell")
n=len(items); P=np.eye(n)
for i,j in itertools.combinations(range(n),2):
    P[i,j]=P[j,i]=polychoric_corr(X.iloc[:,i].values,X.iloc[:,j].values)
P=pd.DataFrame(P,index=items,columns=items); P.to_csv("polychoric_matrix.csv")
print("min eigenvalue polychoric:",np.linalg.eigvalsh(P.values).min().round(4))
def desc(groups): return "\n".join(f"{k} =~ "+" + ".join(v) for k,v in groups.items())
def run(name,groups,obj):
    m=Model(desc(groups)); 
    try:
        res=m.fit(cov=P,obj=obj,n_samples=len(df)); print(name,obj,'converged:',getattr(res,'success',None),'|',getattr(res,'message',''),'| n_it',getattr(res,'n_it',None))
    except TypeError:
        res=m.fit(cov=P,obj=obj); print(name,obj,'converged:',getattr(res,'success',None))
    ins=m.inspect(); 
    # SRMR on polychoric residuals
    S=m.calc_sigma()[0]; S=pd.DataFrame(S,index=m.vars['observed'] if 'observed' in m.vars else None)
    return m,ins
for obj in ["ULS"]:
    for name,g in [("4-factor",CHAR),("1-factor",{"G":items})]:
        try:
            m,ins=run(name,g,obj)
            Sg=np.asarray(m.calc_sigma()[0]); obs=m.vars['observed']; Sg=pd.DataFrame(Sg,index=obs,columns=obs).loc[items,items]
            R=P.values-Sg.values; iu=np.triu_indices(n,1)
            srmr=np.sqrt((R[iu]**2).mean()); mx=np.abs(R[iu]).max()
            st=calc_stats(m).iloc[0]
            print(f"\n[{obj}] {name}: SRMR(poly) {srmr:.3f}, max|resid| {mx:.3f}; chi2 {st.get('chi2',np.nan):.1f} df {st.get('DoF')} CFI {st.get('CFI',np.nan):.3f} TLI {st.get('TLI',np.nan):.3f} RMSEA {st.get('RMSEA',np.nan):.3f}  (naive, unscaled)")
            L=ins[(ins.op=="~")&(ins.rval.isin(g.keys()))] if (ins.op=="~").any() else ins[ins.op=="=~"]
            lam=ins[ins.rval.isin(g.keys())&ins.lval.isin(items)][["lval","rval","Estimate"]] if (ins.op=="~").any() else ins[ins.op=="=~"][["lval","rval","Estimate"]]
            fv=ins[(ins.op=="~~")&(ins.lval==ins.rval)&ins.lval.isin(g.keys())].set_index("lval").Estimate
            lam=lam.assign(std=lam.Estimate*np.sqrt(lam.rval.map(fv)))
            print("standardized loadings range:",lam["std"].min().round(3),"-",lam["std"].max().round(3),"| items with std loading >1:",int((lam["std"]>1).sum()))
            lam.to_csv(f"ordinal_loadings_{name}.csv",index=False)
            if name=="4-factor":
                cv=ins[(ins.op=="~~")&ins.lval.isin(g.keys())&ins.rval.isin(g.keys())&(ins.lval!=ins.rval)]
                var=ins[(ins.op=="~~")&(ins.lval==ins.rval)&ins.lval.isin(g.keys())].set_index("lval").Estimate
                for _,r in cv.iterrows():
                    print(f"  latent corr {r.lval}-{r.rval}: {r.Estimate/np.sqrt(var[r.lval]*var[r.rval]):.3f}")
                uv=ins[(ins.op=="~~")&(ins.lval==ins.rval)&ins.lval.isin(items)].Estimate
                print("  min residual variance (Heywood if <0):",uv.min().round(3))
        except Exception as e: print(obj,name,"failed",repr(e)[:200])
