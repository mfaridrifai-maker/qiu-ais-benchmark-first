#!/usr/bin/env python3
"""Calibration and validation of the ARI threshold tau on the extended simulation (sim_extended.py).
Configurations are split at random (seed 7) into a calibration half (used only to choose tau) and a validation half (used only to assess tau).
Loss(tau) = 0.5 * mean false-structure rate over level-only configurations (classes A, B) + 0.5 * mean missed-profile rate over relevant-profile configurations (class C, gap >= 0.6).
Class D (weak gap 0.2-0.4 or continuous residual factor) is reported but never used for choosing tau.
Usage: python calibrate_tau.py [--raw sim_extended_raw.csv] [--cfg sim_extended_configs.csv]"""
import argparse, numpy as np, pandas as pd
from scipy import stats
ap=argparse.ArgumentParser(); ap.add_argument("--raw",default="sim_extended_raw.csv"); ap.add_argument("--cfg",default="sim_extended_configs.csv"); a=ap.parse_args()
raw=pd.read_csv(a.raw); cfg=pd.read_csv(a.cfg); d=raw.merge(cfg,on="cid")
rng=np.random.default_rng(7); cfg["half"]=""
for c in "ABCD":
    ids=cfg.index[cfg.cls==c].to_numpy(); rng.shuffle(ids); h=len(ids)//2; cfg.loc[ids[:h],"half"]="calibration"; cfg.loc[ids[h:],"half"]="validation"
d=d.merge(cfg[["cid","half"]],on="cid")
TAUS=[round(t,2) for t in np.arange(0.50,0.991,0.05)]+[0.80]; TAUS=sorted(set(TAUS))
def rates(df,tau,kr):
    x=df[df.krule==kr].copy(); x["thr"]=x.ari>=tau
    per=x.groupby(["cid","cls"]).thr.mean().reset_index()   # P(called thresholds) per config
    fs=1-per[per.cls.isin(["A","B"])].thr; mp=per[per.cls=="C"].thr
    return fs.mean(), mp.mean(), per
def loss(df,tau,kr):
    fs,mp,_=rates(df,tau,kr); return 0.5*fs+0.5*mp,fs,mp
print("== loss by tau (k_rule=3), calibration half")
rows=[]
for kr in (2,3,4):
    for t in TAUS:
        l,fs,mp=loss(d[d.half=="calibration"],t,kr); lv,fsv,mpv=loss(d[d.half=="validation"],t,kr); rows.append(dict(krule=kr,tau=t,loss_cal=l,fs_cal=fs,mp_cal=mp,loss_val=lv,fs_val=fsv,mp_val=mpv))
R=pd.DataFrame(rows); R.round(3).to_csv("tau_grid_results.csv",index=False)
print(R[R.krule==3].round(3).to_string(index=False))
best={kr:R[R.krule==kr].sort_values("loss_cal").iloc[0] for kr in (2,3,4)}
for kr,b in best.items(): print(f"k_rule={kr}: tau* (calibration) = {b.tau:.2f}; validation loss {b.loss_val:.3f} (FS {b.fs_val:.3f}, MP {b.mp_val:.3f}); tau=0.80 validation loss {R[(R.krule==kr)&(R.tau==0.80)].loss_val.iloc[0]:.3f}")
# config-level bootstrap CI on validation metrics for tau* and 0.80 (k_rule=3)
def boot(tau,kr=3,B=2000):
    x=d[(d.half=="validation")&(d.krule==kr)].copy(); x["thr"]=x.ari>=tau
    per=x.groupby(["cid","cls"]).thr.mean().reset_index(); fsv=1-per[per.cls.isin(["A","B"])].thr.to_numpy(); mpv=per[per.cls=="C"].thr.to_numpy(); r=np.random.default_rng(11)
    f=[r.choice(fsv,len(fsv)).mean() for _ in range(B)]; m=[r.choice(mpv,len(mpv)).mean() for _ in range(B)]
    return fsv.mean(),np.percentile(f,[2.5,97.5]),mpv.mean(),np.percentile(m,[2.5,97.5])
for tau in sorted({best[3].tau,0.80,0.90}):
    fs,fci,mp,mci=boot(tau); print(f"validation, k=3, tau={tau:.2f}: false-structure {fs*100:.1f}% (config-bootstrap 95% CI {fci[0]*100:.1f}-{fci[1]*100:.1f}); missed-profile {mp*100:.1f}% ({mci[0]*100:.1f}-{mci[1]*100:.1f})")
# pooled Monte Carlo Wilson intervals (replications pooled within class), validation half
print("== pooled Monte Carlo (replication-level) rates, validation half, k_rule=3")
for tau in sorted({best[3].tau,0.80}):
    for grp,lab in (("AB","level-only (false structure = ARI<tau)"),("C","relevant profile (missed = ARI>=tau)")):
        x=d[(d.half=="validation")&(d.krule==3)&(d.cls.isin(list(grp)))]; k=(x.ari<tau).sum() if grp=="AB" else (x.ari>=tau).sum(); n=len(x); ci=stats.binomtest(int(k),n).proportion_ci(method="wilson")
        print(f"tau={tau:.2f} {lab}: {k/n*100:.1f}% (Wilson {ci.low*100:.1f}-{ci.high*100:.1f}, n={n})")
# by class / nuisance factor at tau=0.80 and tau*, validation + calibration pooled
print("== error by class and by gap, k_rule=3, all configs, tau = 0.80 and tau*")
x=d[d.krule==3]
for tau in sorted({best[3].tau,0.80}):
    x2=x.assign(thr=x.ari>=tau)
    print("tau",tau); print((x2.groupby("cls").thr.mean().rename("P(called thresholds)")).round(3).to_string())
    c=x2[x2.cls=="C"].groupby("gap").thr.mean().round(3); print("class C by gap (missed-profile rate):"); print(c.to_string())
    dd=x2[x2.cls=="D"].assign(w=lambda z: np.where(z.gap>0,"gap "+z.gap.astype(str),"rho "+z.rho.astype(str))).groupby("w").thr.mean().round(3); print("class D (weak/continuous) P(called thresholds):"); print(dd.to_string())
# k mismatch for class B (true kG vs rule k)
b=d[d.cls=="B"].assign(match=lambda z:z.krule==z.kG,thr=lambda z:z.ari>=0.80)
print("== class B (level-only groups), P(thresholds) at tau=0.80 by matched/mismatched k_rule"); print(b.groupby("match").thr.mean().round(3).to_string())
# residual-profile test and composite PA by class (per replication)
r=d[d.krule==3].groupby("cls")[["resid_flag","pa_flag"]].mean().round(3); print("== P(flag) by class: residual-profile test and composite parallel analysis"); print(r.to_string())
rc=d[(d.krule==3)&(d.cls=="D")].assign(w=lambda z: np.where(z.gap>0,"gap "+z.gap.astype(str),"rho "+z.rho.astype(str))).groupby("w")[["resid_flag","pa_flag"]].mean().round(3); print(rc.to_string())
rc=d[(d.krule==3)&(d.cls=="C")].groupby("gap")[["resid_flag","pa_flag"]].mean().round(3); print(rc.to_string())
# monotonicity: mean label changes along overall score for composite partition (k=3) by class; ideal = k-1 = 2
print("== mean label changes along the overall score (k_rule=3; 2 = monotone/interval partition)"); print(d[d.krule==3].groupby("cls").mono_changes.mean().round(2).to_string())
print("== share of replications with ARI<tau among class A/B by nuisance factors (tau=0.80, k=3)")
for f in ("n","sigma_item","s_theta","kappa","base","sigma_c","het","sigma_mod","props"):
    z=d[(d.krule==3)&(d.cls.isin(["A","B"]))].assign(fs=lambda q:q.ari<0.80).groupby(f).fs.mean().round(3); print(f,z.to_dict())
