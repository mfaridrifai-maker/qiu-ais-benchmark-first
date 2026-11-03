#!/usr/bin/env python3
"""
Reproduces the analyses reported in the manuscript
"A Benchmark-First Procedure for Checking Whether Clustering Adds Information to Satisfaction Scores in an Academic Information System".

Input : qiu_ais_survey_anonymized.csv  (N = 314, 22 items)
Output: results_summary.txt (also printed)

Usage : python analysis_qiu_clustering.py [--boot 200] [--skip-cfa]
Seeds are fixed (random_state = 42), so results are reproducible for the
library versions listed in requirements.txt. Bootstrap quantities can differ
slightly across scikit-learn / NumPy versions.
"""
import argparse, sys, io, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score)
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="qiu_ais_survey_anonymized.csv")
ap.add_argument("--boot", type=int, default=200)
ap.add_argument("--skip-cfa", action="store_true")
args = ap.parse_args()
SEED = 42
buf = io.StringIO()
def out(*a):
    s = " ".join(str(x) for x in a); print(s); buf.write(s + "\n")
def head(t): out("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)

df = pd.read_csv(args.data)
ITEMS = [c for c in df.columns if c[:3] in ("REG", "GRD", "SCH", "PAY") and c[3:].isdigit()]
assert len(ITEMS) == 22 and len(df) == 314
X = df[ITEMS].astype(float)
CHAR = {  # item -> ISO/IEC 25010 QiU characteristic the item was written for
 "EF": ["REG1","REG2","REG3","GRD1","SCH1","PAY1"],
 "EC": ["REG4","REG5","REG6","GRD2","SCH2","SCH3","PAY3"],
 "ST": ["REG7","GRD3","GRD4","SCH4","PAY2"],
 "CC": ["REG8","GRD5","SCH5","PAY4"]}
MOD = {"Course Registration": [f"REG{i}" for i in range(1,9)],
       "Grade Viewing": [f"GRD{i}" for i in range(1,6)],
       "Class Schedule": [f"SCH{i}" for i in range(1,6)],
       "Payment": [f"PAY{i}" for i in range(1,5)]}
comp = pd.DataFrame({k: X[v].mean(axis=1) for k, v in CHAR.items()})
overall = comp.mean(axis=1)            # overall QiU score = mean of the 4 characteristic composites
modsc = pd.DataFrame({k: X[v].mean(axis=1) for k, v in MOD.items()})
minmax = lambda a: (a - a.min()) / (a.max() - a.min())
Z = comp.apply(minmax).values          # clustering input (min-max normalised composites)

def cronbach(d):
    k = d.shape[1]; return k / (k - 1) * (1 - d.var(ddof=1).sum() / d.sum(axis=1).var(ddof=1))
def km(data, k, seed=SEED, n_init=10):
    return KMeans(n_clusters=k, init="k-means++", n_init=n_init, random_state=seed).fit(data)
def relabel(labels, score):             # 0 = Less, 1 = Moderately, 2 = Highly Satisfied (by group mean)
    order = pd.Series(score).groupby(labels).mean().sort_values().index
    m = {old: new for new, old in enumerate(order)}; return np.array([m[l] for l in labels])
GN = ["Less Satisfied", "Moderately Satisfied", "Highly Satisfied"]

# ---------------------------------------------------------------- 1. reliability
head("1. Reliability (Cronbach's alpha)")
for k, v in CHAR.items(): out(f"{k}: alpha = {cronbach(X[v]):.3f} ({len(v)} items)")
out(f"All 22 items: alpha = {cronbach(X):.3f}")
out(f"Mean overall QiU = {overall.mean():.2f}; item-mean = {X.values.mean():.2f}")

# ---------------------------------------------------------------- 2. dimensionality
head("2. Dimensionality")
R = np.corrcoef(X.values.T)
ev = np.sort(np.linalg.eigvalsh(R))[::-1]
out("Eigenvalues 1-3:", np.round(ev[:3], 3), f"| Factor 1 explains {ev[0]/22*100:.1f}% of item variance")
inv = np.linalg.inv(R); pc = -inv / np.sqrt(np.outer(np.diag(inv), np.diag(inv)))
np.fill_diagonal(pc, 0); r2 = R.copy(); np.fill_diagonal(r2, 0)
out(f"KMO = {(r2**2).sum()/((r2**2).sum()+(pc**2).sum()):.3f}")
def htmt(a, b):
    ia, ib = CHAR[a], CHAR[b]; idx = {c: i for i, c in enumerate(ITEMS)}
    hetero = np.mean([R[idx[i], idx[j]] for i in ia for j in ib])
    mono = lambda s: np.mean([R[idx[i], idx[j]] for n, i in enumerate(s) for j in s[n+1:]])
    return hetero / np.sqrt(mono(ia) * mono(ib))
pairs = [("EF","EC"),("EF","ST"),("EF","CC"),("EC","ST"),("EC","CC"),("ST","CC")]
h = [htmt(a, b) for a, b in pairs]
out("HTMT:", {f"{a}-{b}": round(v, 2) for (a, b), v in zip(pairs, h)}, f"| range {min(h):.2f} to {max(h):.2f}")

if not args.skip_cfa:
    try:
        import semopy
        def cfa(desc):
            m = semopy.Model(desc); m.fit(X, obj="MLW"); s = semopy.calc_stats(m).iloc[0]
            return s
        one = "G =~ " + " + ".join(ITEMS)
        ch4 = "\n".join(f"{k} =~ " + " + ".join(v) for k, v in CHAR.items())
        md4 = "\n".join(f"{k.replace(' ','_')} =~ " + " + ".join(v) for k, v in MOD.items())
        for name, d in [("One-factor", one), ("Four characteristics", ch4), ("Four modules", md4)]:
            s = cfa(d); out(f"CFA {name}: chi2 = {s['chi2']:.2f}, df = {int(s['DoF'])}, CFI = {s['CFI']:.3f}, TLI = {s['TLI']:.3f}, RMSEA = {s['RMSEA']:.3f}")
    except Exception as e:
        out("CFA skipped:", repr(e)[:120])

# ---------------------------------------------------------------- 3. choosing k
head("3. Number of clusters (min-max normalised composites)")
out("k  WCSS    Silhouette")
for k in range(2, 9):
    f = km(Z, k); out(f"{k}  {f.inertia_:6.2f}  {silhouette_score(Z, f.labels_):.4f}")
base = km(Z, 3); lab = relabel(base.labels_, overall.values)
out(f"k=3: Calinski-Harabasz = {calinski_harabasz_score(Z, lab):.1f}, Davies-Bouldin = {davies_bouldin_score(Z, lab):.3f}")
def stability(k, B):
    ref = km(Z, k).labels_; ar = []; R = np.random.RandomState(SEED)
    for b in range(B):
        i = R.randint(0, len(Z), len(Z)); fb = km(Z[i], k, seed=SEED)
        ar.append(adjusted_rand_score(ref, fb.predict(Z)))
    return np.mean(ar), np.std(ar)
for k in (2, 3, 4):
    m, s = stability(k, args.boot); out(f"Bootstrap stability k={k}: mean ARI = {m:.3f} (SD {s:.3f}, B = {args.boot})")

# ---------------------------------------------------------------- 4. groups
head("4. K-Means groups (k = 3)")
for g in range(3):
    m = lab == g
    out(f"{GN[g]:21s} n = {m.sum():3d} ({m.mean()*100:4.1f}%)  mean = {overall[m].mean():.2f}  range {overall[m].min():.2f}-{overall[m].max():.2f}")
b1 = (overall[lab == 0].max() + overall[lab == 1].min()) / 2
b2 = (overall[lab == 1].max() + overall[lab == 2].min()) / 2
out(f"Boundaries (midpoints between adjacent group extremes): {b1:.2f} and {b2:.2f}")
H, p = stats.kruskal(*[overall[lab == g] for g in range(3)])
out(f"Kruskal-Wallis on overall score (circular, descriptive only): H = {H:.1f}, p = {p:.2e}")

# ---------------------------------------------------------------- 5. benchmarking
head("5. Benchmarking against simpler comparators")
def row(name, l, stab=None):
    s = silhouette_score(Z, l); a = adjusted_rand_score(lab, l)
    out(f"{name:34s} Silhouette = {s:.3f}  ARI vs K-Means = {a:.3f}" + (f"  stability = {stab:.3f}" if stab is not None else ""))
terc = pd.qcut(overall.rank(method="first"), 3, labels=False).values; row("Equal-sized tertiles (overall)", terc)
row("K-Means on overall score only", relabel(km(overall.values.reshape(-1,1), 3).labels_, overall.values))
fixed = np.digitize(overall.values, [3.5, 4.5]); row("Fixed cut points 3.5 / 4.5", fixed)
hier = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(Z); row("Hierarchical (Ward)", hier)
def fcm(data, c=3, m=2.0, iters=300, seed=SEED):
    r = np.random.default_rng(seed); u = r.random((len(data), c)); u /= u.sum(1, keepdims=True)
    for _ in range(iters):
        um = u ** m; cen = um.T @ data / um.sum(0)[:, None]
        d = np.linalg.norm(data[:, None, :] - cen[None], axis=2) + 1e-12
        nu = 1 / (d ** (2 / (m - 1)) * np.sum(d ** (-2 / (m - 1)), axis=1, keepdims=True))
        if np.abs(nu - u).max() < 1e-6: u = nu; break
        u = nu
    return u.argmax(1)
row("Fuzzy c-means", fcm(Z))
gmm = GaussianMixture(3, covariance_type="full", n_init=10, random_state=SEED).fit(Z); row("Gaussian mixture", gmm.predict(Z))
out(f"Agreement of fixed cut points with K-Means: {(fixed == lab).sum()} of 314 ({(fixed == lab).mean()*100:.1f}%)")
less = lab == 0
out(f"Rule 'overall < 3.5' flags {(overall < 3.5).sum()} students; captures {((overall < 3.5) & less).sum()} of {less.sum()} Less Satisfied ({((overall < 3.5) & less).sum()/less.sum()*100:.1f}%)")
k_ = (overall < 3.5).sum(); lo, hi = stats.binomtest(int(k_), 314).proportion_ci(method="wilson")
out(f"Share below 3.5 = {k_/314*100:.1f}% (Wilson 95% CI {lo*100:.1f} to {hi*100:.1f})")

head("5b. Threshold sensitivity (bootstrap, K-Means on overall score)")
B1, B2, SH = [], [], []; R = np.random.RandomState(SEED)
for b in range(args.boot):
    i = R.randint(0, 314, 314); s = overall.values[i]
    f = KMeans(3, n_init=10, random_state=SEED).fit(s.reshape(-1, 1)); l = relabel(f.labels_, s)
    B1.append((s[l == 0].max() + s[l == 1].min()) / 2); B2.append((s[l == 1].max() + s[l == 2].min()) / 2); SH.append((l == 0).mean())
q = lambda a: np.percentile(a, [2.5, 97.5])
out(f"Boundary 1: 95% interval {q(B1)[0]:.2f} to {q(B1)[1]:.2f}; Boundary 2: {q(B2)[0]:.2f} to {q(B2)[1]:.2f}; Less Satisfied share {q(SH)[0]*100:.1f}% to {q(SH)[1]*100:.1f}%")

# ---------------------------------------------------------------- 6. modules / lower tail
head("6. Module scores by group and lower tail")
tab = modsc.groupby(lab).mean().round(2); tab.index = GN; out(tab.T.to_string())
d = modsc.loc[less, "Grade Viewing"] - modsc.loc[less, "Payment"]
out(f"Less Satisfied: Grade Viewing - Payment = {d.mean():.2f}; Wilcoxon p = {stats.wilcoxon(modsc.loc[less,'Grade Viewing'], modsc.loc[less,'Payment']).pvalue:.2e}")
out("Less Satisfied item means (Grade): ", X.loc[less, MOD["Grade Viewing"]].mean().round(2).to_dict())
out("Less Satisfied item means (Payment):", X.loc[less, MOD["Payment"]].mean().round(2).to_dict())
lowest = modsc.loc[less].idxmin(axis=1).value_counts(normalize=True).mul(100).round(0)
out("Lowest module among Less Satisfied (%):", lowest.to_dict())
out("Problems reported by Less Satisfied (n): slow loading", int(df.loc[less, 'problem_slow_loading'].sum()),
    "| errors", int(df.loc[less, 'problem_errors_crash'].sum()), "| confusing", int(df.loc[less, 'problem_confusing'].sum()))

# ---------------------------------------------------------------- 7. ceiling effect
head("7. Ceiling effect")
all5 = (X == 5).all(axis=1)
out(f"Respondents with all 22 items = 5: {all5.sum()} ({all5.mean()*100:.1f}%); all in Highly Satisfied: {bool((lab[all5.values] == 2).all())}; share of that group: {all5.sum()/(lab==2).sum()*100:.1f}%")
Zr = Z[~all5.values]; fr = km(Zr, 3)
out(f"Without them (n = {len(Zr)}): Silhouette = {silhouette_score(Zr, fr.labels_):.3f}; ARI vs original = {adjusted_rand_score(lab[~all5.values], fr.labels_):.2f}")

# ---------------------------------------------------------------- 8. external check
head("8. External check (bias-corrected Cramer's V)")
def cramer(a, b):
    t = pd.crosstab(a, b); chi2 = stats.chi2_contingency(t, correction=False)[0]; n = t.values.sum(); r, k = t.shape
    phi2 = max(0, chi2 / n - (k - 1) * (r - 1) / (n - 1)); rc = r - (r - 1) ** 2 / (n - 1); kc = k - (k - 1) ** 2 / (n - 1)
    return np.sqrt(phi2 / max(1e-12, min(kc - 1, rc - 1))), stats.chi2_contingency(t)[1]
for c in ["program_group", "semester_group", "gender", "use_frequency"]:
    v, p = cramer(df[c], lab); out(f"{c:16s} V = {v:.2f}, p = {p:.2f}")
out("(Categories are the grouped ones in the shared file; the manuscript's V values used the original, finer categories.)")

# ---------------------------------------------------------------- 9. exploratory shape check
head("9. Exploratory check of module-specific response patterns")
S = modsc.values - modsc.values.mean(axis=1, keepdims=True)
def silk(Sx, k=3):
    P = PCA(3).fit_transform(Sx); return silhouette_score(P, km(P, k).labels_)
obs = silk(S); nv = []; ns = []; rng = np.random.default_rng(7)
for b in range(args.boot):
    Xs = np.array([rng.permutation(r) for r in X.values])
    Ms = np.column_stack([Xs[:, [ITEMS.index(c) for c in v]].mean(1) for v in MOD.values()])
    Sn = Ms - Ms.mean(1, keepdims=True); nv.append(Sn.var(0).sum()); ns.append(silk(Sn))
out(f"Total pattern variance: observed {S.var(0).sum():.3f} vs shuffled mean {np.mean(nv):.3f} (95th pct {np.percentile(nv,95):.3f})")
out(f"Silhouette (k=3): observed {obs:.3f} vs shuffled mean {np.mean(ns):.3f}; p = {(np.array(ns) >= obs).mean():.3f}")
out(f"Respondents with identical scores on all four modules: {(np.abs(S).max(1) < 1e-9).sum()}")

open("results_summary.txt", "w", encoding="utf-8").write(buf.getvalue())
