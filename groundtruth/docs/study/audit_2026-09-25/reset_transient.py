import numpy as np, warnings
warnings.filterwarnings("ignore")
from gtlab import systems as S
from gtlab.ledger import Ledger, load_runs, data_dir
np.set_printoptions(precision=2, suppress=True, linewidth=220)
# 1. reservoir inflow: deterministic in t across runs?
spec = S.get("reservoir"); runs = load_runs(spec, Ledger(data_dir("reservoir", False), "reservoir", False))
j = spec.observables.index("inflow")
a, b = runs[0].Y[:120, j], runs[1].Y[:120, j]
print("inflow run0 vs run1 first 120 ticks: corr=%.3f  mean|diff|=%.3f  sd(run0)=%.3f" % (np.corrcoef(a, b)[0, 1], np.mean(np.abs(a - b)), a.std()))
print("inflow run0 every 10:", np.round(a[::10], 2)); print("inflow run1 every 10:", np.round(b[::10], 2))
x = runs[1].Y[:, j]; t = np.arange(len(x))
# fit sinusoid grid over period
best = None
for P in np.arange(30, 1200, 1.0):
    X = np.column_stack([np.ones_like(t), np.sin(2*np.pi*t/P), np.cos(2*np.pi*t/P)])
    c, res, *_ = np.linalg.lstsq(X, x, rcond=None); r = x - X @ c
    if best is None or r.var() < best[0]: best = (r.var(), P, c)
print("best period P=%.0f  resid sd=%.3f (raw sd %.3f) coef=%s" % (best[1], np.sqrt(best[0]), x.std(), np.round(best[2], 2)))
print("run1 inflow every 20:", np.round(x[::20], 2))
# 2. reset transient determinism: for each system, corr of Y[:20] between runs regardless of y0 (only where controls equal in first ticks)
for sid in S.SYSTEM_IDS:
    spec = S.get(sid); runs = load_runs(spec, Ledger(data_dir(sid, False), sid, False))
    out = []
    for jj, o in enumerate(spec.observables):
        y0s = [r.y0[jj] for r in runs]; y5 = [r.Y[4, jj] for r in runs]; y20 = [r.Y[19, jj] for r in runs]
        out.append(f"{o}: y0 {np.round(y0s,1)} -> t5 {np.round(y5,1)} -> t20 {np.round(y20,1)}")
    print(f"-- {sid}\n   " + "\n   ".join(out))
