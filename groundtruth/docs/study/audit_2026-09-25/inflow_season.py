import numpy as np, warnings
warnings.filterwarnings("ignore")
from gtlab import systems as S, metric
from gtlab.ledger import Ledger, load_runs, data_dir
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def sinfit(t, x, P, nh=2):
    cols = [np.ones_like(t, float)]
    for h in range(1, nh + 1):
        cols += [np.sin(2*np.pi*h*t/P), np.cos(2*np.pi*h*t/P)]
    X = np.column_stack(cols); c = np.linalg.lstsq(X, x, rcond=None)[0]; return c, X @ c
# reservoir inflow: fit on run1 (400), test on run0 (120)
spec = S.get("reservoir"); runs = load_runs(spec, Ledger(data_dir("reservoir", False), "reservoir", False))
sig = metric.sigma_proxy(runs); j = spec.observables.index("inflow")
x1 = runs[1].Y[:, j]; t1 = np.arange(len(x1)); x0 = runs[0].Y[:, j]; t0 = np.arange(len(x0))
for P in np.arange(60, 76, 0.5):
    c, f = sinfit(t1, x1, P, 2); r = x1 - f
    print(f"P={P:5.1f} resid_sd={r.std():.3f}", end=" | ")
print()
P = 68.0
best = min(((sinfit(t1, x1, P, 2)[1] - x1).std(), P) for P in np.arange(60, 76, 0.1))[1]
c, f1 = sinfit(t1, x1, best, 2); _, f0 = sinfit(t0, x0, best, 2)
f0pred = sinfit(t1, x1, best, 2)[0]; X0 = np.column_stack([np.ones_like(t0,float)] + sum(([np.sin(2*np.pi*h*t0/best), np.cos(2*np.pi*h*t0/best)] for h in (1,2)), []))
pred0 = X0 @ c
s_sin = np.mean(1/(1+np.abs(pred0-x0)/sig[j])); s_const = np.mean(1/(1+np.abs(x1.mean()-x0)/sig[j])); s_pers = np.mean(1/(1+np.abs(runs[0].y0[j]-x0)/sig[j]))
print(f"best P={best:.1f} coef={np.round(c,3)}; run0 out-of-sample: sinusoid {s_sin:.3f}  const {s_const:.3f}  persistence {s_pers:.3f}  (sigma_proxy {sig[j]:.2f}); resid sd on run0 {np.std(pred0-x0):.3f}")
print("also level check: dlevel vs inflow-outflow, run1 t<200:", np.round(np.corrcoef(np.diff(runs[1].Y[:200,0]), (runs[1].Y[1:200,1]-runs[1].Y[1:200,2]))[0,1],3))
# periodogram of hold_rec tails, all systems: dominant period and its power fraction after removing a linear trend
print("\n-- periodic content in every observable (longest constant-control segment, detrended):")
for sid in S.SYSTEM_IDS:
    spec = S.get(sid); runs = load_runs(spec, Ledger(data_dir(sid, False), sid, False))
    for r in runs:
        # longest constant-u segment
        ch = np.flatnonzero(np.any(np.diff(r.U, axis=0) != 0, axis=1)) + 1
        bounds = [0] + list(ch) + [r.T]; segs = [(a, b) for a, b in zip(bounds[:-1], bounds[1:])]
        a, b = max(segs, key=lambda s: s[1]-s[0])
        if b - a < 100: continue
        seg = r.Y[a+30:b]  # drop 30-tick transient
        if len(seg) < 64: continue
        t = np.arange(len(seg)); out = []
        for jj, o in enumerate(spec.observables):
            x = seg[:, jj]; x = x - np.polyval(np.polyfit(t, x, 1), t)
            if x.std() < 1e-9: out.append(f"{o}:flat"); continue
            F = np.abs(np.fft.rfft(x))**2; F[0] = 0; k = np.argmax(F); frac = F[k] / F.sum()
            out.append(f"{o}: P={len(seg)/max(k,1):.0f} pow={frac:.2f} amp={x.std():.2f}")
        print(f"  {sid:<17} {r.exp:<18} seg[{a+30}:{b}] " + " | ".join(out))
