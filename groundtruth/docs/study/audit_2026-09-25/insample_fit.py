import numpy as np, json, warnings
warnings.filterwarnings("ignore")
from gtlab import systems as S, metric, select as SEL, design as D
from gtlab.ledger import Ledger, load_runs, data_dir
from gtlab.models import common as C
np.set_printoptions(precision=3, suppress=True, linewidth=220)
for sid in S.SYSTEM_IDS:
    spec = S.get(sid)
    runs = load_runs(spec, Ledger(data_dir(sid, False), sid, False))
    clip = C.soft_clip(spec, runs); sigma = metric.sigma_proxy(runs)
    m = SEL.make_model("l0b", spec, clip, sigma, cfg={"sq": False, "pairs": None}); m.fit(runs)
    print(f"\n===== {sid}  sigma_proxy={np.round(sigma,3)}  a={np.round(m.a,4)}  tau={np.round(1/(1-m.a),1)}  tr={m.tr['kind']}")
    print(f"   clip lo={np.round(clip[0],2)} hi={np.round(clip[1],2)}")
    W = m.W
    print("   W (rows: controls + const; cols: obs) in transformed units:\n" + "\n".join("     " + str(np.round(W[i],2)) + ("  " + (spec.controls[i] if i < spec.m else "const")) for i in range(W.shape[0])))
    for r in runs:
        Yh = m.rollout(r.y0, r.U); Yp = np.tile(r.y0, (r.T, 1))
        sm = metric.score_per_obs(Yh, r.Y, sigma); sp = metric.score_per_obs(Yp, r.Y, sigma)
        e = np.abs(Yh - r.Y)
        early = np.mean(1/(1+e[:50]/sigma), axis=0); late = np.mean(1/(1+e[50:]/sigma), axis=0) if r.T > 50 else early*np.nan
        print(f"   {r.exp:<20} T={r.T:<4} model={np.round(sm,2)} pers={np.round(sp,2)}  early50={np.round(early,2)} late={np.round(late,2)}  mean|e|={np.round(e.mean(0),2)}")
    # eval-like 4000 rollouts
    rng = np.random.default_rng(1)
    y0 = runs[0].y0
    for cat in ["sustained", "order", "recovery", "composition"]:
        U = D.eval_like(spec, cat, 4000, rng)
        Yh = m.rollout(y0, U)
        lo, hi = np.array(clip[0]), np.array([np.inf if h is None else h for h in clip[1]])
        atlo = np.mean(np.isclose(Yh, lo[None,:]), axis=0); athi = np.mean(np.isclose(Yh, hi[None,:]), axis=0)
        print(f"   eval[{cat:<11}] min={np.round(Yh.min(0),1)} max={np.round(Yh.max(0),1)} end={np.round(Yh[-1],1)} frac@lo={np.round(atlo,2)} frac@hi={np.round(athi,2)}")
