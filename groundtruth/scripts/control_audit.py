"""Control-effect audit of the shipped models. Free: no gateway calls.

For each system and each control:
  model authority: hold the mid-range action with one control at its lower vs upper bound for 300
      ticks from the median reset state; mean |dY| over ticks 100-300 in calibrated sigma per obs
      (0 = the model ignores the control).
  data events: every tick where some control changes (tick 0 counts as a change from the recovery
      action). Window = [t, min(next change, t+W)). Per window: lost score-ticks sum_t mean_obs
      (1 - 1/(1+|e|/sigma)), excess over the 20 ticks before the change (same run), mean signed
      residual z = (Y - Yhat)/sigma, observed step dY_obs and predicted step dY_hat (window mean
      minus the pre-change mean, sigma units). Loss is attributed to the controls that moved in
      proportion to |du| / range. Isolated events (one control moved) are listed separately.
Rank: attributed excess lost score-ticks, per control.

    python scripts/control_audit.py [--systems a,b] [--sub submissions/20260928-1440-final1] [--W 60]
Writes plans/control_audit.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import systems as S  # noqa: E402
from gtlab.ledger import Ledger, data_dir, load_runs  # noqa: E402

warnings.filterwarnings("ignore")
SYSTEMS = ["epidemic", "market", "traffic", "power_grid", "supply_chain", "wildlife", "reservoir",
           "ad_auction", "social_contagion", "hospital_queue"]


def load_pred(sub, sid):
    p = ROOT / sub / sid / "predict.py"
    sp = importlib.util.spec_from_file_location(f"ca_{sid}", str(p))
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def roll(mod, spec, y0, U):
    obs, ctrls = list(spec.observables), list(spec.controls)
    out = mod.predict({o: float(v) for o, v in zip(obs, y0)},
                      [{c: float(v) for c, v in zip(ctrls, r)} for r in U], spec.context())
    return np.array([[row[o] for o in obs] for row in out], float)


def audit(sid, sub, W, pre=20):
    spec = S.get(sid)
    obs, ctrls = list(spec.observables), list(spec.controls)
    sig = np.array(json.loads((ROOT / "plans" / "sigma_calibrated.json").read_text())[sid]["sigma"], float)
    runs = load_runs(spec, Ledger(data_dir(sid, False), sid, False))
    mod = load_pred(sub, sid)
    lo, hi = np.array(spec.lo(), float), np.array(spec.hi(), float)
    rngc = hi - lo
    rec = np.array([spec.recovery[c] for c in ctrls], float)
    y0m = np.median(np.array([r.y0 for r in runs]), axis=0)
    # model authority
    auth = {}
    for j, c in enumerate(ctrls):
        mid = 0.5 * (lo + hi)
        Ua, Ub = np.tile(mid, (300, 1)), np.tile(mid, (300, 1))
        Ua[:, j], Ub[:, j] = lo[j], hi[j]
        d = (roll(mod, spec, y0m, Ub) - roll(mod, spec, y0m, Ua))[100:] / sig
        auth[c] = np.mean(np.abs(d), axis=0).round(3).tolist()
    # events
    events, tot_loss = [], 0.0
    preds = [roll(mod, spec, r.y0, r.U) for r in runs]
    for r, Yh in zip(runs, preds):
        e = (r.Y - Yh) / sig
        loss = np.mean(1.0 - 1.0 / (1.0 + np.abs(e)), axis=1)
        tot_loss += float(loss.sum())
        U = np.asarray(r.U, float)
        prev = np.vstack([rec[None, :], U[:-1]])
        chg = np.where((np.abs(U - prev) > 1e-9 * (1 + np.abs(U))).any(1))[0]
        for k, t in enumerate(chg):
            t2 = min(chg[k + 1] if k + 1 < len(chg) else r.T, t + W)
            du = (U[t] - prev[t]) / rngc
            if t >= pre:
                base_loss = float(loss[t - pre:t].mean())
                yo_pre, yh_pre = r.Y[t - pre:t].mean(0), Yh[t - pre:t].mean(0)
            else:
                base_loss = float(np.median(loss))  # tick-0 change: no pre window, use run median
                yo_pre, yh_pre = r.y0, r.y0
            wl = loss[t:t2]
            events.append({"run": r.exp, "t": int(t), "len": int(t2 - t), "du": du.round(3).tolist(),
                           "moved": [ctrls[j] for j in np.where(np.abs(du) > 1e-9)[0]],
                           "lost": float(wl.sum()), "excess": float((wl - base_loss).sum()),
                           "z_mean": e[t:t2].mean(0).round(2).tolist(),
                           "z_early": e[t:min(t2, t + 15)].mean(0).round(2).tolist(),
                           "dy_obs": ((r.Y[t:t2].mean(0) - yo_pre) / sig).round(2).tolist(),
                           "dy_hat": ((Yh[t:t2].mean(0) - yh_pre) / sig).round(2).tolist()})
    per = {}
    for c in ctrls:
        j = ctrls.index(c)
        per[c] = {"authority": auth[c], "n_events": 0, "n_isolated": 0, "lost_attr": 0.0, "excess_attr": 0.0,
                  "iso": []}
    for ev in events:
        a = np.abs(np.array(ev["du"]))
        if a.sum() == 0:
            continue
        w = a / a.sum()
        for j, c in enumerate(ctrls):
            if w[j] > 0:
                per[c]["n_events"] += 1
                per[c]["lost_attr"] += float(w[j] * ev["lost"])
                per[c]["excess_attr"] += float(w[j] * max(ev["excess"], 0.0))
        if len(ev["moved"]) == 1:
            c = ev["moved"][0]
            per[c]["n_isolated"] += 1
            per[c]["iso"].append({k: ev[k] for k in ("run", "t", "len", "du", "lost", "excess", "z_early", "z_mean",
                                                     "dy_obs", "dy_hat")})
    recov = recoverable(runs, preds, sig, lo, rngc, ctrls)
    for c in ctrls:
        per[c]["recoverable"] = recov[c]
    return {"system": sid, "observables": obs, "recoverable_intercept": recov["_intercept"], "controls": ctrls, "sigma": sig.tolist(),
            "total_lost": tot_loss, "total_ticks": int(sum(r.T for r in runs)), "per_control": per,
            "events": events}


def _feats(U, lo, rngc, j, taus=(8.0, 40.0)):
    """Control j normalised to [0,1], plus its exponential moving averages (memory of past moves)."""
    u = (np.asarray(U, float)[:, j] - lo[j]) / rngc[j]
    cols = [u]
    for tau in taus:
        a, z, out = 1.0 / tau, u[0], np.empty_like(u)
        for t, v in enumerate(u):
            z = z + a * (v - z)
            out[t] = z
        cols.append(out)
    return np.stack(cols, 1)


def recoverable(runs, preds, sig, lo, rngc, ctrls, lam=1.0):
    """Leave-one-run-out: residual (sigma units) ~ ridge on [1, u_j, EMA8(u_j), EMA40(u_j)] per observable.
    Returns, per control, the held-out score gain (score-ticks and per tick) of the corrected prediction
    minus the gain of the intercept alone."""
    E = [(r.Y - Yh) / sig for r, Yh in zip(runs, preds)]
    base = sum(float(np.sum(np.mean(1 / (1 + np.abs(e)), 1))) for e in E)
    nt = sum(r.T for r in runs)

    def gain(fe):
        if len(runs) < 2:
            return 0.0
        tot = 0.0
        for k in range(len(runs)):
            X = np.concatenate([fe(i) for i in range(len(runs)) if i != k])
            Yr = np.concatenate([E[i] for i in range(len(runs)) if i != k])
            A = X.T @ X + lam * np.eye(X.shape[1]) * len(X) * 1e-3
            A[0, 0] -= lam * len(X) * 1e-3
            B = np.linalg.solve(A, X.T @ Yr)
            e2 = E[k] - fe(k) @ B
            tot += float(np.sum(np.mean(1 / (1 + np.abs(e2)), 1)))
        return tot - base

    ones = lambda i: np.ones((runs[i].T, 1))
    g0 = gain(ones)
    out = {"_intercept": {"score_ticks": g0, "per_tick": g0 / nt}}
    for j, c in enumerate(ctrls):
        g = gain(lambda i, j=j: np.hstack([ones(i), _feats(runs[i].U, lo, rngc, j)])) - g0
        out[c] = {"score_ticks": g, "per_tick": g / nt}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--sub", default="submissions/20260928-1440-final1")
    ap.add_argument("--W", type=int, default=60)
    ap.add_argument("--out", default="plans/control_audit.json")
    a = ap.parse_args()
    res = {}
    for sid in a.systems.split(","):
        R = audit(sid, a.sub, a.W)
        res[sid] = R
        print(f"\n== {sid}: in-sample lost {R['total_lost']:.1f} over {R['total_ticks']} ticks "
              f"(score {1 - R['total_lost'] / R['total_ticks']:.3f}) obs {R['observables']}")
        for c, p in sorted(R["per_control"].items(), key=lambda kv: -kv[1]["excess_attr"]):
            print(f"  {c:<22} auth {p['authority']}  events {p['n_events']:>2} iso {p['n_isolated']:>2} "
                  f"lost {p['lost_attr']:7.1f} excess {p['excess_attr']:6.1f} recov/tick {p['recoverable']['per_tick']:+.4f}")
            for ev in p["iso"]:
                print(f"      iso {ev['run']:<22} t{ev['t']:>3} len {ev['len']:>3} du {ev['du']} exc {ev['excess']:6.1f} "
                      f"z_early {ev['z_early']} obs {ev['dy_obs']} hat {ev['dy_hat']}")
    Path(a.out).write_text(json.dumps(res, indent=1))
    rows = []
    for sid, R in res.items():
        for c, p in R["per_control"].items():
            rows.append((p["excess_attr"] / R["total_ticks"], sid, c, p["recoverable"]["per_tick"], p["authority"]))
    print("\nRANK (attributed excess lost score per system tick; recov = held-out gain of a linear control correction)")
    for x in sorted(rows, reverse=True)[:30]:
        print(f"  {x[1]:<17} {x[2]:<22} excess/tick {x[0]:.4f}  recov/tick {x[3]:+.4f}  authority {x[4]}")


if __name__ == "__main__":
    main()
