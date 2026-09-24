"""Fit the model ladder, validate, stress-gate, and pick (PLAN section 5, "Selection").

    pick = fit_and_select(system_id, runs, out_dir, models=("l0a","l0b","l1","l2","ode"),
                          time_budget_s=1800)

- train = runs tagged split != "val"; val = split == "val" (if none, every 5th run except the
  longest is held out).
- sigma = metric.sigma_proxy(all runs); clip range = soft range of all runs.
- score = metric.robust_score on concatenated val runs (noisy Y; also truth if Ytrue exists).
- stress gate: eval-like 4000-step schedules from pool initial conditions; reject nonfinite raw
  outputs, raw outputs beyond data range +- gate_margin*range (a blow-up detector: true 4000-step
  trajectories of integrating observables legitimately leave the short-run data range several x)
  on > max_out_frac of ticks, or a shipped-runtime rollout slower than max_ms_per_step (2 ms).
  worst_softclip_frac (time spent at the soft clip) is reported for information.
- candidates: passing ladder models + median ensemble of the top 3 + per-observable pick.
- choose best robust score; within `tie` prefer the simpler model (ladder order).
Writes out_dir/<model_id>/model.json for each candidate, out_dir/leaderboard.json, and
out_dir/pick/model.json (refit on all runs when refit_all=True). Returns the pick dict.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np

from gtlab import metric
from gtlab import systems as S
from gtlab.models import common as C
from gtlab.runtime import infer as rt

LADDER = ("l0a", "l0b", "l1", "l2", "ode", "ens3", "perobs")
DEFAULT_SHARES = {"l0a": 0.0, "l0b": 0.02, "l1": 0.2, "l2": 0.25, "ode": 0.5}


def split_runs(runs):
    train = [r for r in runs if r.tags.get("split") != "val"]
    val = [r for r in runs if r.tags.get("split") == "val"]
    if val or len(runs) < 2:
        return train, val
    longest = max(range(len(runs)), key=lambda i: runs[i].T)
    idx = [i for i in range(len(runs)) if i % 5 == 4 and i != longest] or \
          [next(i for i in range(len(runs) - 1, -1, -1) if i != longest)]
    val = [runs[i] for i in idx]
    train = [r for i, r in enumerate(runs) if i not in idx]
    return train, val


def make_model(kind, spec, clip, sigma, cfg=None, val_runs=None, time_budget_s=None, l1=None):
    cfg = dict(cfg or {})
    if kind == "l0a":
        from gtlab.models.l0 import L0a
        return L0a(spec, clip=clip, sigma=sigma, **cfg)
    if kind == "l0b":
        from gtlab.models.l0 import L0b
        return L0b(spec, clip=clip, sigma=sigma, **cfg)
    if kind == "l1":
        from gtlab.models.l1 import L1
        return L1(spec, clip=clip, sigma=sigma, time_budget_s=time_budget_s, **cfg)
    if kind == "l2":
        from gtlab.models.l2 import L2
        return L2(spec, clip=clip, sigma=sigma, time_budget_s=time_budget_s, l1=l1, **cfg)
    if kind == "ode":
        from gtlab.models.ode_model import ODEModel
        return ODEModel(spec, clip=clip, sigma=sigma, val_runs=val_runs, time_budget_s=time_budget_s, **cfg)
    raise ValueError(kind)


def _predict(model, runs):
    return [model.rollout(r.y0, r.U) for r in runs]


def evaluate(model, runs, sigma):
    if not runs:
        return {}
    P = np.concatenate(_predict(model, runs), axis=0)
    Y = np.concatenate([r.Y for r in runs], axis=0)
    out = {"val_robust": metric.robust_score(P, Y, sigma),
           "val_score": metric.score(P, Y, sigma),
           "val_per_obs": [float(np.mean([metric.score_per_obs(P, Y, np.asarray(sigma) * k)[j]
                                          for k in (0.5, 1.0, 2.0)])) for j in range(Y.shape[1])]}
    if all(r.Ytrue is not None for r in runs):
        Yt = np.concatenate([r.Ytrue for r in runs], axis=0)
        out["val_truth_robust"] = metric.robust_score(P, Yt, sigma)
    return out


def gate_range(runs, margin=1.0):
    """Divergence range for the stress gate: data range widened by margin * range each side.
    (Outputs are clipped to the narrower soft range anyway; the gate only catches blow-ups.)"""
    Y = np.concatenate([np.vstack([r.y0[None, :], r.Y]) for r in runs], axis=0)
    mn, mx = Y.min(axis=0), Y.max(axis=0)
    R = np.maximum(mx - mn, 1e-9 + 1e-3 * np.abs(mx))
    return mn - margin * R, mx + margin * R


def stress(model, spec, inits, gate=None, n=200, T=4000, seed=0, budget_s=60.0, max_out_frac=0.01,
           max_ms_per_step=2.0, check_timing=True):
    from gtlab.check import schedules
    lo, hi = rt.clip_vectors(model.meta) if gate is None else gate
    slo, shi = rt.clip_vectors(model.meta)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    worst_out, worst_soft, nonfinite, done = 0.0, 0.0, 0, 0
    members = getattr(model, "members", None)
    for cat, U in schedules(spec, n, T, seed):
        y0 = inits[int(rng.integers(len(inits)))]
        for m in (members or [model]):
            Y = m.raw_rollout(y0, U)
            bad = ~np.isfinite(Y)
            nonfinite += int(bad.any())
            out = np.any((Y < lo - 1e-9) | (Y > hi + 1e-9) | bad, axis=1).mean()
            worst_out = max(worst_out, float(out))
            soft = np.any((Y < slo - 1e-9) | (Y > shi + 1e-9), axis=1).mean()
            worst_soft = max(worst_soft, float(soft))
        done += 1
        if time.time() - t0 > budget_s and done >= 10:
            break
    res = {"n": done, "nonfinite_runs": nonfinite, "worst_out_frac": round(worst_out, 5),
           "worst_softclip_frac": round(worst_soft, 5),
           "seconds": round(time.time() - t0, 1)}
    if check_timing:
        U = schedules(spec, 1, T, seed + 1)[0][1]
        t1 = time.time()
        rt.rollout_from_blob(model.doc(), inits[0], U)
        res["ms_per_step"] = round(1000 * (time.time() - t1) / T, 4)
    res["pass"] = bool(nonfinite == 0 and worst_out <= max_out_frac and
                       res.get("ms_per_step", 0.0) <= max_ms_per_step)
    return res


def _write_doc(out_dir, model_id, model, extra_info):
    d = Path(out_dir) / model_id
    d.mkdir(parents=True, exist_ok=True)
    doc = model.doc()
    doc["info"] = dict(doc.get("info") or {}, model_id=model_id, **extra_info)
    (d / "model.json").write_text(json.dumps(_jsonable(doc), sort_keys=True, allow_nan=False))
    return d / "model.json", doc


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return f if np.isfinite(f) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def fit_and_select(system_id, runs, out_dir, models=("l0a", "l0b", "l1", "l2", "ode"), time_budget_s=1800,
                   cfgs=None, n_stress=200, stress_budget_s=60.0, max_out_frac=0.01, gate_margin=50.0, clip_margin=None, tie=0.005,
                   per_obs=True, ensemble=True, refit_all=False, seed=0, verbose=True):
    t_start = time.time()
    spec = S.get(system_id)
    cfgs = dict(cfgs or {})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train, val = split_runs(runs)
    sigma = metric.sigma_proxy(runs)
    clip = C.soft_clip(spec, runs, clip_margin)
    inits = [r.y0 for r in runs]
    gate = gate_range(runs, gate_margin)
    if "ode" in models:
        from gtlab.models.ode_model import has_family
        if not has_family(system_id):
            models = tuple(m for m in models if m != "ode")
    shares = {k: DEFAULT_SHARES.get(k, 0.1) for k in models}
    tot = sum(shares.values()) or 1.0
    board, fitted = {}, {}

    def log(*a):
        if verbose:
            print(f"[{system_id} {time.time() - t_start:6.0f}s]", *a, flush=True)

    log(f"train {len(train)} runs / {sum(r.T for r in train)} ticks, val {len(val)} runs; sigma {np.round(sigma, 4).tolist()}")
    for kind in models:
        left = time_budget_s - (time.time() - t_start)
        budget = max(5.0, left * shares[kind] / max(1e-9, sum(shares[k] for k in models[models.index(kind):])))
        entry = {"model_id": kind, "kind": kind}
        try:
            t0 = time.time()
            m = make_model(kind, spec, clip, sigma, cfgs.get(kind), val_runs=val, time_budget_s=budget,
                           l1=fitted.get("l1"))
            m.fit(train)
            entry["fit_s"] = round(time.time() - t0, 1)
            entry.update(evaluate(m, val, sigma))
            entry["stress"] = stress(m, spec, inits, gate=gate, n=n_stress, seed=seed, budget_s=stress_budget_s,
                                     max_out_frac=max_out_frac)
            entry["pass"] = entry["stress"]["pass"]
            entry["info"] = _jsonable(m.info)
            fitted[kind] = m
            log(f"{kind:6s} val {entry.get('val_robust', float('nan')):.4f}  fit {entry['fit_s']}s  "
                f"stress {'ok' if entry['pass'] else 'FAIL'} {entry['stress']}")
        except Exception as e:
            entry["error"] = repr(e)
            entry["trace"] = traceback.format_exc()[-2000:]
            entry["pass"] = False
            log(f"{kind:6s} ERROR {e!r}")
        board[kind] = entry

    passing = [k for k in models if board[k].get("pass") and k in fitted]
    if val:
        passing.sort(key=lambda k: -board[k].get("val_robust", -1))
    if ensemble and len(passing) >= 3 and val:
        from gtlab.models.ensemble import Ensemble
        mem = [fitted[k] for k in passing[:3]]
        ens = Ensemble(spec, mem)
        e = {"model_id": "ens3", "kind": "ensemble", "members": passing[:3], **evaluate(ens, val, sigma),
             "stress": {"pass": True, "note": "members passed"}, "pass": True}
        e["stress"]["ms_per_step"] = sum(board[k]["stress"].get("ms_per_step", 0) for k in passing[:3])
        fitted["ens3"] = ens
        board["ens3"] = e
        log(f"ens3   val {e['val_robust']:.4f}  members {passing[:3]}")
    if per_obs and len(passing) >= 2 and val:
        from gtlab.models.ensemble import PerObs
        cands = [k for k in passing]
        if "ens3" in fitted:
            cands.append("ens3")
        best_j = [max(cands, key=lambda k: board[k]["val_per_obs"][j]) for j in range(spec.p)]
        uniq = sorted(set(best_j), key=cands.index)
        if len(uniq) >= 2:
            po = PerObs(spec, [fitted[k] for k in uniq], [uniq.index(k) for k in best_j])
            e = {"model_id": "perobs", "kind": "perobs", "sources": best_j, **evaluate(po, val, sigma),
                 "stress": {"pass": True, "note": "members passed"}, "pass": True}
            fitted["perobs"] = po
            board["perobs"] = e
            log(f"perobs val {e['val_robust']:.4f}  sources {best_j}")

    cand = [k for k in board if board[k].get("pass") and k in fitted]
    if not cand:
        from gtlab.models.l0 import L0a
        fitted["l0a"] = L0a(spec, clip=clip, sigma=sigma).fit(train)
        cand = ["l0a"]
        board.setdefault("l0a", {"model_id": "l0a", "kind": "l0a", "pass": True})
    order = {k: i for i, k in enumerate(LADDER)}
    if val:
        top = max(board[k].get("val_robust", -1) for k in cand)
        near = [k for k in cand if board[k].get("val_robust", -1) >= top - tie]
        pick_id = min(near, key=lambda k: order.get(k, 99))
    else:
        pick_id = max(cand, key=lambda k: order.get(k, -1) if k not in ("ens3", "perobs") else -1)

    for k in cand:
        path, _ = _write_doc(out_dir, k, fitted[k], {"val_robust": board[k].get("val_robust")})
        board[k]["path"] = str(path)
    pick_model = fitted[pick_id]
    refit_note = "none"
    if refit_all and val:
        try:
            pick_model = _refit(pick_id, board, spec, clip, sigma, cfgs, runs, fitted, time_budget_s)
            refit_note = "all runs"
        except Exception as e:
            refit_note = f"refit failed: {e!r}"
    path, doc = _write_doc(out_dir, "pick", pick_model,
                           {"picked": pick_id, "val_robust": board[pick_id].get("val_robust"), "refit": refit_note})
    lb = {"system": system_id, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "pick": pick_id, "refit": refit_note,
          "sigma": sigma.tolist(), "n_train": len(train), "n_val": len(val),
          "total_s": round(time.time() - t_start, 1),
          "models": sorted(board.values(), key=lambda e: -(e.get("val_robust") or -1))}
    (out_dir / "leaderboard.json").write_text(json.dumps(_jsonable(lb), indent=2))
    log(f"PICK {pick_id} (val {board[pick_id].get('val_robust')}) -> {path}")
    return {"system": system_id, "model_id": pick_id, "doc": doc, "path": str(path),
            "score": board[pick_id].get("val_robust"), "leaderboard": str(out_dir / "leaderboard.json")}


def _refit(pick_id, board, spec, clip, sigma, cfgs, runs, fitted, time_budget_s):
    """Refit the picked configuration on all runs (train + val)."""
    def one(kind):
        m = make_model(kind, spec, clip, sigma, cfgs.get(kind), val_runs=[],
                       time_budget_s=0.5 * time_budget_s)
        if kind == "ode":
            # keep the validated mechanism pair; refit theta on everything
            m.pairs = [fitted["ode"].mech]
        return m.fit(runs)
    if pick_id in ("l0a", "l0b", "l1", "l2", "ode"):
        return one(pick_id)
    if pick_id == "ens3":
        from gtlab.models.ensemble import Ensemble
        return Ensemble(spec, [one(k) for k in board["ens3"]["members"]])
    if pick_id == "perobs":
        from gtlab.models.ensemble import PerObs
        src = board["perobs"]["sources"]
        uniq = sorted(set(src), key=src.index)
        mem = {}
        for k in uniq:
            mem[k] = Ensemble_refit(k, board, one, spec) if k == "ens3" else one(k)
        return PerObs(spec, [mem[k] for k in uniq], [uniq.index(k) for k in src])
    raise ValueError(pick_id)


def Ensemble_refit(k, board, one, spec):
    from gtlab.models.ensemble import Ensemble
    return Ensemble(spec, [one(m) for m in board["ens3"]["members"]])
