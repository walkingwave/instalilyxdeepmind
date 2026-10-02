"""ens9 stage 2: nested cross-validated ensemble selection over the stage-1 held-out predictions.

    python scripts/lab_ens9_select.py [--systems a,b] [--margin 0.005] [--out plans/ens9_select.json]

Inputs: ENS9_SCRATCH/<system>/<label>__<held>.json (scripts/lab_ens9_loo.py). Every member
prediction is finalized with the Final-slot-1 doc's clip vectors; the doc's post rules are applied
after combining. Score per observable = mean_t 1/(1+|e|/sigma), calibrated sigma 1.0x.

Option pool per observable (all shippable with the current runtime unless marked):
  single member; median of any 2 (= mean) or 3 members; median of all ODE members; median of all
  members incl. l0b_lin; [ext] horizon switch sw(A,B,H): member A for the first H ticks after any
  control change (and after reset), member B afterwards; [ext] mixture mix(A,B,w): w A + (1-w) B.
  "ext" options need a runtime kind that does not exist yet; they are reported, never auto-picked.

Selection strategies, each scored by nested leave-one-run-out (the choice for fold k is made
from the other folds only; ties to the pick unless the inner gain > margin):
  pick           : the shipped per-observable option (reference)
  sys_best       : one option for all observables
  obs_best       : per observable best option
  obs_best_ext   : per observable, pool incl. ext options
  med_all / mean-free fixed rules are also reported un-nested (no selection involved).
Acceptance for a final (all-fold) choice: nested gain > 0, all-fold gain > margin, does not lose
the pulse / recovery-shaped fold(s) nor the exam fold(s) by more than 0.002.
"""
import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from gtlab import metric  # noqa: E402
from gtlab.runtime import infer as rt  # noqa: E402
import lab_ens9_loo as L  # noqa: E402

PICK = {
    "ad_auction": [("v8b",), ("v8b", "s1"), ("v8b", "min")],
    "power_grid": [("v9c",), ("v9c",), ("v8k", "min")],
    "epidemic": "y2", "hospital_queue": "p3", "market": "y3", "reservoir": "v8", "social_contagion": "z20",
    "supply_chain": "v8b", "traffic": "z8", "wildlife": "v8h",
}
PULSE_KEYS = ("pulse", "recovery")
EXAM_KEYS = ("exam", "testlike")
HS = (5, 10, 20, 40, 80)
WS = (0.25, 0.5, 0.75)


def since_change(U):
    """ticks since the last control change (0 at reset and at every change tick)."""
    T = U.shape[0]
    d = np.zeros(T)
    for t in range(1, T):
        d[t] = 0 if np.any(np.abs(U[t] - U[t - 1]) > 1e-9) else d[t - 1] + 1
    return d


def load_preds(system, runs, labels, doc):
    lo, hi = rt.clip_vectors(doc)
    P = {}
    for l in labels:
        P[l] = {}
        for r in runs:
            f = L.SCRATCH / system / f"{l}__{r.exp}.json"
            if not f.exists():
                return None, f"missing {f.name}"
            pay = json.loads(f.read_text())
            if "error" in pay:
                return None, f"{l} {r.exp}: {pay['error']}"
            P[l][r.exp] = rt.finalize(np.asarray(pay["Y"], float), r.y0, lo, hi)
    return P, None


def opt_name(o):
    k = o[0]
    if k == "m":
        return o[1][0] if len(o[1]) == 1 else "med(" + ",".join(o[1]) + ")"
    if k == "sw":
        return f"sw({o[1]}->{o[2]},H{o[3]})"
    return f"mix({o[1]},{o[2]},{o[3]})"


def build(P, o, r, D):
    k = o[0]
    if k == "m":
        return np.median(np.stack([P[l][r.exp] for l in o[1]]), axis=0)
    A, B = P[o[1]][r.exp], P[o[2]][r.exp]
    if k == "sw":
        m = (D[r.exp] < o[3])[:, None]
        return np.where(m, A, B)
    return o[3] * A + (1 - o[3]) * B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default=None)
    ap.add_argument("--margin", type=float, default=0.005)
    ap.add_argument("--no-ext", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "plans" / "ens9_select.json"))
    a = ap.parse_args()
    systems = a.systems.split(",") if a.systems else L.SYSTEMS
    out = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    for s in systems:
        spec, runs = L.get_runs(s)
        doc = json.loads((L.FINAL / s / "model.json").read_text())
        sigma = L.cal_sigma(s)
        labels = list(L.MEMBERS[s]) + ["l0b"]
        P, err = load_preds(s, runs, labels, doc)
        if P is None:
            print(f"{s}: skipped ({err})")
            continue
        p = len(spec.observables)
        odes = [l for l in labels if l != "l0b"]
        base = [("m", (l,)) for l in labels]
        base += [("m", c) for c in itertools.combinations(labels, 2)]
        base += [("m", c) for c in itertools.combinations(labels, 3)]
        base += [("m", tuple(odes)), ("m", tuple(labels))]
        base = list(dict.fromkeys(base))
        ext = []
        if not a.no_ext:
            for A, B in itertools.permutations(odes, 2):
                ext += [("sw", A, B, H) for H in HS]
            for A, B in itertools.combinations(labels, 2):
                ext += [("mix", A, B, w) for w in WS]
        D = {r.exp: since_change(r.U) for r in runs}
        pk = PICK[s]
        pick_opts = [("m", tuple(x)) for x in pk] if isinstance(pk, list) else [("m", (pk,))] * p
        allopts = list(dict.fromkeys(base + ext + pick_opts))
        # SC[o] = [folds, p] per-observable held-out scores
        SC = {}
        for o in allopts:
            rows = []
            for r in runs:
                Y = rt.apply_post(doc, build(P, o, r, D).copy(), r.U, r.y0)
                rows.append(metric.score_per_obs(Y, r.Y, sigma))
            SC[o] = np.array(rows)
        nf = len(runs)
        exps = [r.exp for r in runs]
        pick_sc = np.stack([SC[pick_opts[j]][:, j] for j in range(p)], axis=1)       # [folds, p]

        def choose(pool, idx, per_obs):
            """options chosen from folds idx; falls back to the pick unless inner gain > margin."""
            if per_obs:
                ch = []
                for j in range(p):
                    b = pick_sc[idx, j].mean()
                    best = max(pool, key=lambda o: SC[o][idx, j].mean())
                    ch.append(best if SC[best][idx, j].mean() > b + a.margin else pick_opts[j])
                return ch
            b = pick_sc[idx].mean()
            best = max(pool, key=lambda o: SC[o][idx].mean())
            return [best] * p if SC[best][idx].mean() > b + a.margin else list(pick_opts)

        def topk(idx, k, per_obs, odes_only=True):
            cands = odes if odes_only else labels
            if per_obs:
                ch = []
                for j in range(p):
                    r_ = sorted(cands, key=lambda l: -SC[("m", (l,))][idx, j].mean())[:k]
                    ch.append(("m", tuple(sorted(r_, key=labels.index))))
                return ch
            r_ = sorted(cands, key=lambda l: -SC[("m", (l,))][idx].mean())[:k]
            return [("m", tuple(sorted(r_, key=labels.index)))] * p

        def score_of(ch, k):
            return float(np.mean([SC[ch[j]][k, j] for j in range(p)]))

        strategies = {"sys_best": (base, False), "obs_best": (base, True)}
        if ext:
            strategies["obs_best_ext"] = (base + ext, True)
        rep = {"runs": exps, "labels": labels, "observables": list(spec.observables),
               "pick": [opt_name(o) for o in pick_opts],
               "pick_fold": {e: float(pick_sc[k].mean()) for k, e in enumerate(exps)},
               "pick_mean": float(pick_sc.mean())}
        rep["members"] = {l: {"per_obs": SC[("m", (l,))].mean(0).tolist(), "mean": float(SC[("m", (l,))].mean()),
                              "fold": {e: float(SC[("m", (l,))][k].mean()) for k, e in enumerate(exps)}} for l in labels}
        fixed = {"med_odes": ("m", tuple(odes)), "med_all": ("m", tuple(labels))}
        rep["fixed"] = {n: {"mean": float(SC[o].mean()), "per_obs": SC[o].mean(0).tolist(),
                            "fold": {e: float(SC[o][k].mean()) for k, e in enumerate(exps)}} for n, o in fixed.items()}
        rep["nested"] = {}
        for kk in (2, 3):
            strategies[f"top{kk}_sys"] = (("top", kk), False)
            strategies[f"top{kk}_obs"] = (("top", kk), True)
        for name, (pool, per_obs) in strategies.items():
            fold = {}
            chosen = {}
            sel = (lambda idx: topk(idx, pool[1], per_obs)) if isinstance(pool, tuple) else (lambda idx: choose(pool, idx, per_obs))
            for k in range(nf):
                idx = [i for i in range(nf) if i != k]
                ch = sel(idx)
                fold[exps[k]] = score_of(ch, k)
                chosen[exps[k]] = [opt_name(o) for o in ch]
            full = sel(list(range(nf)))
            for o in full:
                if o not in SC:
                    SC[o] = np.array([metric.score_per_obs(rt.apply_post(doc, build(P, o, r, D).copy(), r.U, r.y0), r.Y, sigma)
                                      for r in runs])
            full_fold = {e: score_of(full, k) for k, e in enumerate(exps)}
            nested_mean = float(np.mean(list(fold.values())))
            full_mean = float(np.mean(list(full_fold.values())))
            pulse = [e for e in exps if any(q in e for q in PULSE_KEYS)]
            exam = [e for e in exps if any(q in e for q in EXAM_KEYS)]
            loses = [e for e in pulse + exam if full_fold[e] < rep["pick_fold"][e] - 0.002]
            rep["nested"][name] = {
                "nested_fold": fold, "nested_mean": nested_mean, "nested_gain": nested_mean - rep["pick_mean"],
                "nested_choices": chosen, "full_choice": [opt_name(o) for o in full], "full_fold": full_fold,
                "full_mean": full_mean, "full_gain": full_mean - rep["pick_mean"],
                "full_per_obs": [float(SC[full[j]][:, j].mean()) for j in range(p)],
                "pick_per_obs": pick_sc.mean(0).tolist(),
                "full_obs_wins": [int(np.sum(SC[full[j]][:, j] > pick_sc[:, j] + 1e-12)) for j in range(p)],
                "guard_folds": pulse + exam, "loses_guard": loses,
                "accept": bool(nested_mean > rep["pick_mean"] and full_mean > rep["pick_mean"] + a.margin and not loses
                               and not any(o[0] != "m" for o in full)),
            }
        # top options per observable (all folds) for the report
        rep["top_per_obs"] = {}
        for j, ob in enumerate(spec.observables):
            ranked = sorted(allopts, key=lambda o: -SC[o][:, j].mean())[:6]
            rep["top_per_obs"][ob] = [(opt_name(o), float(SC[o][:, j].mean()),
                                       int(np.sum(SC[o][:, j] > pick_sc[:, j] + 1e-12))) for o in ranked]
        out[s] = rep
        n = rep["nested"]
        print(f"\n== {s}: pick {rep['pick']} mean {rep['pick_mean']:.4f}  folds {nf}")
        print("   members: " + ", ".join(f"{l} {rep['members'][l]['mean']:.3f}" for l in labels))
        print("   fixed:   " + ", ".join(f"{k} {v['mean']:.3f}" for k, v in rep["fixed"].items()))
        for name, v in n.items():
            print(f"   {name:<13} nested {v['nested_mean']:.4f} ({v['nested_gain']:+.4f})  full {v['full_mean']:.4f} "
                  f"({v['full_gain']:+.4f}) choice {v['full_choice']} wins {v['full_obs_wins']} "
                  f"loses {v['loses_guard']} accept {v['accept']}")
        for ob, rows in rep["top_per_obs"].items():
            print(f"   top {ob:<18} " + "; ".join(f"{nm} {sc:.3f} w{w}" for nm, sc, w in rows[:4]))
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
