"""ens9 stage 3: turn an accepted per-observable choice into a shippable model doc (no credits).

    python scripts/lab_ens9_assemble.py --system market --choice "y3;med(y3,d3);y3" [--full-budget 240]

choice: one option per observable separated by ';' (names as printed by lab_ens9_select.py: a member
label, or med(a,b[,c...]) = per-tick median). Members that are the current pick reuse the shipped theta;
other members use a full-data refit on every current run (lab protocol, cold, cached in
ENS9_SCRATCH/<system>/<label>__FULL.json). l0b_lin is refit on every run.
Writes plans/ens9_<system>_doc.json (Final-slot-1 doc with the model replaced by a perobs of
ensembles; clip vectors and post rules unchanged) and checks: runtime == manual combination,
4,000-tick eval-shaped rollouts finite, seconds per episode, fraction outside the data range.
Ship with scripts/build_cfg.py: {"<system>": {"doc": "plans/ens9_<system>_doc.json"}}.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import copy
import importlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from gtlab import design as D, metric, select as SEL  # noqa: E402
from gtlab.models import common as C  # noqa: E402
from gtlab.ode import core, fit as F  # noqa: E402
from gtlab.runtime import infer as rt  # noqa: E402
import lab_ens9_loo as L  # noqa: E402

PICK_FAMILY = {}  # label -> family of the shipped single-ODE pick, filled per system


def parse(choice):
    out = []
    for tok in choice.split(";"):
        tok = tok.strip()
        m = re.match(r"med\((.*)\)$", tok)
        out.append(tuple(x.strip() for x in m.group(1).split(",")) if m else (tok,))
    return out


def shipped_odes(doc):
    """{family: ode blob} for every ODE blob inside the shipped model."""
    res = {}

    def walk(b):
        if b["kind"] == "ode":
            res[b["family"]] = b
        for m in b.get("members", []) + ([b["member"]] if "member" in b else []):
            walk(m)
    walk(doc["model"])
    return res


def full_blob(system, label, runs, spec, sigma, shipped, budget, starts, nfev):
    if label == "l0b":
        clip = C.soft_clip(spec, runs, margin=1.0)
        with C.single_thread():
            m = SEL.make_model("l0b", spec, clip, sigma, cfg={"sq": False, "pairs": None}).fit(runs)
        return SEL._jsonable(m.export())
    blob = L.member_blob(system, label)
    if blob["family"] in shipped:
        return copy.deepcopy(shipped[blob["family"]])
    f = L.SCRATCH / system / f"{label}__FULL.json"
    if f.exists():
        th = json.loads(f.read_text())["theta"]
    else:
        mod = importlib.import_module(f"gtlab.ode.{blob['family']}")
        n_sub = int(blob.get("n_sub", getattr(mod, "N_SUB", 2)))
        t0 = time.time()
        with C.single_thread():
            thv, info = F.fit_ode(mod, runs, sigma, frozenset(blob["mech"]), n_starts=starts, max_nfev=nfev, n_sub=n_sub,
                                  early_T=None, time_budget=budget, theta0=None, spread=0.5)
        th = [float(v) for v in thv]
        f.write_text(json.dumps({"theta": th, "cost": float(info["cost"]), "sec": time.time() - t0,
                                 "runs": [r.exp for r in runs]}))
    return {"kind": "ode", "family": blob["family"], "theta": th, "mech": sorted(blob["mech"]),
            "n_sub": int(blob.get("n_sub", 2))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--choice", required=True)
    ap.add_argument("--full-budget", type=float, default=240)
    ap.add_argument("--starts", type=int, default=10)
    ap.add_argument("--nfev", type=int, default=60)
    a = ap.parse_args()
    s = a.system
    spec, runs = L.get_runs(s)
    sigma = L.cal_sigma(s)
    doc = json.loads((L.FINAL / s / "model.json").read_text())
    shipped = shipped_odes(doc)
    chosen = parse(a.choice)
    assert len(chosen) == len(spec.observables), "one option per observable"
    used = list(dict.fromkeys(l for o in chosen for l in o))
    blobs = {l: full_blob(s, l, runs, spec, sigma, shipped, a.full_budget, a.starts, a.nfev) for l in used}
    members, idx = [], {}
    for o in dict.fromkeys(chosen):
        idx[o] = len(members)
        members.append(blobs[o[0]] if len(o) == 1 else {"kind": "ensemble", "members": [blobs[l] for l in o]})
    edoc = copy.deepcopy(doc)
    edoc["model"] = members[0] if len(members) == 1 else {"kind": "perobs", "members": members, "map": [idx[o] for o in chosen]}
    edoc["info"] = {"model_id": "ens9 perobs: " + ", ".join(f"{ob}={'+'.join(o)}" for ob, o in zip(spec.observables, chosen)),
                    "members": {l: b.get("family", b["kind"]) + ":" + "".join(b.get("mech", "")) for l, b in blobs.items()}}
    lo, hi = rt.clip_vectors(edoc)
    chk = {"max_err_vs_manual_sigma": 0.0, "insample": {}}
    for r in runs:
        Yz = rt.rollout_from_blob(edoc, r.y0, r.U, doc=edoc)
        M = {}
        for l, b in blobs.items():
            Yl = rt.rollout_from_blob(b, r.y0, r.U, doc=edoc, clip=False)
            M[l] = rt.finalize(Yl, r.y0, lo, hi)
        Ym = np.stack([np.median(np.stack([M[l] for l in o]), axis=0)[:, j] for j, o in enumerate(chosen)], axis=1)
        Ym = rt.apply_post(edoc, rt.finalize(Ym, r.y0, lo, hi), r.U, r.y0)
        chk["max_err_vs_manual_sigma"] = max(chk["max_err_vs_manual_sigma"], float(np.max(np.abs(Ym - Yz) / sigma)))
        chk["insample"][r.exp] = metric.score_per_obs(Yz, r.Y, sigma).tolist()
    Yall = np.concatenate([r.Y for r in runs])
    ymin, ymax = Yall.min(0), Yall.max(0)
    rg = ymax - ymin
    rng = np.random.default_rng(0)
    chk["eval"] = {}
    for cat in ("sustained", "order", "recovery", "composition"):
        U = D.eval_like(spec, cat, 4000, rng)
        t0 = time.time()
        Y = rt.rollout_from_blob(edoc, runs[0].y0, U, doc=edoc)
        dt = time.time() - t0
        chk["eval"][cat] = {"finite": bool(np.all(np.isfinite(Y))), "seconds": dt,
                            "frac_outside": float(np.mean((Y > ymax + 0.05 * rg) | (Y < ymin - 0.05 * rg)))}
    chk["ok"] = bool(all(v["finite"] and v["seconds"] < 2.0 for v in chk["eval"].values())
                     and chk["max_err_vs_manual_sigma"] < 1e-9)
    edoc["info"]["checks"] = chk
    out = ROOT / "plans" / f"ens9_{s}_doc.json"
    out.write_text(json.dumps(SEL._jsonable(edoc), indent=1))
    ins = np.mean([np.mean(v) for v in chk["insample"].values()])
    print(f"{s}: ok={chk['ok']} err={chk['max_err_vs_manual_sigma']:.1e} insample {ins:.4f} "
          + " ".join(f"{c}:{v['seconds']:.2f}s/out{v['frac_outside']:.2f}/fin{v['finite']}" for c, v in chk["eval"].items())
          + f" -> {out}")


if __name__ == "__main__":
    main()
