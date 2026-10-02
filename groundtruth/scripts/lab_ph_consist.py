"""Public-history consistency (free, no gateway).

Every distinct scored predictor of a system (deduplicated on predict.py+model.json) plus the candidate
models is rolled once on the same eval-like episodes (gtlab.check.schedules, 40 non-pinned episodes,
T = 4,000, y0 from the reset pool). Then every model C is taken in turn as the noiseless truth and
every scored predictor Q is scored against it at the calibrated sigma. Implied scores are compared
with the public scores: mean absolute gap, correlation, mean offset (public - implied), also on the
strong uploads only (public >= 0.7). C's own upload is excluded from its statistics.

    python scripts/lab_ph_consist.py <system> [--n 40] [--seed 11]
Writes plans/ph_<system>.json.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import systems as S, check as CK                                   # noqa: E402
from gtlab.runtime import infer                                                # noqa: E402

warnings.filterwarnings("ignore")
CANDS = {
    "final4": "submissions/20260929-2000-final4",
    "alt1b": "submissions/20260929-1840-alt1b",
    "final1": "submissions/20260928-1440-final1",
}
EXTRA_DOCS = {"traffic": {"hi_all_s": "plans/traffic_hi_all_s_doc.json"}}


def folder_hash(folder, sid):
    p = ROOT / folder.replace("\\", "/") / sid / "predict.py"
    if not p.exists():
        return None, None
    mj = p.parent / "model.json"
    return p, hashlib.sha1(p.read_bytes() + (mj.read_bytes() if mj.exists() else b"")).hexdigest()[:10]


def load(p, tag):
    sp = importlib.util.spec_from_file_location(f"ph_{tag}", str(p))
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    sid = a.system
    spec = S.get(sid)
    obs, ctrls = list(spec.observables), list(spec.controls)
    sig = np.array(json.loads((ROOT / "plans" / "sigma_calibrated.json").read_text())[sid]["sigma"], float)

    reg = json.loads((ROOT / "tune" / "registry.json").read_text())["uploads"]
    bands = {r["upload"]: r for r in json.loads((ROOT / "tune" / "bands.json").read_text())["rows"]
             if r["system"] == sid}
    models = {}      # hash -> {"path", "uploads", "pub", "names"}
    for u in reg:
        s = (u.get("scores") or {}).get(sid)
        if s is None:
            continue
        p, h = folder_hash(u["dir"], sid)
        if p is None:
            if u["id"] != "u001":
                continue
            p, h = "persistence", "persist"
        m = models.setdefault(h, {"path": p, "uploads": [], "pubs": [], "sus": [], "seq": []})
        m["uploads"].append(u["id"])
        m["pubs"].append(float(s))
        if u["id"] in bands:
            m["sus"].append(bands[u["id"]]["sustained"])
            m["seq"].append(bands[u["id"]]["sequence"])
    for name, d in CANDS.items():
        p, h = folder_hash(d, sid)
        m = models.setdefault(h, {"path": p, "uploads": [], "pubs": [], "sus": [], "seq": []})
        m.setdefault("cands", []).append(name)
    for name, dp in EXTRA_DOCS.get(sid, {}).items():
        m = models.setdefault("doc:" + name, {"path": ("doc", dp), "uploads": [], "pubs": [], "sus": [], "seq": []})
        m.setdefault("cands", []).append(name)
    for h, m in models.items():
        m["name"] = "/".join(m.get("cands", [])) or m["uploads"][-1]
        if m["uploads"] and m.get("cands"):
            m["name"] += f"(={m['uploads'][-1]})"
        m["pub"] = float(np.mean(m["pubs"])) if m["pubs"] else None
        m["pub_sus"] = float(np.mean(m["sus"])) if m["sus"] else None
        m["pub_seq"] = float(np.mean(m["seq"])) if m["seq"] else None

    sch = CK.schedules(spec, a.n + 2, 4000, a.seed)
    inits, src = CK.initial_conditions(spec, a.n + 2, a.seed)
    eps = [(cat, np.array([y0[o] for o in obs], float), U) for (cat, U), y0 in zip(sch, inits)
           if not cat.startswith("pinned")][: a.n]
    cats = np.array([e[0] for e in eps])
    print(sid, "episodes", len(eps), "y0 source", src, "models", len(models), flush=True)

    R = {}
    for h, m in models.items():
        t0 = time.time()
        out = []
        if m["path"] == "persistence":
            out = [np.tile(y0, (4000, 1)) for _, y0, _ in eps]
        elif isinstance(m["path"], tuple):
            doc = json.loads((ROOT / m["path"][1]).read_text())
            out = [np.asarray(infer.rollout_from_blob(doc, y0, U), float) for _, y0, U in eps]
        else:
            mod = load(m["path"], h)
            for _, y0, U in eps:
                o = mod.predict({k: float(v) for k, v in zip(obs, y0)},
                                [{c: float(v) for c, v in zip(ctrls, r)} for r in U], spec.context())
                Y = np.array([[row[k] for k in obs] for row in o], float)
                out.append(np.where(np.isfinite(Y), Y, np.nan))
        R[h] = np.stack(out)
        print(f"  rolled {m['name']:28s} {time.time() - t0:5.1f}s", flush=True)

    def sc(Q, T):   # per-episode score
        e = np.abs(np.nan_to_num(Q, nan=1e12) - T)
        return np.mean(1.0 / (1.0 + e / sig[None, None, :]), axis=(1, 2))

    hs = list(models)
    scored = [h for h in hs if models[h]["pub"] is not None]
    out = {"system": sid, "n_eps": len(eps), "seed": a.seed, "models": {}, "truths": {}}
    for h in hs:
        m = models[h]
        out["models"][h] = {k: m[k] for k in ("name", "uploads", "pub", "pub_sus", "pub_seq")}
        out["models"][h]["cands"] = m.get("cands", [])
    for ht in hs:
        T = R[ht]
        imp = {}
        for hq in hs:
            s = sc(R[hq], T)
            imp[hq] = {"all": float(s.mean()), "sus": float(s[cats == "sustained"].mean()),
                       "seq": float(s[cats != "sustained"].mean())}
        hist = [h for h in scored if h != ht]
        x = np.array([imp[h]["all"] for h in hist])
        y = np.array([models[h]["pub"] for h in hist])
        top = np.array([models[h]["pub"] >= 0.7 for h in hist])
        st = {"gap": float(np.mean(np.abs(y - x))), "corr": float(np.corrcoef(x, y)[0, 1]),
              "offset": float(np.mean(y - x)), "n": int(len(hist)),
              "gap_top": float(np.mean(np.abs(y - x)[top])) if top.any() else None,
              "offset_top": float(np.mean((y - x)[top])) if top.any() else None, "n_top": int(top.sum())}
        if len(hist) >= 3:
            b, c0 = np.polyfit(x, y, 1)
            st["fit"] = [float(b), float(c0)]
        bh = [h for h in hist if models[h]["pub_sus"] is not None]
        if bh:
            st["gap_sus"] = float(np.mean([abs(models[h]["pub_sus"] - imp[h]["sus"]) for h in bh]))
            st["gap_seq"] = float(np.mean([abs(models[h]["pub_seq"] - imp[h]["seq"]) for h in bh]))
        out["truths"][ht] = {"stats": st, "implied": imp}
        print(f"truth {models[ht]['name']:28s} gap {st['gap']:.3f} corr {st['corr']:.3f} off {st['offset']:+.3f} "
              f"gap_top {st['gap_top'] if st['gap_top'] is None else round(st['gap_top'], 3)} "
              f"off_top {st['offset_top'] if st['offset_top'] is None else round(st['offset_top'], 3)}", flush=True)
        print("     implied: " + "  ".join(f"{models[h]['name']}={imp[h]['all']:.3f}"
                                           + (f"(pub {models[h]['pub']:.3f})" if models[h]['pub'] else "")
                                           for h in hs if h != ht), flush=True)
    suf = "" if a.seed == 11 else f"_s{a.seed}"
    (ROOT / "plans" / f"ph_{sid}{suf}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
