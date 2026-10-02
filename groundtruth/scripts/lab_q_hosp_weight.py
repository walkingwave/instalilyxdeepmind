"""hospital_queue: weight w on canon2 in a per-tick mix with h9w1 (free, no gateway).

Truth family T_w = w*canon2 + (1-w)*h9w1 (noiseless rollouts of the two full-data docs) on 40
eval-like episodes. For each truth: implied score of every past scored predictor (public history,
now incl. u019 = final4 = mean of the two), offset/SD vs public, the direct pair u017 (final1) and
u019 (final4) vs 0.6896 / 0.7294, and the offset-free gap final4 - final1 vs +0.0398.
Also the implied score of candidate predictors P_v = v*canon2 + (1-v)*h9w1 under each truth.

    python scripts/lab_q_hosp_weight.py [--seed 11] [--n 40]
Writes plans/q_hosp_weight[_sS].json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from gtlab import systems as S                     # noqa: E402
from gtlab.check import make_episodes              # noqa: E402
from gtlab.runtime import infer                    # noqa: E402
from lab_ph2_history import fhash, load, CATS      # noqa: E402

SID = "hospital_queue"
DOCS = {"h9w1": "plans/hospital_queue_hospital_queue_hosp9_h9w1_doc.json",
        "canon2": "plans/hospital_queue_canon_hospital_queue_canon2_hr_doc.json"}
WT = [0.0, 0.15, 0.2, 0.25, 0.3, 0.5, 0.65, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35]
WP = [0.0, 0.5, 0.65, 0.8, 1.0]
PUB = {"u017": 0.6896, "u019": 0.7294}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    spec = S.get(SID)
    obs = list(spec.observables)
    ctx = spec.context()
    sig = np.array(json.loads((ROOT / "plans" / "sigma_calibrated.json").read_text())[SID]["sigma"], float)
    eps, _ = make_episodes(spec, n=a.n, T=4000, seed=a.seed)
    cat = np.array([e["category"] for e in eps])

    reg = json.loads((ROOT / "tune" / "registry.json").read_text())["uploads"]
    P = {}
    for u in reg:
        pub = (u.get("scores") or {}).get(SID)
        if pub is None:
            continue
        h = fhash(u["dir"], SID)
        if h is None:
            if u["id"] != "u001":
                continue
            h = "persistence"
        g = P.setdefault(h, {"dir": u["dir"], "ups": [], "pub": []})
        g["ups"].append(u["id"])
        g["pub"].append(pub)

    Y = {}
    for h, g in P.items():
        out = []
        if h == "persistence":
            for e in eps:
                y0 = np.array([e["initial"][o] for o in obs], float)
                out.append(np.repeat(y0[None, :], 4000, axis=0))
        else:
            mod = load(g["dir"], SID, h)
            for e in eps:
                res = mod.predict(dict(e["initial"]), [dict(r) for r in e["interventions"]], ctx)
                out.append(np.array([[r[o] for o in obs] for r in res], float))
        Y[h] = np.nan_to_num(np.stack(out), nan=1e12, posinf=1e12, neginf=-1e12)
    D = {}
    for nm, f in DOCS.items():
        doc = json.loads((ROOT / f).read_text())
        idx = [doc["observables"].index(o) for o in obs]
        out = []
        for e in eps:
            y0 = np.array([e["initial"][o] for o in doc["observables"]], float)
            U = np.array([[r[c] for c in doc["controls"]] for r in e["interventions"]], float)
            out.append(np.asarray(infer.rollout_from_blob(doc, y0, U), float)[:, idx])
        D[nm] = np.stack(out)
    H, C = D["h9w1"], D["canon2"]

    def sc(Q, T):
        s = np.mean(1.0 / (1.0 + np.abs(Q - T) / sig[None, None, :]), axis=1)   # ep x obs
        per = {c: s[cat == c].mean(axis=0) for c in CATS}
        tot = np.mean([per[c] for c in CATS], axis=0)                          # per obs
        return float(tot.mean()), [float(x) for x in tot]

    name = {h: "/".join(g["ups"]) for h, g in P.items()}
    h17 = next(h for h, g in P.items() if "u017" in g["ups"])
    h19 = next(h for h, g in P.items() if "u019" in g["ups"])
    # sanity: final4 == mean of docs?
    mix = 0.5 * (H + C)
    print("final4 vs mean(docs) max|diff| per obs", np.abs(Y[h19] - mix).max(axis=(0, 1)).round(4))

    res = {"seed": a.seed, "n": a.n, "obs": obs, "truths": {}}
    print(f"\n{'w_truth':>7} {'off':>7} {'sd':>6} {'u017':>6} {'u019':>6} {'gap':>6} | "
          + " ".join(f"P{v:<5}" for v in WP))
    for w in WT:
        T = w * C + (1 - w) * H
        rows = [(h, float(np.mean(P[h]["pub"])), sc(Y[h], T)[0]) for h in P]
        d = np.array([r[2] - r[1] for r in rows])
        pu = np.array([r[1] for r in rows]); im = np.array([r[2] for r in rows])
        keep = im < 0.999
        A = np.vstack([np.ones(keep.sum()), im[keep]]).T
        cf, *_ = np.linalg.lstsq(A, pu[keep], rcond=None)
        sd_aff = float(np.std(pu[keep] - A @ cf))
        i17, i19 = sc(Y[h17], T), sc(Y[h19], T)
        cand = {str(v): sc(v * C + (1 - v) * H, T) for v in WP}
        res["truths"][str(w)] = {"offset": float(d.mean()), "sd": float(d.std()),
                                 "rows": [{"q": name[h], "pub": p, "imp": i} for h, p, i in rows],
                                 "u017": i17, "u019": i19, "gap": i19[0] - i17[0], "cand": cand,
                                 "sd_aff": sd_aff, "aff": [float(x) for x in cf]}
        print(f"{w:7.2f} {d.mean():+.3f} {d.std():.3f} {i17[0]:.3f} {i19[0]:.3f} {i19[0]-i17[0]:+.3f} {sd_aff:.3f} | "
              + " ".join(f"{cand[str(v)][0]:.3f}" for v in WP))
    print("public: u017 0.6896 u019 0.7294 gap +0.0398")
    print("\nper-obs implied (wait, queue, discharges): u017 / u019 / P1.0 under truth w")
    for w in WT:
        r = res["truths"][str(w)]
        print(f"  w={w:.2f}  u017 {np.round(r['u017'][1],3)}  u019 {np.round(r['u019'][1],3)}  "
              f"P1 {np.round(r['cand']['1.0'][1],3)}  P.5 {np.round(r['cand']['0.5'][1],3)}")
    sfx = "" if a.seed == 11 else f"_s{a.seed}"
    (ROOT / "plans" / f"q_hosp_weight{sfx}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
