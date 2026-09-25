"""Gate the grey-box lab reports and emit build picks.

    python scripts/assemble_picks.py [--min-gain 0.05] [--max-outside 0.10] [--max-seconds 1.0]
                                     [--pattern "plans/*_min.json"] [--out plans/picks_ode.json]

Gates (all must pass for a system to be picked):
  1. leave-one-run-out mean of the ODE > l0b_lin mean + min_gain on the same folds
     (sigma_proxy inflates scores, so diffs under 0.05 are ties), and not below persistence
     on any fold
  2. every eval-shaped 4,000-tick rollout finite, per-episode time < max_seconds,
     fraction of ticks outside the observed data range (+-5 %) < max_outside on every category
Writes {system: doc_path} for the passing systems and prints the full table.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-gain", type=float, default=0.05)
    ap.add_argument("--max-outside", type=float, default=0.10)
    ap.add_argument("--max-seconds", type=float, default=1.0)
    ap.add_argument("--pattern", default="plans/*_min.json")
    ap.add_argument("--out", default="plans/picks_ode.json")
    a = ap.parse_args()
    picks, rows = {}, []
    for f in sorted(glob.glob(a.pattern)):
        if f.endswith("_doc.json") or f.endswith("_notes.json"):
            continue
        rep = json.loads(Path(f).read_text())
        if "loo" not in rep or "eval" not in rep:
            continue
        sid, fam = rep["system"], rep["family"]
        loo = rep["loo"]
        ode = np.mean([np.mean(x["ode"]) for x in loo]) if loo else float("nan")
        base = np.mean([np.mean(x["l0b_lin"]) for x in loo]) if loo else float("nan")
        worst_vs_pers = min((np.mean(x["ode"]) - np.mean(x["persistence"])) for x in loo) if loo else float("nan")
        ev = rep["eval"]
        finite = all(v["finite"] for v in ev.values())
        secs = max(v["seconds"] for v in ev.values())
        outside = max(v["frac_outside"] for v in ev.values())
        ins = rep.get("insample_mean", float("nan"))
        g1 = bool(loo) and (ode > base + a.min_gain) and (worst_vs_pers > -1e-9)
        g2 = finite and secs < a.max_seconds and outside < a.max_outside
        ok = g1 and g2
        doc = str(Path(f).with_name(Path(f).stem + "_doc.json"))
        rows.append((sid, fam, ode, base, worst_vs_pers, ins, outside, secs, rep.get("at_bound", []), ok, doc))
        if ok and Path(doc).exists():
            picks[sid] = doc
    print(f"{'system':<17} {'family':<22} {'LOO ode':>8} {'l0b_lin':>8} {'vs pers':>8} {'in-samp':>8} {'outside':>8} {'sec':>5}  gate  at_bound")
    for sid, fam, ode, base, wp, ins, out, secs, ab, ok, doc in rows:
        print(f"{sid:<17} {fam:<22} {ode:8.3f} {base:8.3f} {wp:+8.3f} {ins:8.3f} {out:8.2f} {secs:5.2f}  {'PASS' if ok else 'fail'}  {','.join(ab)}")
    Path(a.out).write_text(json.dumps(picks, indent=1))
    print(f"{len(picks)} picks -> {a.out}: {sorted(picks)}")


if __name__ == "__main__":
    main()
