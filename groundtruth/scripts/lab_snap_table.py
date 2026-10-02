"""Tables for the round-constant snap test: reads plans/snap_loo_*.json and prints markdown
(per system: variant LOO means, per-fold delta vs base, fold spread of the pinned parameters).

    python scripts/lab_snap_table.py > plans/snap_tables.md
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def short(e):
    return e.split(".", 1)[1] if "." in e else e


def main():
    for f in sorted((ROOT / "plans").glob("snap_loo_*.json")):
        if f.stem.endswith("_ctl"):
            continue
        r = json.loads(f.read_text())
        V = r["variants"]
        ctl = f.with_name(f.stem + "_ctl.json")
        if ctl.exists():
            c = json.loads(ctl.read_text())
            V.update(c["variants"]); r["pins"].update(c["pins"])
        if "base" not in V:
            continue
        base = V["base"]
        print(f"\n### {r['system']} / {r['family']} (mech {''.join(r['mech'])})\n")
        folds = [x["held_out"] for x in base["folds"]]
        hdr = "| variant | pins | LOO | " + " | ".join(short(e) for e in folds) + " |"
        print(hdr)
        print("|" + "---|" * (3 + len(folds)))
        print(f"| base | 0 | {base['loo_mean']:.4f} | " + " | ".join(f"{x['mean']:.3f}" for x in base["folds"]) + " |")
        for v, d in V.items():
            if v == "base":
                continue
            n = len(d["folds"])
            if n < len(folds):
                m = np.mean([x["mean"] for x in d["folds"]])
                mb = np.mean([x["mean"] for x in base["folds"][:n]])
                lab = f"{m - mb:+.4f} ({n}/{len(folds)} folds)"
            else:
                lab = f"{d['loo_mean']:.4f} ({d['loo_mean'] - base['loo_mean']:+.4f})"
            cells = [f"{x['mean'] - b['mean']:+.3f}" for x, b in zip(d["folds"], base["folds"])]
            cells += [""] * (len(folds) - n)
            print(f"| {v} | {len(r['pins'][v])} | {lab} | " + " | ".join(cells) + " |")
        sp = r.get("base_fold_spread")
        pins = set()
        for v in r["pins"].values():
            pins |= set(v)
        if sp and pins:
            print("\nFold spread (base) of the flagged parameters: " + ", ".join(
                f"{n} {sp[n]['min']:.4g}-{sp[n]['max']:.4g}" for n in sorted(pins)))


if __name__ == "__main__":
    main()
