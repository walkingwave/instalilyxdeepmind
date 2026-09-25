"""Fit one model kind per system on whatever real data exists and package a submission ZIP.

Systems with no data fall back to persistence (l0a). No gateway calls, no spending.

  python scripts/build_from_data.py --tag probe-l0b --kind l0b_lin
  python scripts/build_from_data.py --tag mix --picks '{"power_grid":"l0b_lin","wildlife":"l0a"}'
  python scripts/build_from_data.py --mock --data-root /tmp/mockroot --tag mocktest   # mock data

Kinds: l0a, l0b (quadratic+pair features), l0b_lin (linear features), l1.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import systems as S, metric
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab import select as SEL
from gtlab.models import common as C
from gtlab.package import build


def fit_doc(sid, kind, runs, time_budget_s=120, include_val=False):
    """Fit `kind` on the train runs (val.* runs are held out unless include_val, e.g. the final refit)."""
    spec = S.get(sid)
    if not include_val:
        runs = [r for r in runs if r.tags.get("split") != "val"]
    if not runs or kind == "l0a":
        return C.make_doc(spec, {"kind": "l0a"}, runs=runs or None, info={"model_id": "l0a"}), "l0a"
    clip = C.soft_clip(spec, runs)
    sigma = metric.sigma_proxy(runs)
    if kind == "l0b_lin":
        m = SEL.make_model("l0b", spec, clip, sigma, cfg={"sq": False, "pairs": None})
    else:
        m = SEL.make_model(kind, spec, clip, sigma, time_budget_s=time_budget_s)
    m.fit(runs)
    doc = m.doc()
    doc["info"] = dict(doc.get("info") or {}, model_id=kind, n_runs=len(runs),
                       n_ticks=int(sum(r.T for r in runs)))
    return SEL._jsonable(doc), kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--kind", default="l0b_lin")
    ap.add_argument("--picks", help="JSON {system: kind} overriding --kind per system")
    ap.add_argument("--systems", default=",".join(S.SYSTEM_IDS))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--include-val", action="store_true", help="final refit: train on held-out val runs too")
    a = ap.parse_args()
    over = json.loads(a.picks) if a.picks else {}
    picks, summary = {}, {}
    for sid in a.systems.split(","):
        spec = S.get(sid)
        d = data_dir(sid, a.mock, a.data_root)
        runs = []
        if (d / "ledger.jsonl").exists():
            runs = load_runs(spec, Ledger(d, sid, mock=a.mock))
        kind = over.get(sid, a.kind)
        doc, used = fit_doc(sid, kind, runs, include_val=a.include_val)
        picks[sid] = doc
        summary[sid] = {"kind": used, "runs": len(runs), "ticks": int(sum(r.T for r in runs))}
        print(sid, summary[sid], flush=True)
    out = build(list(picks), a.tag, picks)
    (out / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(out)


if __name__ == "__main__":
    main()
