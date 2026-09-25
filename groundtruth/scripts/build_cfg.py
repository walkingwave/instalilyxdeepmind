"""Build a submission from an explicit per-system configuration (one JSON file).

    python scripts/build_cfg.py --tag u006 --cfg plans/u006.json [--include-val]

cfg JSON: {system: {"kind": "l0b_lin"|"l0b"|"l1"|"l2"|"l0a", "cfg": {...model kwargs...},
                    "lam": 1.0, "clip_margin": 3.0, "post": [rules] or null (null = default rules)}}
Every system in gtlab.systems.SYSTEM_IDS must be present (all-10 rule). The config is copied
into the build folder as config.json so the version is reproducible.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import systems as S, metric, select as SEL
from gtlab.models import common as C
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.package import build


def fit_system(sid, spec_cfg, include_val=False, time_budget_s=120):
    spec = S.get(sid)
    runs = load_runs(spec, Ledger(data_dir(sid, False), sid, False))
    if not include_val:
        runs = [r for r in runs if r.tags.get("split") != "val"]
    kind = spec_cfg.get("kind", "l0b_lin")
    margin = spec_cfg.get("clip_margin")
    clip = C.soft_clip(spec, runs, margin=margin) if runs else None
    if kind == "l0a" or not runs:
        doc = C.make_doc(spec, {"kind": "l0a"}, runs=runs or None, clip=clip, info={"model_id": "l0a"})
    else:
        sigma = metric.sigma_proxy(runs)
        cfg = dict(spec_cfg.get("cfg") or {})
        if kind == "l0b_lin":
            cfg = dict({"sq": False, "pairs": None}, **cfg)
            m = SEL.make_model("l0b", spec, clip, sigma, cfg=cfg)
        else:
            m = SEL.make_model(kind, spec, clip, sigma, cfg=cfg, time_budget_s=time_budget_s)
        m.fit(runs)
        doc = m.doc()
    lam = float(spec_cfg.get("lam", 1.0))
    if abs(lam - 1.0) > 1e-9 and doc["model"].get("kind") != "l0a":
        doc["model"] = {"kind": "blend", "lam": lam, "member": doc["model"]}
    if spec_cfg.get("post") is not None:
        doc["post"] = spec_cfg["post"]
    doc["info"] = dict(doc.get("info") or {}, model_id=f"{kind}@{lam:g}", n_runs=len(runs),
                       n_ticks=int(sum(r.T for r in runs)))
    return SEL._jsonable(doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--include-val", action="store_true")
    a = ap.parse_args()
    cfg = json.loads(Path(a.cfg).read_text())
    missing = [s for s in S.SYSTEM_IDS if s not in cfg]
    if missing:
        raise SystemExit(f"config must cover all 10 systems; missing {missing}")
    picks = {}
    for sid in S.SYSTEM_IDS:
        picks[sid] = fit_system(sid, cfg[sid], a.include_val)
        print(sid, picks[sid]["info"].get("model_id"), cfg[sid].get("cfg") or "", "post" if picks[sid].get("post") else "", flush=True)
    out = build(list(picks), a.tag, picks)
    shutil.copy(a.cfg, out / "config.json")
    print(out)


if __name__ == "__main__":
    main()
