"""Assemble a submission from the best-scoring version of every system across all uploads.

Reads tune/registry.json (upload id -> dir, per-system public scores), picks per system the
upload with the highest score (or a forced id via --pin system=uid), copies that system folder
verbatim, zips. Any old model can be revived this way without refitting.

    python scripts/best_of.py --tag friday-best [--pin epidemic=u004 --pin market=u003]
    python scripts/best_of.py --list        # table of every scored version per system
"""
import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import systems as S

ROOT = Path(__file__).resolve().parent.parent
REG = ROOT / "tune" / "registry.json"


def versions():
    reg = json.loads(REG.read_text())
    out = {s: [] for s in S.SYSTEM_IDS}
    for u in reg["uploads"]:
        d = ROOT / u["dir"]
        for s, sc in (u.get("scores") or {}).items():
            cfg = (u.get("systems") or {}).get(s)
            out[s].append({"id": u["id"], "date": u["date"], "score": float(sc), "dir": d / s,
                           "config": cfg, "exists": (d / s / "predict.py").exists()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="best")
    ap.add_argument("--pin", action="append", default=[], help="system=upload_id to force a version")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    V = versions()
    if a.list:
        for s in S.SYSTEM_IDS:
            rows = sorted(V[s], key=lambda r: -r["score"])
            print(s + ":  " + "  ".join(f"{r['id']}={r['score']:.4f}{'' if r['exists'] else '(missing)'}" for r in rows))
        return
    pins = dict(p.split("=", 1) for p in a.pin)
    stamp = time.strftime("%Y%m%d-%H%M")
    out = ROOT / "submissions" / f"{stamp}-{a.tag}"
    out.mkdir(parents=True)
    chosen = {}
    for s in S.SYSTEM_IDS:
        rows = [r for r in V[s] if r["exists"]]
        if s in pins:
            rows = [r for r in rows if r["id"] == pins[s]]
        if not rows:
            raise SystemExit(f"{s}: no scored version on disk")
        r = max(rows, key=lambda r: r["score"])
        shutil.copytree(r["dir"], out / s)
        chosen[s] = {"from": r["id"], "score": r["score"], "config": r["config"]}
        print(f"{s:<17} {r['id']}  {r['score']:.4f}  {r['config']}")
    (out / "manifest.json").write_text(json.dumps(chosen, indent=1))
    zp = out / "submission.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for s in S.SYSTEM_IDS:
            for f in sorted((out / s).iterdir()):
                z.write(f, f"{s}/{f.name}")
    print("mean of chosen:", round(sum(c["score"] for c in chosen.values()) / len(chosen), 4))
    print(zp)


if __name__ == "__main__":
    main()
