"""Build submission folders + a deterministic submission.zip (PLAN section 6).

    build(systems, tag, picks) -> Path of submissions/<YYYYMMDD-HHMM>-<tag>/
    build_persistence_all(tag) -> zero-training L0a build for all 10 systems

picks[system] may be a doc dict (see gtlab.runtime.infer), a model blob dict (wrapped with
spec meta and hard-bound-only clipping), a path to a model.json, or a directory holding one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

from gtlab import systems as S
from gtlab.models.common import make_doc

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = Path(__file__).resolve().parent / "runtime"
ODE_DIR = Path(__file__).resolve().parent / "ode"
SUBMISSIONS = ROOT / "submissions"
ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def _as_doc(system_id, pick):
    spec = S.get(system_id)
    if isinstance(pick, (str, Path)):
        p = Path(pick)
        if p.is_dir():
            p = p / "model.json"
        pick = json.loads(p.read_text())
    if not isinstance(pick, dict):
        raise TypeError(f"bad pick for {system_id}: {type(pick)}")
    if "model" in pick and "observables" in pick:
        doc = dict(pick)
    else:
        doc = make_doc(spec, pick, runs=None)
    if doc.get("system") != system_id:
        raise ValueError(f"pick for {system_id} belongs to {doc.get('system')}")
    if list(doc["observables"]) != list(spec.observables) or list(doc["controls"]) != list(spec.controls):
        raise ValueError(f"pick for {system_id} has mismatched observables/controls")
    return doc


def ode_families(blob):
    """All ODE family names referenced by a (possibly nested) model blob."""
    out = set()
    if isinstance(blob, dict):
        if blob.get("kind") == "ode":
            out.add(blob["family"])
        for m in blob.get("members", []) or []:
            out |= ode_families(m)
        if isinstance(blob.get("member"), dict):
            out |= ode_families(blob["member"])
    return out


def render_predict(spec):
    text = (RUNTIME / "predict_template.py").read_text()
    hard = {o: list(spec.output_bounds(o)) for o in spec.observables}
    text = text.replace('SYSTEM = "__SYSTEM__"', f"SYSTEM = {spec.id!r}", 1)
    text = text.replace("HARD = {}", f"HARD = {hard!r}", 1)
    assert "__SYSTEM__" not in text
    return text


def write_system_folder(folder: Path, system_id, doc):
    """Exactly two files: predict.py (runtime inlined, no comments) and model.json (numbers only)."""
    from gtlab import flatpack
    flatpack.write_folder(folder, S.get(system_id), doc, ode_families(doc["model"]))


def write_zip(build_dir: Path, systems, zip_path: Path):
    """Deterministic zip: sorted entries, fixed timestamps/permissions, no caches."""
    entries = []
    for sid in sorted(systems):
        for f in sorted((build_dir / sid).rglob("*")):
            rel = f.relative_to(build_dir).as_posix()
            if "__pycache__" in rel or rel.endswith((".pyc", ".DS_Store")):
                continue
            entries.append((rel + "/", None) if f.is_dir() else (rel, f))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for sid in sorted(systems):
            zi = zipfile.ZipInfo(sid + "/", date_time=ZIP_DATE)
            zi.external_attr = (0o40755 << 16) | 0x10
            zf.writestr(zi, b"")
        for rel, f in entries:
            if f is None:
                continue
            zi = zipfile.ZipInfo(rel, date_time=ZIP_DATE)
            zi.external_attr = 0o100644 << 16
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, f.read_bytes(), compresslevel=9)
    return zip_path


def build(systems, tag, picks, stamp=None, out_root=None):
    systems = list(systems)
    for s in systems:
        if s not in S.SYSTEM_IDS:
            raise ValueError(f"unknown system {s}")
    stamp = stamp or time.strftime("%Y%m%d-%H%M")
    out_root = Path(out_root) if out_root else SUBMISSIONS
    build_dir = out_root / f"{stamp}-{tag}"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    manifest = {"tag": tag, "stamp": stamp, "git_sha": _git_sha(), "systems": {}}
    for sid in systems:
        doc = _as_doc(sid, picks[sid])
        doc.setdefault("info", {})
        doc["info"] = dict(doc["info"], git_sha=manifest["git_sha"], tag=tag)
        write_system_folder(build_dir / sid, sid, doc)
        manifest["systems"][sid] = {"kind": doc["model"].get("kind"),
                                    "model_id": doc["info"].get("model_id", doc["model"].get("kind")),
                                    "ode": sorted(ode_families(doc["model"]))}
    zp = write_zip(build_dir, systems, build_dir / "submission.zip")
    manifest["zip_sha256"] = hashlib.sha256(zp.read_bytes()).hexdigest()
    manifest["zip_bytes"] = zp.stat().st_size
    (build_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return build_dir


def build_persistence_all(tag="persist", stamp=None, out_root=None):
    """Zero-training L0a submission for all 10 systems (clipped to hard bounds only)."""
    picks = {sid: make_doc(S.get(sid), {"kind": "l0a"}, runs=None,
                           info={"model_id": "l0a"}) for sid in S.SYSTEM_IDS}
    return build(S.SYSTEM_IDS, tag, picks, stamp=stamp, out_root=out_root)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m gtlab.package")
    ap.add_argument("--persistence", action="store_true", help="build the L0a zip for all systems")
    ap.add_argument("--tag", default="persist")
    ap.add_argument("--pick", action="append", default=[],
                    help="SYSTEM=path/to/model.json (or dir); repeatable")
    a = ap.parse_args(argv)
    if a.persistence:
        d = build_persistence_all(a.tag)
    else:
        picks = dict(p.split("=", 1) for p in a.pick)
        d = build(list(picks), a.tag, picks)
    print(d / "submission.zip")


if __name__ == "__main__":
    main()
