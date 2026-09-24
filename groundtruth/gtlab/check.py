"""Local contract checker (PLAN section 6). `python -m gtlab.check <zip> [--episodes 40]`.

Hard-fails on: zip layout/sizes, secrets, non-whitelisted imports / file writes, and a sandboxed
run of every system folder (pinned py3.12 venv, sockets disabled, RLIMIT_AS 3 GiB, 2 CPUs, cwd
elsewhere) with 40 x 4000-step eval-like episodes, output-contract asserts, fresh-state
determinism, input immutability, timing and peak RSS. Writes check_report.json beside the zip.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

from gtlab import systems as S

ROOT = Path(__file__).resolve().parent.parent
SANDBOX_PY = os.environ.get("GT_SANDBOX_PY", sys.executable)
MAX_ZIP = 30 * 2 ** 20
MAX_EXPANDED = 300 * 2 ** 20
TIME_LIMIT_S = 400.0
RSS_LIMIT_MB = 1536.0
MEM_LIMIT = 3 * 2 ** 30

ALLOWED_IMPORTS = {
    "numpy", "scipy", "sklearn", "joblib",
    "json", "math", "pathlib", "os.path", "functools", "itertools", "typing", "dataclasses",
    "warnings", "collections", "importlib", "importlib.util", "__future__", "re", "copy",
    "bisect", "heapq", "operator", "numbers", "abc", "enum",
}
BANNED_NAMES = {"torch", "httpx", "requests", "socket", "urllib", "subprocess", "multiprocessing",
                "http", "ssl", "ftplib", "smtplib", "asyncio", "ctypes", "pickle", "shutil"}
SECRET_PATTERNS = [b"GROUNDTRUTH_KEY", b"GEMMA_API_KEY", b"Bearer ", b"AIza"]
BANNED_FILES = re.compile(r"(^|/)(\.env[^/]*|venv/|\.venv/|site-packages/|__pycache__/|.*\.pyc$|\.git/)")


# ----------------------------------------------------------------------------- static checks
def _import_ok(mod):
    if mod in ALLOWED_IMPORTS:
        return True
    top = mod.split(".")[0]
    if top in {"numpy", "scipy", "sklearn", "joblib", "importlib", "collections"}:
        return True
    return False


def scan_python(src: str, fname: str):
    errors = []
    try:
        tree = ast.parse(src, filename=fname)
    except SyntaxError as e:
        return [f"{fname}: syntax error {e}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if not _import_ok(a.name):
                    errors.append(f"{fname}:{node.lineno}: import {a.name} not whitelisted")
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                errors.append(f"{fname}:{node.lineno}: relative import not allowed in submission")
            elif not _import_ok(node.module or "") and not (
                    node.module == "os" and all(a.name == "path" for a in node.names)):
                errors.append(f"{fname}:{node.lineno}: from {node.module} import not whitelisted")
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")
            if name in {"__import__", "eval", "exec", "system", "popen"}:
                errors.append(f"{fname}:{node.lineno}: call {name}() not allowed")
            if name == "open":
                mode = None
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    mode = node.args[1].value
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode = kw.value.value
                if mode is None and (len(node.args) >= 2 or any(k.arg == "mode" for k in node.keywords)):
                    errors.append(f"{fname}:{node.lineno}: open() with non-constant mode")
                elif isinstance(mode, str) and any(c in mode for c in "wax+"):
                    errors.append(f"{fname}:{node.lineno}: open() for writing")
            if name in {"write_text", "write_bytes", "mkdir", "unlink", "rmdir", "rename", "savez",
                        "save", "savetxt", "dump"}:
                errors.append(f"{fname}:{node.lineno}: write call {name}() not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            errors.append(f"{fname}:{node.lineno}: banned name {node.id}")
    return errors


def static_checks(zpath: Path):
    errors, warnings = [], []
    info = {"zip_bytes": zpath.stat().st_size}
    if info["zip_bytes"] > MAX_ZIP:
        errors.append(f"zip is {info['zip_bytes']} bytes > 30 MiB")
    try:
        zf = zipfile.ZipFile(zpath)
    except zipfile.BadZipFile as e:
        return [f"zip does not open: {e}"], warnings, info, []
    with zf:
        names = zf.namelist()
        expanded = sum(i.file_size for i in zf.infolist())
        info["expanded_bytes"] = expanded
        if expanded > MAX_EXPANDED:
            errors.append(f"expanded size {expanded} > 300 MiB")
        roots = sorted({n.split("/", 1)[0] for n in names})
        for n in names:
            if "/" not in n.rstrip("/"):
                if not n.endswith("/"):
                    errors.append(f"file at zip root: {n}")
            if BANNED_FILES.search(n):
                errors.append(f"banned file in zip: {n}")
            if n.startswith("/") or ".." in n.split("/"):
                errors.append(f"unsafe path: {n}")
        systems = [r for r in roots if r]
        for r in systems:
            if r not in S.SYSTEM_IDS:
                errors.append(f"root entry {r!r} is not a system id")
            elif f"{r}/predict.py" not in names:
                errors.append(f"{r}/predict.py missing")
        if not systems:
            errors.append("zip contains no system folders")
        env_secrets = [os.environ.get(k, "").encode() for k in ("GROUNDTRUTH_KEY", "GEMMA_API_KEY")]
        env_secrets = [s for s in env_secrets if len(s) >= 8]
        for n in names:
            if n.endswith("/"):
                continue
            data = zf.read(n)
            for pat in SECRET_PATTERNS + env_secrets:
                if pat in data:
                    shown = pat.decode(errors="replace") if pat in SECRET_PATTERNS else "<env secret value>"
                    errors.append(f"secret pattern {shown!r} in {n}")
            if n.endswith(".py"):
                errors.extend(scan_python(data.decode("utf-8", errors="replace"), n))
            if n.endswith(".json"):
                try:
                    json.loads(data)
                except Exception as e:
                    errors.append(f"{n}: invalid json ({e})")
    systems = [s for s in systems if s in S.SYSTEM_IDS]
    return errors, warnings, info, systems


# ----------------------------------------------------------------------------- episodes
def _simple_eval_like(spec, category, T, rng):
    lo, hi = np.array(spec.lo()), np.array(spec.hi())
    rec = np.array([spec.recovery[c] for c in spec.controls])
    pul = np.array([spec.pulse[c] for c in spec.controls])
    U = np.tile(rec, (T, 1))
    t = 0
    while t < T:
        d = int(rng.integers(5, 400))
        if category == "recovery":
            alpha = rng.uniform(0.7, 1.0, size=len(rec))
            lvl = rec + alpha * (pul - rec) if rng.random() < 0.4 else rec
        else:
            lvl = rng.uniform(lo, hi)
        U[t:t + d] = lvl
        t += d
    return np.clip(U, lo, hi)


def schedules(spec, n, T, seed=0):
    rng = np.random.default_rng(seed)
    try:
        from gtlab.design import eval_like
    except Exception:
        eval_like = None
    cats = ["sustained", "order", "recovery", "composition"]
    out = []
    for k in range(n):
        if k == n - 2:
            U = np.tile(np.array(spec.lo(), float), (T, 1)); cat = "pinned_lo"
        elif k == n - 1:
            U = np.tile(np.array(spec.hi(), float), (T, 1)); cat = "pinned_hi"
        else:
            cat = cats[k % 4]
            try:
                U = eval_like(spec, cat, T, rng) if eval_like else _simple_eval_like(spec, cat, T, rng)
            except Exception:
                U = _simple_eval_like(spec, cat, T, rng)
        U = np.asarray(U, float)
        assert U.shape == (T, spec.m), U.shape
        out.append((cat, U))
    return out


def initial_pool(spec, base=None):
    """Initial observations from data/<sys>/{resets,ledger}.jsonl, else data_mock, else synthetic."""
    base = Path(base) if base else ROOT
    pool = []
    for sub in ("data", "data_mock"):
        for fn in ("resets.jsonl", "ledger.jsonl"):
            p = base / sub / spec.id / fn
            if not p.exists():
                continue
            for line in p.read_text().splitlines():
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                obs = r.get("obs")
                if (r.get("t") in (None, "reset", "shop") and isinstance(obs, dict)
                        and all(o in obs for o in spec.observables)):
                    pool.append({o: float(obs[o]) for o in spec.observables})
        if pool:
            return pool, sub
    return [], "synthetic"


def initial_conditions(spec, n, seed=0, base=None):
    pool, src = initial_pool(spec, base)
    rng = np.random.default_rng(seed + 1)
    ranges = None
    if not pool:
        try:
            from gtlab.mocks.base import Y0_RANGE
            ranges = Y0_RANGE.get(spec.id)
            if ranges is not None and len(ranges) == spec.p:
                src = "mock_y0_range"
            else:
                ranges = None
        except Exception:
            ranges = None
    out = []
    for k in range(n):
        if pool:
            out.append(dict(pool[k % len(pool)]))
            continue
        if ranges is not None:
            out.append({o: float(rng.uniform(a, b)) for o, (a, b) in zip(spec.observables, ranges)})
            continue
        d = {}
        for o in spec.observables:
            lo, hi = spec.output_bounds(o)
            if hi is not None and hi <= 1.0:
                d[o] = float(rng.uniform(0.05, 0.95))
            elif o == "frequency":
                d[o] = float(50.0 + rng.normal(0, 0.05))
            else:
                d[o] = float(np.exp(rng.uniform(np.log(0.5), np.log(500.0))))
        out.append(d)
    return out, src


def make_episodes(spec, n=40, T=4000, seed=0, base=None):
    scheds = schedules(spec, n, T, seed)
    inits, src = initial_conditions(spec, n, seed, base)
    eps = []
    for (cat, U), y0 in zip(scheds, inits):
        eps.append({"category": cat, "initial": y0,
                    "interventions": [dict(zip(spec.controls, map(float, row))) for row in U]})
    return eps, src


# ----------------------------------------------------------------------------- sandbox runner
RUNNER = r'''
import copy, hashlib, importlib.util, json, math, resource, socket, sys, time
def _no_net(*a, **k):
    raise OSError("network disabled in checker sandbox")
socket.socket = _no_net
socket.create_connection = _no_net
socket.getaddrinfo = _no_net
folder, epfile, outfile = sys.argv[1], sys.argv[2], sys.argv[3]
job = json.load(open(epfile))
context, hard, eps = job["context"], job["hard"], job["episodes"]
t0 = time.time()
rep = {"errors": [], "warnings": [], "episodes": []}
try:
    spec = importlib.util.spec_from_file_location("predict", folder + "/predict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
except Exception as e:
    rep["errors"].append("import failed: %r" % (e,))
    json.dump(rep, open(outfile, "w")); sys.exit(0)
rep["load_s"] = time.time() - t0

def run(i):
    ep = eps[i]
    init_c, inter_c, ctx_c = copy.deepcopy(ep["initial"]), copy.deepcopy(ep["interventions"]), copy.deepcopy(context)
    ts = time.time()
    try:
        out = mod.predict(ep["initial"], ep["interventions"], context)
    except Exception as e:
        return {"error": "predict raised %r" % (e,)}, None, time.time() - ts
    dt = time.time() - ts
    errs, warns = [], []
    if ep["initial"] != init_c or ep["interventions"] != inter_c or context != ctx_c:
        errs.append("inputs mutated")
    names = list(init_c.keys())
    if not isinstance(out, list):
        errs.append("returned %s not list" % type(out).__name__)
        return {"error": "; ".join(errs)}, None, dt
    if len(out) != len(inter_c):
        errs.append("len %d != %d" % (len(out), len(inter_c)))
    h = hashlib.sha256()
    nonfloat = 0
    viol = 0
    for t, row in enumerate(out):
        if not isinstance(row, dict) or set(row.keys()) != set(names):
            errs.append("tick %d: keys %r" % (t, list(row.keys()) if isinstance(row, dict) else type(row)))
            break
        for k in names:
            v = row[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                if type(v).__module__ == "numpy":
                    nonfloat += 1
                    v = float(v)
                else:
                    errs.append("tick %d %s: non-numeric %r" % (t, k, type(v))); break
            if not math.isfinite(v):
                errs.append("tick %d %s: nonfinite" % (t, k)); break
            lo, hi = hard.get(k, [None, None])
            if (lo is not None and v < lo - 1e-12) or (hi is not None and v > hi + 1e-12):
                viol += 1
            h.update(repr(float(v)).encode())
        if errs:
            break
    if viol:
        errs.append("%d hard-bound violations" % viol)
    if nonfloat:
        warns.append("%d numpy scalars (not python floats)" % nonfloat)
    res = {"category": ep.get("category"), "seconds": round(dt, 3)}
    if errs:
        res["error"] = "; ".join(errs[:5])
    if warns:
        res["warning"] = "; ".join(warns)
    return res, h.hexdigest(), dt

digests = []
for i in range(len(eps)):
    r, d, dt = run(i)
    rep["episodes"].append(r)
    digests.append(d)
if len(eps) >= 2:
    r, d, _ = run(0)
    if d is None or d != digests[0]:
        rep["errors"].append("determinism: episode A differs after running B..Z (A,B,A)")
    r, d2, _ = run(1)
    if d2 is None or d2 != digests[1]:
        rep["errors"].append("determinism: episode B differs on repeat")
rep["total_s"] = time.time() - t0
rep["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
json.dump(rep, open(outfile, "w"))
'''


def _limits():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT, MEM_LIMIT))


def run_system(folder: Path, spec, episodes, workdir: Path, timeout=1500):
    job = {"context": spec.context(documents=_documents(spec.id)),
           "hard": {o: list(spec.output_bounds(o)) for o in spec.observables},
           "episodes": episodes}
    assert len(job["context"]) == 7
    epfile = workdir / f"{spec.id}.episodes.json"
    outfile = workdir / f"{spec.id}.result.json"
    epfile.write_text(json.dumps(job))
    runner = workdir / "runner.py"
    runner.write_text(RUNNER)
    env = {"PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
           "MKL_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1", "HOME": str(workdir),
           "LANG": "C.UTF-8"}
    cmd = [SANDBOX_PY, "-I", "-B", str(runner), str(folder), str(epfile), str(outfile)]
    if shutil.which("taskset"):
        cmd = ["taskset", "-c", "0,1"] + cmd
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True,
                              timeout=timeout, preexec_fn=_limits)
    except subprocess.TimeoutExpired:
        return {"errors": [f"timeout after {timeout}s"], "wall_s": timeout}
    wall = time.time() - t0
    if not outfile.exists():
        return {"errors": [f"runner crashed rc={proc.returncode}: {proc.stderr[-2000:]}"], "wall_s": wall}
    rep = json.loads(outfile.read_text())
    rep["wall_s"] = round(wall, 2)
    if proc.stderr.strip():
        rep.setdefault("warnings", []).append("stderr: " + proc.stderr[-500:])
    return rep


def _documents(sid):
    p = ROOT / "data" / sid / "docs.json"
    if p.exists():
        try:
            d = json.loads(p.read_text())
            docs = d.get("documents") if isinstance(d, dict) else d
            if isinstance(docs, dict):
                docs = docs.get("documents", [])
            return docs if isinstance(docs, list) else []
        except Exception:
            return []
    return []


def check(zip_path, episodes=40, T=4000, seed=0, systems=None, report_path=None, verbose=True):
    zpath = Path(zip_path).resolve()
    report = {"zip": str(zpath), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "episodes": episodes,
              "T": T, "sandbox_python": SANDBOX_PY, "systems": {}}
    errors, warnings, info, found = static_checks(zpath)
    report.update(info)
    report["static_errors"] = errors
    report["warnings"] = warnings
    if systems:
        found = [s for s in found if s in systems]
    tmp = Path(tempfile.mkdtemp(prefix="gtcheck_"))
    try:
        ext = tmp / "extract"
        ext.mkdir()
        if not errors or found:
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(ext)
        for sid in found:
            spec = S.get(sid)
            work = tmp / f"work_{sid}"
            work.mkdir()
            eps, src = make_episodes(spec, episodes, T, seed)
            rep = run_system(ext / sid, spec, eps, work)
            ep_err = [f"ep{i} ({e.get('category')}): {e['error']}" for i, e in
                      enumerate(rep.get("episodes", [])) if "error" in e]
            sys_err = list(rep.get("errors", [])) + ep_err
            total = rep.get("total_s", rep.get("wall_s", 0.0))
            if total > TIME_LIMIT_S:
                sys_err.append(f"too slow: {total:.1f}s > {TIME_LIMIT_S}s")
            if rep.get("peak_rss_mb", 0) > RSS_LIMIT_MB:
                sys_err.append(f"peak RSS {rep['peak_rss_mb']:.0f} MB > {RSS_LIMIT_MB} MB")
            if len(rep.get("episodes", [])) != episodes:
                sys_err.append("not all episodes ran")
            ep_s = [e.get("seconds", 0.0) for e in rep.get("episodes", [])]
            ep_warn = sorted({e["warning"] for e in rep.get("episodes", []) if "warning" in e})
            report["systems"][sid] = {
                "pass": not sys_err, "errors": sys_err[:20], "warnings": rep.get("warnings", []) + ep_warn,
                "load_s": round(rep.get("load_s", 0.0), 3), "total_s": round(total, 2),
                "wall_s": rep.get("wall_s"), "max_episode_s": round(max(ep_s), 3) if ep_s else None,
                "mean_episode_s": round(float(np.mean(ep_s)), 3) if ep_s else None,
                "peak_rss_mb": round(rep.get("peak_rss_mb", 0.0), 1), "initial_source": src,
            }
            if verbose:
                r = report["systems"][sid]
                print(f"  {sid:18s} {'PASS' if r['pass'] else 'FAIL'}  total {r['total_s']:7.2f}s  "
                      f"max ep {r['max_episode_s']}s  rss {r['peak_rss_mb']} MB  init={src}"
                      + ("" if r["pass"] else f"  {r['errors'][:3]}"), flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    report["pass"] = (not errors) and bool(report["systems"]) and all(
        r["pass"] for r in report["systems"].values())
    rp = Path(report_path) if report_path else zpath.with_name("check_report.json")
    rp.write_text(json.dumps(report, indent=2))
    report["report_path"] = str(rp)
    if verbose:
        for e in errors:
            print("  STATIC:", e)
        print(("PASS" if report["pass"] else "FAIL"), "->", rp)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m gtlab.check")
    ap.add_argument("zip")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--T", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--systems", nargs="*")
    a = ap.parse_args(argv)
    zp = Path(a.zip)
    if zp.is_dir():
        zp = zp / "submission.zip"
    rep = check(zp, a.episodes, a.T, a.seed, a.systems)
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
