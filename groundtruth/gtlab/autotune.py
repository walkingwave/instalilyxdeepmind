"""Leaderboard feedback loop: public scores in, next submission ZIP out.

Zero simulator credits. It only refits on data already bought and uses the public score of each
upload as a noiseless, out-of-sample signal to choose the next thing to try, per system.

Workflow (PowerShell or bash, from the groundtruth/ folder):
    python -m gtlab.autotune register submissions/<build>      # a ZIP you uploaded (e.g. persistence)
    python -m gtlab.autotune score u001 scores.txt              # paste scores: "power_grid 0.61" per line
    python -m gtlab.autotune propose                            # builds + checks the next ZIP
    python -m gtlab.autotune uploaded u002                      # after uploading it
    python -m gtlab.autotune score u002 scores.txt
    python -m gtlab.autotune status
    python -m gtlab.autotune propose --final                    # best-known config for every system

Per-system policy (deterministic, explained in each proposal's "why"):
  1. Score persistence (l0a) once: it is the floor and the lam=0 end of every blend.
  2. Try each available fitted kind at lam=1 (kinds unlock with data: l0b_lin >0 ticks,
     l1 >= 600 ticks, l2 >= 1000 ticks, ode when --with-ode and a fit exists).
  3. Take the best fitted kind k*. Tune lam in y = y0 + lam*(k* - y0), lam in [0, 1.3]:
     second point is 0.5 (if k* lost to persistence) or 0.75 (if it won); after that fit a
     parabola through the best three (lam, score) points and try its peak, rounded to 0.05.
     Stop when the peak is within 0.05 of a tried lam.
  4. Otherwise re-submit nothing: the system keeps its live model and saves a slot.
When the data grows (new ticks), fitted-kind scores are stale: step 2 reruns for the best kinds.
The public set is 40 fixed episodes; with ~1-2 knobs per system the overfitting risk is small.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from gtlab import systems as S

ROOT = Path(__file__).resolve().parent.parent
TUNE = ROOT / "tune"
REG = TUNE / "registry.json"
MOCK = False          # tests/simulation only: read data_mock/ instead of data/
KIND_ORDER = ["l0b_lin", "l1", "l2", "ode"]
MIN_TICKS = {"l0b_lin": 1, "l1": 600, "l2": 1000, "ode": 1}
MAX_UPLOADS_PER_DAY = 3
LAM_MAX = 1.3


# ----------------------------------------------------------------------------- registry
def toronto_today():
    # America/Toronto is UTC-4 (EDT) for the whole competition window.
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=4)).date().isoformat()


def load():
    if REG.exists():
        return json.loads(REG.read_text())
    return {"uploads": []}


def save(reg):
    TUNE.mkdir(exist_ok=True)
    tmp = REG.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=1, sort_keys=True))
    tmp.replace(REG)


def next_id(reg):
    return f"u{len(reg['uploads']) + 1:03d}"


def cfg_key(c):
    if c["kind"] == "l0a":
        return ("l0a", 0.0, 0)
    return (c["kind"], round(float(c.get("lam", 1.0)), 3), int(c.get("data_ticks", 0)))


def history(reg, sid):
    """[(config, score)] for scored uploads that included sid, oldest first."""
    out = []
    for u in reg["uploads"]:
        if sid in u["systems"] and u.get("scores", {}).get(sid) is not None:
            out.append((u["systems"][sid], float(u["scores"][sid])))
    return out


def live(reg, sid, phase="public"):
    for u in reversed(reg["uploads"]):
        if u.get("uploaded") and u.get("phase", "public") == phase and sid in u["systems"]:
            return u["systems"][sid]
    return None


def slots_used(reg, sid, day=None):
    day = day or toronto_today()
    return sum(1 for u in reg["uploads"] if u.get("uploaded") and u.get("date") == day and sid in u["systems"])


# ----------------------------------------------------------------------------- data
def data_ticks(sid, data_root=None):
    from gtlab.ledger import Ledger, data_dir, load_runs
    d = data_dir(sid, MOCK, data_root)
    if not (d / "ledger.jsonl").exists():
        return 0, []
    runs = load_runs(S.get(sid), Ledger(d, sid, mock=MOCK))
    return int(sum(r.T for r in runs)), runs


def available_kinds(ticks, with_ode=False):
    ks = [k for k in KIND_ORDER if ticks >= MIN_TICKS[k]]
    if not with_ode:
        ks = [k for k in ks if k != "ode"]
    return ks


# ----------------------------------------------------------------------------- local screen
SCREEN_MARGIN = 0.02      # a kind that loses to persistence locally by more than this never gets a slot
MIN_RUNS_VETO = 4         # below this many runs the leave-one-out screen is too noisy to veto anything


def local_screen(sid, kinds, runs, time_budget_s=60):
    """Leave-one-run-out score of each kind (and l0a) on our own data. Free, noisy, unlimited.
    Used to order candidates and to veto clear losers before they burn an upload slot."""
    from gtlab import metric
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_from_data import fit_doc
    from gtlab.runtime import infer as rt
    if len(runs) < 2:
        return {}
    sigma = metric.sigma_proxy(runs)
    out = {}
    for k in ["l0a"] + list(kinds):
        if k == "ode":
            continue
        sc = []
        for i, hold in enumerate(runs):
            train = [r for j, r in enumerate(runs) if j != i]
            try:
                doc = fit_doc(sid, k, train, time_budget_s=time_budget_s)[0]
                P = rt.rollout_from_blob(doc, hold.y0, hold.U, doc=doc, base_dir=ROOT / "gtlab" / "runtime")
                sc.append(metric.robust_score(P, hold.Y, sigma))
            except Exception:
                sc.append(0.0)
        out[k] = float(np.mean(sc))
    out["_n_runs"] = len(runs)
    return out


# ----------------------------------------------------------------------------- policy
def _best(pairs):
    return max(pairs, key=lambda p: p[1]) if pairs else None


def propose_one(hist, kinds, ticks, final=False, screen=None):
    """Returns (config, why). config None = keep the live model.
    screen: {kind: local leave-one-run-out score} orders untried kinds and vetoes clear losers."""
    screen = screen or {}
    if screen and "l0a" in screen and screen.get("_n_runs", 0) >= MIN_RUNS_VETO:
        base = screen["l0a"]
        kinds = sorted([k for k in kinds if k not in screen or screen[k] >= base - SCREEN_MARGIN],
                       key=lambda k: -screen.get(k, base))
    l0 = [s for c, s in hist if c["kind"] == "l0a"]
    cur = [(c, s) for c, s in hist if c["kind"] != "l0a" and int(c.get("data_ticks", 0)) == ticks]
    if final:
        pool = [(c, s) for c, s in hist if c["kind"] == "l0a" or int(c.get("data_ticks", 0)) == ticks]
        b = _best(pool) or _best(hist)
        if b is None:
            return {"kind": "l0a"}, "final: nothing scored yet, persistence"
        return dict(b[0], data_ticks=ticks if b[0]["kind"] != "l0a" else 0), f"final: best scored {b[1]:.4f}"
    if not l0:
        return {"kind": "l0a"}, "score persistence first (floor, lam=0 anchor)"
    s0 = max(l0)
    for k in kinds:
        if not any(c["kind"] == k and abs(float(c.get("lam", 1)) - 1.0) < 1e-9 for c, _ in cur):
            loc = f", local {screen[k]:.3f} vs persistence {screen.get('l0a', float('nan')):.3f}" if k in screen else ""
            return {"kind": k, "lam": 1.0, "data_ticks": ticks}, f"untried kind {k} on {ticks} ticks{loc}"
    full = [(c, s) for c, s in cur if abs(float(c.get("lam", 1)) - 1.0) < 1e-9]
    if not full:
        return None, "no fitted kind available (no data); keep live model"
    kstar, sstar = _best(full)
    k = kstar["kind"]
    pts = {0.0: s0}
    for c, s in cur:
        if c["kind"] == k:
            lam = round(float(c.get("lam", 1.0)), 3)
            pts[lam] = max(s, pts.get(lam, -1))
    if len(pts) == 2:
        lam = 0.75 if sstar > s0 else 0.5
        return {"kind": k, "lam": lam, "data_ticks": ticks}, \
            f"{k}@1 {'beat' if sstar > s0 else 'lost to'} persistence ({sstar:.4f} vs {s0:.4f}); probe lam={lam}"
    xs = np.array(sorted(pts))
    ys = np.array([pts[x] for x in xs])
    top = np.argsort(ys)[-3:]
    x3, y3 = xs[top], ys[top]
    lam_star = float(xs[np.argmax(ys)])
    if len(set(np.round(x3, 3))) == 3:
        a, b, _ = np.polyfit(x3, y3, 2)
        if a < 0:
            lam_star = float(np.clip(-b / (2 * a), 0.0, LAM_MAX))
    lam_star = round(round(lam_star / 0.05) * 0.05, 2)
    if min(abs(lam_star - x) for x in xs) < 0.05 - 1e-9:
        bl = float(xs[np.argmax(ys)])
        if bl == 0.0:
            return {"kind": "l0a"}, f"lam search for {k} converged at 0: persistence is best ({s0:.4f})"
        return {"kind": k, "lam": bl, "data_ticks": ticks}, \
            f"lam search for {k} converged at {bl} (best {ys.max():.4f}); make it live"
    return {"kind": k, "lam": lam_star, "data_ticks": ticks}, \
        f"parabola through best 3 of {len(xs)} points -> lam={lam_star}"


# ----------------------------------------------------------------------------- build
def _fit_doc(sid, cfg, runs, cache, include_val=False):
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_from_data import fit_doc
    if cfg["kind"] == "l0a":
        doc, _ = fit_doc(sid, "l0a", runs)
        doc["info"] = dict(doc.get("info") or {}, model_id="l0a")
        return doc
    key = (sid, cfg["kind"], cfg.get("data_ticks"), include_val)
    if key not in cache:
        cache[key] = fit_doc(sid, cfg["kind"], runs, include_val=include_val)[0]
    base = json.loads(json.dumps(cache[key]))
    lam = float(cfg.get("lam", 1.0))
    if abs(lam - 1.0) > 1e-9:
        base["model"] = {"kind": "blend", "lam": lam, "member": base["model"]}
    base["info"] = dict(base.get("info") or {}, model_id=f"{cfg['kind']}@{lam:g}")
    return base


def cmd_propose(a):
    from gtlab.package import build
    reg = load()
    picks, plan, cache = {}, {}, {}
    for sid in S.SYSTEM_IDS:
        ticks, runs = data_ticks(sid, a.data_root)
        kinds = available_kinds(ticks, a.with_ode)
        screen = {} if (a.final or a.no_screen) else local_screen(sid, kinds, runs)
        cfg, why = propose_one(history(reg, sid), kinds, ticks, final=a.final, screen=screen)
        lv = live(reg, sid, "final" if a.final else "public")
        if cfg is not None and lv is not None and cfg_key(cfg) == cfg_key(lv):
            cfg, why = None, why + " (already live)"
        used = slots_used(reg, sid)
        if cfg is not None and used >= MAX_UPLOADS_PER_DAY:
            cfg, why = None, f"no slots left today ({used}/3); wanted: {why}"
        plan[sid] = {"config": cfg, "why": why, "slots_used_today": used, "screen": screen}
        if cfg is not None:
            picks[sid] = _fit_doc(sid, cfg, runs, cache, include_val=a.final)
    for sid, p in plan.items():
        print(f"{sid:17s} {'SKIP' if p['config'] is None else json.dumps(p['config']):52s} {p['why']}")
    if not picks:
        print("nothing new to upload")
        return 0
    uid = next_id(reg)
    out = build(list(picks), f"{uid}-{'final' if a.final else 'auto'}", picks)
    ok = True
    if not a.no_check:
        ok = run_check(out / "submission.zip")
    reg["uploads"].append({"id": uid, "date": toronto_today(), "phase": "final" if a.final else "public",
                           "dir": str(out.relative_to(ROOT)), "systems": {s: plan[s]["config"] for s in picks},
                           "why": {s: plan[s]["why"] for s in picks}, "uploaded": False, "scores": {},
                           "local": {s: plan[s]["screen"] for s in picks if plan[s]["screen"]},
                           "check": "pass" if ok else "FAIL"})
    save(reg)
    print(f"\n{uid}: {out / 'submission.zip'}  check={'PASS' if ok else 'FAIL'}  systems={len(picks)}")
    print(f"upload it in the {'Final private' if a.final else 'Public development'} tab, then: "
          f"python -m gtlab.autotune uploaded {uid}")
    return 0 if ok else 1


def run_check(zp):
    try:
        from gtlab.check import check
        rep = check(zp, verbose=False)
        return bool(rep.get("pass"))
    except Exception as e:                    # checker needs the Linux sandbox bits; report, don't block
        print(f"checker unavailable here ({type(e).__name__}: {e}); run it before uploading")
        return False


def cmd_register(a):
    reg = load()
    d = (ROOT / a.build) if not Path(a.build).is_absolute() else Path(a.build)
    man = json.loads((d / "manifest.json").read_text())
    systems = {}
    for sid, info in man["systems"].items():
        mid = info.get("model_id", info.get("kind"))
        if info.get("kind") == "l0a":
            systems[sid] = {"kind": "l0a"}
        else:
            m = re.match(r"(\w+)@([\d.]+)", mid or "")
            systems[sid] = {"kind": m.group(1) if m else mid, "lam": float(m.group(2)) if m else 1.0,
                            "data_ticks": a.data_ticks}
    uid = a.id or next_id(reg)
    reg["uploads"].append({"id": uid, "date": a.date or toronto_today(), "phase": a.phase,
                           "dir": str(d.relative_to(ROOT)) if d.is_relative_to(ROOT) else str(d),
                           "systems": systems, "why": {}, "uploaded": True, "scores": {}})
    save(reg)
    print(f"registered {uid} ({len(systems)} systems) as uploaded on {reg['uploads'][-1]['date']}")


def cmd_uploaded(a):
    reg = load()
    u = next(u for u in reg["uploads"] if u["id"] == a.id)
    u["uploaded"] = True
    u["date"] = a.date or toronto_today()
    save(reg)
    print(f"{a.id} marked uploaded on {u['date']}")


def parse_scores(text):
    text = text.strip()
    if text.startswith("{"):
        return {k: float(v) for k, v in json.loads(text).items() if k in S.SYSTEM_IDS}
    out = {}
    for sid in S.SYSTEM_IDS:
        m = re.search(sid.replace("_", r"[_ ]") + r"\D{0,40}?(\d*\.\d+|\d+)", text, flags=re.I)
        if m:
            out[sid] = float(m.group(1))
    return out


def cmd_score(a):
    reg = load()
    u = next(u for u in reg["uploads"] if u["id"] == a.id)
    text = Path(a.file).read_text() if a.file and a.file != "-" else sys.stdin.read()
    sc = parse_scores(text)
    extra = [s for s in sc if s not in u["systems"]]
    for s in extra:
        sc.pop(s)
    u["scores"].update(sc)
    u["uploaded"] = True
    save(reg)
    print(f"{a.id}: recorded {len(sc)} scores " + " ".join(f"{k}={v:.4f}" for k, v in sc.items()))
    if extra:
        print(f"ignored (not in this upload): {extra}")


def cmd_status(a):
    reg = load()
    print(f"{'system':17s} {'slots':5s} {'best':>7s}  best config / live config")
    for sid in S.SYSTEM_IDS:
        h = history(reg, sid)
        b = _best(h)
        print(f"{sid:17s} {slots_used(reg, sid)}/3   {b[1] if b else float('nan'):7.4f}  "
              f"{json.dumps(b[0]) if b else '-'} / {json.dumps(live(reg, sid))}")
    pairs = []
    for u in reg["uploads"]:
        for sid, sc in u.get("scores", {}).items():
            c = u["systems"].get(sid) or {}
            loc = (u.get("local") or {}).get(sid, {}).get(c.get("kind"))
            if loc is not None:
                pairs.append((loc, sc))
    if len(pairs) >= 3:
        a_, b_ = np.array(pairs).T
        print(f"local-vs-public calibration: n={len(pairs)} corr={np.corrcoef(a_, b_)[0, 1]:.2f} "
              f"mean gap={np.mean(b_ - a_):+.3f}  (low corr = local screen is not predictive, trust public)")
    for u in reg["uploads"]:
        mean = np.mean(list(u["scores"].values())) if u["scores"] else float("nan")
        print(f"  {u['id']} {u['date']} {u.get('phase','public'):6s} up={u['uploaded']!s:5s} "
              f"n={len(u['systems'])} scored={len(u['scores'])} mean={mean:.4f} {u['dir']}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="autotune")
    ap.add_argument("--data-root", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose"); p.add_argument("--final", action="store_true")
    p.add_argument("--with-ode", action="store_true"); p.add_argument("--no-check", action="store_true")
    p.add_argument("--no-screen", action="store_true", help="skip the local leave-one-run-out screen")
    p.set_defaults(fn=cmd_propose)
    p = sub.add_parser("register"); p.add_argument("build"); p.add_argument("--id")
    p.add_argument("--phase", default="public"); p.add_argument("--date"); p.add_argument("--data-ticks", type=int, default=0)
    p.set_defaults(fn=cmd_register)
    p = sub.add_parser("uploaded"); p.add_argument("id"); p.add_argument("--date"); p.set_defaults(fn=cmd_uploaded)
    p = sub.add_parser("score"); p.add_argument("id"); p.add_argument("file", nargs="?", default="-")
    p.set_defaults(fn=cmd_score)
    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
