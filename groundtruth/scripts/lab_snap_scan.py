"""Round-constant scan: list every fitted parameter of the shipped final2 models and flag the ones
that sit near a "round" number (in the value itself, its reciprocal, or 1 - value for fractions above 0.5).

    python scripts/lab_snap_scan.py [--sub submissions/20260928-1810-final2] [--tol 0.03]
Writes plans/snap_scan.json. Free, no gateway calls.

Round grid per decade: tier 1 = {1, 2, 5}, tier 2 = {1.5, 2.5, 3, 4, 6, 7.5, 8}, tier 3 = {1/3, 2/3, 1/6}
(times 10^k). Decade invariance means percentages and per-sub-step rates are covered. Chance rate for a
log-uniform number is printed so flags can be read against the look-elsewhere baseline.
"""
import argparse, importlib, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

T1 = [1.0, 2.0, 5.0]
T2 = [1.5, 2.5, 3.0, 4.0, 6.0, 7.5, 8.0]
T3 = [10 / 3, 20 / 3, 10 / 6]
MANT = [(m, 1) for m in T1] + [(m, 2) for m in T2] + [(m, 3) for m in T3]


def nearest_round(v):
    """best (round value, rel err, tier) for |v| > 0 over the decade grid."""
    a = abs(v)
    if a < 1e-12:
        return 0.0, 0.0, 0
    e = math.floor(math.log10(a))
    best = None
    for k in (e - 1, e, e + 1):
        for m, t in MANT:
            r = m * 10.0 ** k
            rel = abs(a - r) / r
            if best is None or rel < best[1] - 1e-15 or (abs(rel - best[1]) < 1e-15 and t < best[2]):
                best = (math.copysign(r, v), rel, t)
    return best


def chance(tol, tiers=(1, 2, 3)):
    pts = [m for m, t in MANT if t in tiers]
    return min(1.0, len(pts) * 2 * math.log1p(tol) / math.log(10))


def members(model, path="model"):
    if model.get("kind") == "ode":
        yield path, model
    for i, m in enumerate(model.get("members", []) or []):
        yield from members(m, f"{path}.members[{i}]")


def scan_member(fam, theta, tol):
    mod = importlib.import_module(f"gtlab.ode.{fam}")
    out = []
    for (name, init, lo, hi, lg), v in zip(mod.PARAMS, theta):
        span = hi - lo
        at_b = (v <= lo + 1e-3 * span) or (v >= hi - 1e-3 * span)
        at_init = abs(v - init) <= 1e-6 * max(1.0, abs(init))
        cands = []
        r, rel, t = nearest_round(v)
        cands.append(("v", r, rel, t))
        if abs(v) > 1e-12:
            r2, rel2, t2 = nearest_round(1.0 / v)
            cands.append(("1/v", r2, rel2, t2))
        if 0.5 < v < 1.0:
            r3, rel3, t3 = nearest_round(1.0 - v)
            cands.append(("1-v", r3, rel3, t3))
        ok = [c for c in cands if c[2] <= tol]
        ok.sort(key=lambda c: (c[3], c[2]))
        best = ok[0] if ok else min(cands, key=lambda c: c[2])
        if best[0] == "v":
            snap = best[1]
        elif best[0] == "1/v":
            snap = 1.0 / best[1]
        else:
            snap = 1.0 - best[1]
        out.append({"name": name, "value": float(v), "init": init, "lo": lo, "hi": hi, "log": bool(lg),
                    "at_bound": bool(at_b), "at_init": bool(at_init),
                    "form": best[0], "round": best[1], "rel": best[2], "tier": best[3],
                    "snap_value": float(snap), "flag": bool(ok) and not at_b and not at_init})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="submissions/20260928-1810-final2")
    ap.add_argument("--tol", type=float, default=0.03)
    a = ap.parse_args()
    sub = ROOT / a.sub
    rep = {"tol": a.tol, "chance_any_form_tier123": None, "systems": {}}
    # chance per parameter: up to three forms tested, tiers 1-3
    p1 = chance(a.tol)
    rep["chance_per_form"] = p1
    print(f"chance a log-uniform number lies within {a.tol:.0%} of a tier1-3 point: {p1:.2f} per form; "
          f"tier1 only {chance(a.tol, (1,)):.2f}")
    for sd in sorted(p for p in sub.iterdir() if p.is_dir()):
        m = json.loads((sd / "model.json").read_text())
        sysrep = []
        seen = set()
        for path, mem in members(m["model"]):
            key = (mem["family"], tuple(mem["theta"]))
            if key in seen:
                continue
            seen.add(key)
            rows = scan_member(mem["family"], mem["theta"], a.tol)
            sysrep.append({"path": path, "family": mem["family"], "mech": mem.get("mech"), "params": rows})
            print(f"\n== {sd.name} {path} {mem['family']} mech {mem.get('mech')}")
            for r in rows:
                tag = "FLAG" if r["flag"] else ("bound" if r["at_bound"] else ("init" if r["at_init"] else ""))
                print(f"  {r['name']:<10} {r['value']:>12.5g}  [{r['lo']:g},{r['hi']:g}]  {r['form']:>3} -> {r['round']:<10.5g} "
                      f"rel {r['rel']*100:5.2f}% t{r['tier']} {tag}")
        rep["systems"][sd.name] = sysrep
    (ROOT / "plans" / "snap_scan.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
