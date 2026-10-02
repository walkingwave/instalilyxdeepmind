"""Design check for the 140-tick market discovery run (no credits): simulate a schedule with the canon3 / canon4 full
fits and the y3 AB 9-run refit, report per-block separation in sigma units.

    python scripts/lab_canon_market_p9design.py [--write]
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab.ode import core

SIG = np.asarray(json.loads(Path("plans/sigma_calibrated.json").read_text())["market"]["sigma"], float)
Y0S = {"lowP": [93.4, 91.6, 117.3], "midP": [102.4, 96.9, 82.6], "highP": [105.6, 116.5, 118.3]}

# (rate, tax, ticks, label)
BLOCKS = [
    (0.06, 0.041, 40, "B1 reset->(0.06,0.041): hysteresis frozen vs static trades; F builds at a 2nd rate"),
    (0.06, 0.0, 20, "B2 release tax at rate 0.06: F released slowly (canon3) vs fast"),
    (0.06, 0.041, 15, "B3 same tax from an open book: hysteresis trades, static-low threshold stays frozen"),
    (0.1, 0.05, 8, "B4 close the book (tax max)"),
    (0.1, 0.040, 12, "B5 x_lo step 0.040 at rate 0.1 (reopening = fast price fall)"),
    (0.1, 0.038, 12, "B6 x_lo step 0.038"),
    (0.1, 0.036, 12, "B7 x_lo step 0.036"),
    (0.0, 0.0, 21, "B8 recovery"),
]


def load(tag):
    if tag == "y3":
        mod = importlib.import_module("gtlab.ode.market_y3")
        th = json.loads(Path("plans/market_canon_y3ref_full.json").read_text())["theta"]
    else:
        mod = importlib.import_module(f"gtlab.ode.market_canon{tag[1]}")
        th = json.loads(Path(f"plans/market_canon_{tag}_loo.json").read_text())["full"]["theta"]
    return mod, core.theta_dict(mod, th)


def sched(blocks):
    U = np.concatenate([np.tile([r, x], (n, 1)) for r, x, n, _ in blocks])
    edges = np.cumsum([0] + [n for *_, n, _ in blocks])
    return U, edges


def run(mod, th, y0, U, over=None):
    th = dict(th, **(over or {}))
    return core.rollout(mod, np.asarray(y0, float), U, th, frozenset("AB"), n_sub=2, return_states=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    U, E = sched(BLOCKS)
    assert len(U) == 140
    models = {t: load(t) for t in ["c3", "c4", "y3"]}
    variants = {"xlo.0355": ("c4", {"x_lo": 0.0355}), "xlo.037": ("c4", {"x_lo": 0.037}),
                "xlo.039": ("c4", {"x_lo": 0.039}), "xlo.0402": ("c4", {"x_lo": 0.0402}),
                "static.0395": ("c4", {"x_lo": 0.0395, "x_c": 0.0395})}
    for yn, y0 in Y0S.items():
        P = {t: run(m, th, y0, U) for t, (m, th) in models.items()}
        P.update({k: run(*models[t], y0, U, o) for k, (t, o) in variants.items()})
        print(f"== y0 {yn} {y0}")
        for b, (r, x, n, lab) in enumerate(BLOCKS):
            s, e = E[b], E[b + 1]
            pr = lambda k: " ".join(f"{P[k][0][e - 1, j]:.1f}" for j in range(3)) + (f" g{P[k][1][e - 1, 14]:.2f}" if P[k][1].shape[1] > 14 else "")
            def sep(i, j):
                d = np.abs(P[i][0][s:e] - P[j][0][s:e]) / SIG
                return f"P{d[:, 0].mean():.1f} D{d[:, 2].mean():.1f}"
            print(f" {lab}\n    end c3[{pr('c3')}] c4[{pr('c4')}] y3[{pr('y3')}]"
                  f"\n    sep c3-y3 {sep('c3', 'y3')} | c4-y3 {sep('c4', 'y3')} | c3-c4 {sep('c3', 'c4')}"
                  f"\n    x_lo .0355/.037 {sep('xlo.0355', 'xlo.037')} | .037/.039 {sep('xlo.037', 'xlo.039')}"
                  f" | .039/.0402 {sep('xlo.039', 'xlo.0402')} | hyst c4 vs static .0395 {sep('c4', 'static.0395')}")
    if a.write:
        note = ("canon hysteresis / x_lo / funding tie-up test (plans/market_canon_notes.md): " +
                "; ".join(f"t{E[b]}-{E[b + 1]} ({r}, {x}) {lab.split(' ', 1)[1]}" for b, (r, x, n, lab) in enumerate(BLOCKS)))
        out = {"market": [{"exp_id": "canon_hyst", "category": "sustained", "note": note, "U": U.tolist()}]}
        Path("plans/p9_market_alt.json").write_text(json.dumps(out))
        print("wrote plans/p9_market_alt.json")


if __name__ == "__main__":
    main()
