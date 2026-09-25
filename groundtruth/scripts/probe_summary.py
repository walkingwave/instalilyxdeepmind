"""Summarize hold runs: per observable, initial, start/end levels, exponential time constant,
tail noise and whether the tail is still drifting. Free: reads the ledger only.

    python scripts/probe_summary.py [--exp p1.hold_rec] [--tail 40]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import systems as S
from gtlab.ledger import Ledger, data_dir, load_runs


def fit_tau(y):
    """y_t = y_inf + (y_s - y_inf) exp(-t/tau), t = 1..T. Returns (tau, y_inf, rmse)."""
    T = len(y)
    t = np.arange(1, T + 1, dtype=float)
    s = np.std(y) + 1e-12

    def res(p):
        y_inf, y_s, ltau = p
        return (y_inf + (y_s - y_inf) * np.exp(-t / np.exp(ltau)) - y) / s

    best = None
    for tau0 in (2.0, 8.0, 30.0, 100.0, 400.0):
        p0 = [y[-10:].mean(), y[0], np.log(tau0)]
        r = least_squares(res, p0, bounds=([-np.inf, -np.inf, np.log(0.3)], [np.inf, np.inf, np.log(5000)]))
        if best is None or r.cost < best.cost:
            best = r
    y_inf, y_s, ltau = best.x
    rmse = float(np.sqrt(np.mean((res(best.x) * s) ** 2)))
    return float(np.exp(ltau)), float(y_inf), rmse


def summarize(spec, run, tail):
    Y, y0 = run.Y, run.y0
    out = []
    for j, o in enumerate(spec.observables):
        y = Y[:, j]
        tl = y[-tail:]
        d = np.diff(tl)
        noise = float(np.median(np.abs(d - np.median(d))) * 1.4826 / np.sqrt(2))
        slope = float(np.polyfit(np.arange(tail), tl, 1)[0])
        tau, y_inf, rmse = fit_tau(y)
        scale = max(abs(y_inf), abs(y0[j]), 1e-9)
        drift_sig = abs(slope) * tail / max(noise * np.sqrt(2.0 / tail) * 3, 1e-12)  # tail move vs 3 SE
        out.append(dict(obs=o, y0=y0[j], y1=y[0], y_end=tl.mean(), y_inf=y_inf,
                        chg_pct=100 * (tl.mean() - y0[j]) / scale, tau=tau, fit_rmse_pct=100 * rmse / scale,
                        noise_pct=100 * noise / scale, drift=drift_sig))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="p1.hold_rec")
    ap.add_argument("--tail", type=int, default=40)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    print(f"{'system':<17}{'obs':<18}{'y0':>10}{'y1':>10}{'y_end':>10}{'chg%':>7}{'tau':>8}{'fit%':>6}{'noise%':>7}{'drift':>6}")
    for sid in S.SYSTEM_IDS:
        spec = S.get(sid)
        d = data_dir(sid, a.mock)
        if not (Path(d) / "ledger.jsonl").exists():
            continue
        led = Ledger(d, sid, a.mock)
        runs = [r for r in load_runs(spec, led) if r.exp == a.exp]
        pool = led.reset_pool()
        for run in runs:
            for r in summarize(spec, run, a.tail):
                print(f"{sid:<17}{r['obs']:<18}{r['y0']:>10.3f}{r['y1']:>10.3f}{r['y_end']:>10.3f}{r['chg_pct']:>7.1f}"
                      f"{r['tau']:>8.1f}{r['fit_rmse_pct']:>6.1f}{r['noise_pct']:>7.2f}{r['drift']:>6.1f}")
        if pool:
            P = np.array([[c["obs"][o] for o in spec.observables] for c in pool])
            spread = ", ".join(f"{o}:{P[:, j].min():.3g}..{P[:, j].max():.3g}" for j, o in enumerate(spec.observables))
            print(f"{'':<17}reset pool n={len(pool)}  {spread}")


if __name__ == "__main__":
    main()
