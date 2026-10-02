"""Purchase 9 (final discovery round): one run per system in regimes we have never held.

Levels are on the brief's recovery -> pulse axis at alpha = 0.85 (the middle of the test's 0.7-1.0
range) unless noted. Every run leaves RESERVE = 20 credits. Ranking evidence: plans/p9_committee.json
(scripts/lab_p9_committee.py) and the coverage table (scripts/lab_p9_coverage.py); reasons in
plans/p9_design.md.

    python scripts/make_p9.py                 # write plans/p9.json, print blocks and credits
    python scripts/make_p9.py --check         # + free dry-run of phase p9 against the real ledgers
                                              #   (in memory: writes nothing under data/)
    python scripts/make_p9.py --eval          # + committee loss of each final design (free, slow)
    python scripts/make_p9.py --budget        # set p9 caps in data/<sys>/budget.json (run before buying)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import systems as S, design as D, budget as B       # noqa: E402
from gtlab.ledger import Ledger, data_dir                        # noqa: E402

ALPHA, RESERVE = 0.85, 20
REC = "__rec__"

# system -> (blocks [(pulsed controls | REC, ticks)], note). Ticks sum to balance - RESERVE.
DESIGN = {
    "reservoir": ([("all", 420), (REC, 120), (["release_rate", "irrigation_allocation"], 220)],
                  "full pulse 420 (fouling and level floor past the 200 ticks we own), refill 120, "
                  "then release+irrigation alone with screens clean (drain without fouling)"),
    "ad_auction": ([("all", 230), (["bid"], 150)],
                   "full pulse 230 (audience depletion; we own 60), then bid alone 150 (never held)"),
    "power_grid": ([(["price_signal", "interconnector"], 200),
                    (["price_signal", "interconnector", "charging_allowance"], 180)],
                   "price+interconnector 200 (never held), then add charging_allowance cut: its "
                   "sustained footprint under a depleted store"),
    "hospital_queue": ([(["overtime"], 110), (REC, 100)],
                       "overtime alone 110 (the drained-queue wait climb, +16.7 sigma miss; owned 30), "
                       "then recovery 100: does wait keep climbing without overtime"),
    "supply_chain": ([(["order_quantity"], 130), (["order_quantity", "lead_time_buy"], 80)],
                     "orders alone 130 (committee split: supplier 0 vs 362), then add lead_time_buy "
                     "cut with orders flowing: its sustained footprint"),
    "epidemic": ([(["school_closure"], 169)],
                 "school closure alone 169 (never held; highest late-window disagreement)"),
    "social_contagion": ([(["seeding", "incentive"], 169)],
                         "seeding+incentive without bridge 169 (never held; largest disagreement)"),
    "wildlife": ([(["hunting_quota", "corridor_access"], 169)],
                 "harvest with corridors open under full protection 169 (never held; largest "
                 "long-horizon disagreement)"),
    "market": ([(["transaction_tax"], 140)],
               "tax alone 140 (owned only 60 ticks; depth under tax is the weak observable)"),
    "traffic": ([("all-toll", 80), (["signal_timing", "lane_closure", "ramp_metering"], 60)],
                "harsh pulse with toll kept at 5 for 80 (toll-elastic demand: u016 says speeds 27/32, "
                "older fits 11/12), then freight_priority and clearance_effort back to recovery "
                "alone for 60: their sustained footprint under congestion"),
}


def level(spec, which):
    ctrls = list(spec.controls)
    r = np.array([spec.recovery[c] for c in ctrls], float)
    p = np.array([spec.pulse[c] for c in ctrls], float)
    if which == REC:
        return r
    if which == "all":
        on = set(ctrls)
    elif which == "all-toll":
        on = set(ctrls) - {"toll"}
    else:
        on = set(which)
    assert on <= set(ctrls), (spec.id, on)
    return np.array([r[i] + ALPHA * (p[i] - r[i]) if c in on else r[i] for i, c in enumerate(ctrls)])


def build():
    plan, summary = {}, {}
    for sid, (blocks, note) in DESIGN.items():
        spec = S.get(sid)
        led = Ledger(data_dir(sid, False), sid, False)
        bal = 2000 - led.steps_charged()
        U = np.concatenate([np.repeat(level(spec, w)[None], n, 0) for w, n in blocks])
        T = len(U)
        assert T <= bal - RESERVE, (sid, T, bal)
        lo, hi = np.array(spec.lo()), np.array(spec.hi())
        assert np.all(U >= lo - 1e-12) and np.all(U <= hi + 1e-12), sid
        plan[sid] = [{"exp_id": "discover", "category": "sustained", "note": "final discovery: " + note,
                      "U": [[round(float(v), 6) for v in row] for row in U]}]
        summary[sid] = {"balance": bal, "T": T, "left": bal - T, "blocks": [
            {"on": w if isinstance(w, str) else list(w), "ticks": n,
             "action": {c: round(float(v), 4) for c, v in zip(spec.controls, level(spec, w))}}
            for w, n in blocks], "note": note}
    return plan, summary


def check(plan):
    """Free dry-run: phase_cost of p9 against every real ledger, in memory."""
    from gtlab.collect import phase_cost, format_cost
    tot = 0
    for sid in plan:
        spec = S.get(sid)
        ex = D.experiments_from_file(spec, str(ROOT / "plans" / "p9.json"), "p9")
        ph = D.phase_plan(spec, "p9", 0, ex)
        led = Ledger(data_dir(sid, False), sid, False)
        rows = phase_cost(led, ph)
        print(format_cost(sid, "p9", rows))
        tot += sum(r["todo"] for r in rows)
    print(f"TOTAL p9 credits: {tot}")


def set_budget(summary):
    for sid, s in summary.items():
        d = data_dir(sid, False)
        plan = B.load(d, sid)
        led = Ledger(d, sid, False)
        plan["phases"]["p9"] = s["T"]
        other = sum(v for k, v in plan["phases"].items() if k != "reserve")
        plan["phases"]["reserve"] = max(led.phase_charged("reserve"), 2000 - other)
        B.validate(plan)
        B.path(d).write_text(json.dumps(plan, indent=1) + "\n")
        print(sid, plan["phases"])


def evaluate(summary):
    sys.path.insert(0, str(ROOT / "scripts"))
    import lab_p9_committee as L
    from gtlab.ledger import load_runs
    for sid in summary:
        spec = S.get(sid)
        sig = np.array(json.loads((ROOT / "plans" / "sigma_calibrated.json").read_text())[sid]["sigma"], float)
        runs = load_runs(spec, Ledger(data_dir(sid, False), sid, False))
        Y0 = np.array([r.y0 for r in runs], float)
        y0 = Y0[int(np.argmin(np.sum(((Y0 - np.median(Y0, 0)) / sig) ** 2, axis=1)))]
        U = np.array(json.loads((ROOT / "plans" / "p9.json").read_text())[sid][0]["U"])
        com = L.committee(sid)
        Ys = [L.roll(m, spec, y0, U) for _, m in com]
        Ys = [Y for Y in Ys if Y is not None]
        print(f"{sid:18s} L {L.closs(Ys, sig):.3f}  last-40% {L.closs(Ys, sig, int(0.6 * len(U))):.3f}  "
              f"ends " + "; ".join(f"{n} {np.round(Y[-1], 1).tolist()}" for (n, _), Y in zip(com, Ys)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--budget", action="store_true")
    a = ap.parse_args()
    plan, summary = build()
    (ROOT / "plans" / "p9.json").write_text(json.dumps(plan))
    (ROOT / "plans" / "p9_summary.json").write_text(json.dumps(summary, indent=1))
    tot = 0
    for sid, s in summary.items():
        tot += s["T"]
        print(f"{sid:18s} bal {s['balance']:4d}  buy {s['T']:4d}  left {s['left']:3d}  " +
              " | ".join(f"{b['on'] if isinstance(b['on'], str) else '+'.join(b['on'])} x{b['ticks']}"
                         for b in s["blocks"]))
    print(f"total {tot}")
    if a.check:
        check(plan)
    if a.eval:
        evaluate(summary)
    if a.budget:
        set_budget(summary)


if __name__ == "__main__":
    main()
