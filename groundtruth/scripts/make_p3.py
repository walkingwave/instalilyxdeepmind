"""Saturday purchase (phase p3): schedules aimed at the two weak bands (MATH_LOG Fri §11).

Per system, from separate resets:
  p3.compose   single-control blocks at alpha=0.85 (each control alone, dwell D, back to recovery
               D/2), then the joint alpha=0.85 pulse for D, then recovery for D   [composition]
  p3.train     recovery baseline with 6 pulses (alpha ~ U(0.7,1) per control), lengths 10-30,
               gaps 10-150, lead 20                                               [recovery]
  p3.hold_mid  one long hold at the alpha=0.5 interior level                      [sustained]

    python scripts/make_p3.py [--reserve 300] [--out plans/p3.json] [--apply-budget]

--apply-budget rewrites data/<sys>/budget.json with a p3 cap = server balance - reserve.
Then:  python -m gtlab.cli plan --all --phase p3 --experiments plans/p3.json --real-dir
       python -m gtlab.cli collect --all --phase p3 --dry-run
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import budget as B, design as D, systems as S
from gtlab.ledger import Ledger, data_dir

SLOW = {"reservoir", "supply_chain", "social_contagion", "wildlife", "epidemic"}
LONG_HOLD = {"reservoir": 450, "supply_chain": 450, "social_contagion": 450, "wildlife": 450, "epidemic": 450}
COMPOSE_D = {2: 60, 3: 45, 4: 40, 6: 30}          # dwell by number of controls


def compose(spec, alpha=0.85):
    r = D.rec(spec)
    p = D.pulse_level(spec, alpha)
    Dw = COMPOSE_D[spec.m]
    rows = []
    for j in range(spec.m):
        u = r.copy()
        u[j] = p[j]
        rows += [D.hold(spec, u, Dw), D.hold(spec, r, Dw // 2)]
    rows += [D.hold(spec, p, Dw), D.hold(spec, r, Dw)]
    return D.clip(spec, np.concatenate(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reserve", type=int, default=300)
    ap.add_argument("--out", default="plans/p3.json")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--apply-budget", action="store_true")
    a = ap.parse_args()
    plan, total = {}, 0
    print(f"{'system':<17} {'compose':>7} {'train':>6} {'hold':>5} {'sum':>6} {'balance':>8} {'left':>6}")
    for sid in S.SYSTEM_IDS:
        spec = S.get(sid)
        rng = np.random.default_rng([a.seed, sum(map(ord, sid))])
        led = Ledger(data_dir(sid, False), sid, False)
        bal = (led.last_budget() or {}).get("remaining")
        Uc = compose(spec)
        Ut = D.pulse_train(spec, rng, n=6, L=(10, 30), gaps=(10, 150), lead=20, T=350)
        Th = LONG_HOLD.get(sid, 400)
        Uh = D.hold(spec, D.pulse_level(spec, 0.5), Th)
        ex = [{"exp_id": "compose", "U": Uc.tolist(), "category": "composition",
               "note": "single controls at alpha 0.85, then joint; additivity of the equilibrium map"},
              {"exp_id": "train", "U": Ut.tolist(), "category": "recovery",
               "note": "6 pulses, gaps 10-150: history dependence"},
              {"exp_id": "hold_mid", "U": Uh.tolist(), "category": "sustained",
               "note": f"alpha 0.5 interior hold, {Th} ticks: sustained level and saturation"}]
        n = sum(len(e["U"]) for e in ex)
        total += n
        plan[sid] = ex
        left = None if bal is None else bal - n
        print(f"{sid:<17} {len(Uc):>7} {len(Ut):>6} {len(Uh):>5} {n:>6} {bal!s:>8} {left!s:>6}")
        if left is not None and left < a.reserve:
            print(f"   WARNING {sid}: leaves {left} < reserve {a.reserve}")
        if a.apply_budget and bal is not None:
            d = data_dir(sid, False)
            bp = B.load(d, sid)
            spent = led.steps_charged()
            cap = int(bal - a.reserve)
            bp["phases"] = {"p1": bp["phases"].get("p1", 0), "p2": bp["phases"].get("p2", 0), "p3": cap,
                            "val": 0, "reserve": a.reserve}
            # keep the sum within total: shrink p1/p2 caps to what was actually charged
            bp["phases"]["p1"] = led.phase_charged("p1")
            bp["phases"]["p2"] = led.phase_charged("p2")
            assert sum(bp["phases"].values()) <= bp["total"], (sid, bp["phases"], spent)
            B.write(d, bp, overwrite=True)
    Path(a.out).write_text(json.dumps(plan))
    print(f"total {total} ticks -> {a.out}" + ("  (budget.json p3 caps written)" if a.apply_budget else ""))


if __name__ == "__main__":
    main()
