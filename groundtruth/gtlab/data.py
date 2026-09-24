"""Run container shared by collection, fitting and validation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Run:
    system: str
    exp: str                    # experiment id, e.g. "p1.hold_rec"
    y0: np.ndarray              # [p] noisy initial observation (from reset)
    U: np.ndarray               # [T, m] physical actions, CTRL order
    Y: np.ndarray               # [T, p] noisy observations after each action, OBS order
    Ytrue: np.ndarray | None = None   # [T, p] noiseless truth (mocks only)
    tags: dict = field(default_factory=dict)   # e.g. {"split": "train"|"val", "category": ...}

    @property
    def T(self):
        return self.U.shape[0]


def runs_from_ledger(spec, ledger_rows):
    """Group ledger rows (dicts, see gtlab.ledger) into Run objects, in tick order."""
    by_run = {}
    order = []
    for r in ledger_rows:
        if r.get("t") == "reset":
            key = r["run_id"]
            by_run[key] = {"exp": r["exp"], "y0": r["obs"], "steps": {}}
            order.append(key)
        elif r.get("t") == "step" and r["run_id"] in by_run:
            by_run[r["run_id"]]["steps"][int(r["tick"])] = (r["action"], r["obs"])
    runs = []
    for key in order:
        d = by_run[key]
        if not d["steps"]:
            continue
        ticks = sorted(d["steps"])
        U = np.array([[d["steps"][k][0][c] for c in spec.controls] for k in ticks], float)
        Y = np.array([[d["steps"][k][1][o] for o in spec.observables] for k in ticks], float)
        y0 = np.array([d["y0"][o] for o in spec.observables], float)
        split = "val" if d["exp"].startswith("val") else "train"
        runs.append(Run(spec.id, d["exp"], y0, U, Y, tags={"split": split, "run_id": key}))
    return runs
