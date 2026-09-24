"""Model API shared by every rung of the ladder.

Training side (dev only, may use scipy):
    model = SomeModel(spec, **cfg)
    model.fit(train_runs)                 # list[gtlab.data.Run]
    Yhat = model.rollout(y0, U)           # [T, p] physical units, open loop from y0
    blob = model.export()                 # JSON-serializable dict (numbers/lists only)

Runtime side (ships in the submission, numpy only):
    gtlab/runtime/infer.py implements rollout_from_blob(blob, y0, U, spec_meta) for every
    model kind, re-implementing the forward pass without scipy. Every model kind must have
    a parity test: Model.rollout == infer.rollout_from_blob(model.export()).

blob always carries {"kind": "<l0a|l0b|l1|l2|ode|ensemble>", ...}.
"""
from __future__ import annotations

import numpy as np


class Model:
    kind = "base"

    def __init__(self, spec, **cfg):
        self.spec = spec
        self.cfg = cfg

    def fit(self, runs):
        return self

    def rollout(self, y0, U) -> np.ndarray:
        raise NotImplementedError

    def export(self) -> dict:
        raise NotImplementedError
