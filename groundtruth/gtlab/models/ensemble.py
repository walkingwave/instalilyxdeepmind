"""Per-observable median ensemble and per-observable source selection.

Ensemble:   y_t[j] = median_i clip(member_i(y0, U))_t[j]
PerObs:     y_t[j] = clip(member_{map[j]}(y0, U))_t[j]
Members are fitted DevModels sharing the same spec and clip range.
"""
from __future__ import annotations

import numpy as np

from gtlab.models import common as C


class Ensemble(C.DevModel):
    kind = "ensemble"

    def __init__(self, spec, members, **cfg):
        super().__init__(spec, **cfg)
        self.members = list(members)
        if self.members:
            self.meta = self.members[0].meta
            self.clip = self.members[0].clip

    def fit(self, runs):
        return self                              # members are fitted beforehand

    def raw_rollout(self, y0, U):
        return np.median(np.stack([m.rollout(y0, U) for m in self.members], axis=0), axis=0)

    def export(self):
        return {"kind": "ensemble", "members": [m.export() for m in self.members]}


class PerObs(C.DevModel):
    kind = "perobs"

    def __init__(self, spec, members, obs_map, **cfg):
        super().__init__(spec, **cfg)
        self.members = list(members)
        self.map = [int(i) for i in obs_map]
        assert len(self.map) == spec.p
        self.meta = self.members[0].meta
        self.clip = self.members[0].clip

    def fit(self, runs):
        return self

    def raw_rollout(self, y0, U):
        outs = {}
        Y = np.empty((U.shape[0], self.spec.p))
        for j, i in enumerate(self.map):
            if i not in outs:
                outs[i] = self.members[i].rollout(y0, U)
            Y[:, j] = outs[i][:, j]
        return Y

    def export(self):
        return {"kind": "perobs", "members": [m.export() for m in self.members], "map": self.map}
