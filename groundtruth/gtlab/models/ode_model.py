"""Model wrapper for the grey-box ODE families (kind "ode").

Holds the family module name, theta vector (natural units, PARAMS order), active mechanisms and
n_sub. Rollout via gtlab.ode.core.rollout; fit delegates to gtlab.ode.fit (imported lazily).
"""
from __future__ import annotations

import importlib
import time

import numpy as np

from gtlab.models import common as C


def family_module(family):
    return importlib.import_module(f"gtlab.ode.{family}")


def has_family(family):
    try:
        family_module(family)
        return True
    except Exception:
        return False


class ODEModel(C.DevModel):
    kind = "ode"

    def __init__(self, spec, family=None, theta=None, mech=None, n_sub=None, val_runs=None,
                 time_budget_s=None, pairs=None, fit_kw=None, **cfg):
        super().__init__(spec, **cfg)
        self.family = family or spec.id
        self.mod = family_module(self.family)
        self.theta = None if theta is None else np.asarray(theta, float)
        self.mech = None if mech is None else "".join(sorted(mech))
        self.n_sub = int(n_sub or getattr(self.mod, "N_SUB", 4))
        self.val_runs = val_runs
        self.time_budget_s = time_budget_s
        self.pairs = pairs
        self.fit_kw = dict(fit_kw or {})
        if list(self.mod.OBS) != list(spec.observables) or list(self.mod.CTRL) != list(spec.controls):
            raise ValueError(f"ODE family {self.family} OBS/CTRL do not match the spec order")

    def fit(self, runs, val_runs=None):
        from gtlab.ode import fit as F
        t0 = time.time()
        self._prepare(runs)
        val = val_runs if val_runs is not None else self.val_runs
        kw = dict(self.fit_kw)
        if self.pairs:
            kw["pairs"] = tuple(self.pairs)
        kw.setdefault("n_sub", self.n_sub)
        with C.single_thread():
            res = F.fit_all_mechs(self.mod, runs, val or [], self.sigma, time_budget=self.time_budget_s, **kw)
        self.mech = "".join(sorted(res["best"]))
        self.theta = np.asarray(res["theta"], float)
        self.info = {"mech": self.mech, "fit_s": round(time.time() - t0, 1),
                     "mechs": {k: {"val": float(v.get("val", np.nan)),
                                   "cost": float(v.get("info", {}).get("cost", np.nan))}
                               for k, v in res.get("results", {}).items()}}
        return self

    def theta_dict(self):
        from gtlab.ode import core
        vec = self.theta if self.theta is not None else None
        return core.theta_dict(self.mod, vec)

    def raw_rollout(self, y0, U):
        from gtlab.ode import core
        return core.rollout(self.mod, np.asarray(y0, float), U, self.theta_dict(),
                            frozenset(self.mech or "AB"), n_sub=self.n_sub)

    def export(self):
        theta = self.theta if self.theta is not None else [p[1] for p in self.mod.PARAMS]
        return {"kind": "ode", "family": self.family, "theta": [float(v) for v in theta],
                "mech": sorted(self.mech or "AB"), "n_sub": int(self.n_sub),
                "param_names": [p[0] for p in self.mod.PARAMS]}
