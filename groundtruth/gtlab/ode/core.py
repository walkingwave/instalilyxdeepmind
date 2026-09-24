"""ODE core: family interface + fixed-step RK4 rollout.

NUMPY ONLY. This file is copied verbatim into every submission folder (as gt_ode_core.py),
so it must not import anything outside numpy / stdlib.

A family module (gtlab/ode/<family>.py, copied as gt_ode_<family>.py) defines:

    FAMILY   : str                       system id, e.g. "power_grid"
    OBS      : list[str]                 observable names, in brief order
    CTRL     : list[str]                 control names, in brief order
    STATE    : list[str]                 hidden state names (documentation + clipping)
    MECHS    : dict[str, str]            {"A": "...", "B": "...", "C": "..."} exactly two active
    PARAMS   : list[tuple]               (name, init, lo, hi, log_scale: bool)
    STATE_LO : np.ndarray | None         per-state lower clip (default 0)
    STATE_HI : np.ndarray | None         per-state upper clip (default +inf)

    def x0(y0: np.ndarray, th: dict, mech: frozenset) -> np.ndarray
        deterministic hidden initial state from the (noisy) initial observation
    def f(x: np.ndarray, u: np.ndarray, th: dict, mech: frozenset) -> np.ndarray
        dx/dt, one tick == 1 time unit; u in PHYSICAL units, ordered as CTRL
    def h(x: np.ndarray, u: np.ndarray, th: dict, mech: frozenset) -> np.ndarray
        observation after the tick, ordered as OBS

Observation convention: y_t = h(x_t, u_t) where x_t is the state AFTER integrating tick t
with action u_t. (Matches the gateway: step(u) returns the observation after the action.)
"""
from __future__ import annotations

import numpy as np


def theta_dict(mod, vec=None):
    """Parameter vector (natural units) -> dict. vec=None -> module defaults."""
    names = [p[0] for p in mod.PARAMS]
    if vec is None:
        vec = [p[1] for p in mod.PARAMS]
    return dict(zip(names, (float(v) for v in vec)))


def _clip(mod, x):
    lo = getattr(mod, "STATE_LO", None)
    hi = getattr(mod, "STATE_HI", None)
    x = np.maximum(x, 0.0 if lo is None else lo)
    if hi is not None:
        x = np.minimum(x, hi)
    return x


def rollout(mod, y0, U, th, mech, n_sub=4, return_states=False):
    """Integrate from x0(y0) through actions U[T, m] (physical). Returns Y[T, p]."""
    y0 = np.asarray(y0, dtype=float)
    U = np.asarray(U, dtype=float)
    mech = frozenset(mech)
    x = _clip(mod, np.asarray(mod.x0(y0, th, mech), dtype=float))
    T = U.shape[0]
    Y = np.empty((T, len(mod.OBS)))
    X = np.empty((T, x.size)) if return_states else None
    dt = 1.0 / n_sub
    f = mod.f
    for t in range(T):
        u = U[t]
        for _ in range(n_sub):
            k1 = f(x, u, th, mech)
            k2 = f(_clip(mod, x + 0.5 * dt * k1), u, th, mech)
            k3 = f(_clip(mod, x + 0.5 * dt * k2), u, th, mech)
            k4 = f(_clip(mod, x + dt * k3), u, th, mech)
            x = _clip(mod, x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=0.0)
        Y[t] = mod.h(x, u, th, mech)
        if return_states:
            X[t] = x
    return (Y, X) if return_states else Y
