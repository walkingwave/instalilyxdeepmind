"""Synthetic linear+saturation system for model tests (independent of gtlab.mocks)."""
from __future__ import annotations

import numpy as np

from gtlab import systems as S
from gtlab.data import Run

SYSTEM = "power_grid"   # borrow its spec: 4 controls, 3 observables (load, frequency, renewable_share)


def _sched(spec, T, rng):
    lo, hi = np.array(spec.lo()), np.array(spec.hi())
    U = np.empty((T, spec.m))
    t = 0
    while t < T:
        d = int(rng.integers(5, 60))
        U[t:t + d] = rng.uniform(lo, hi) if rng.random() < 0.7 else np.array([spec.recovery[c] for c in spec.controls])
        t += d
    return U


def simulate(spec, x0, U):
    lo, hi = np.array(spec.lo()), np.array(spec.hi())
    Un = (U - lo) / (hi - lo)
    x1, x2, x3 = x0
    Y = np.empty((len(U), 3))
    for t, u in enumerate(Un):
        x1 = 0.85 * x1 + 0.15 * (np.tanh(2 * u[0] - 1) + 0.5 * u[1] * u[2])
        x2 = 0.99 * x2 + 0.01 * (u[3] + u[0])
        x3 = 0.95 * x3 + 0.05 * (u[1] - 0.5)
        Y[t] = [100 + 30 * x1 + 20 * x2, 50 + 0.1 * (x1 - x2) + 0.05 * x3, 1 / (1 + np.exp(-(x2 - x1 + x3)))]
    return Y


def x0_from(rng):
    return rng.uniform(-0.5, 0.5, size=3)


def obs0(x0):
    x1, x2, x3 = x0
    return np.array([100 + 30 * x1 + 20 * x2, 50 + 0.1 * (x1 - x2) + 0.05 * x3, 1 / (1 + np.exp(-(x2 - x1 + x3)))])


def make_runs(n=8, T=200, seed=0, noise=0.01, split="train"):
    spec = S.get(SYSTEM)
    rng = np.random.default_rng(seed)
    runs = []
    scale = np.array([5.0, 0.01, 0.02])
    for i in range(n):
        x0 = x0_from(rng)
        U = _sched(spec, T, rng)
        Yt = simulate(spec, x0, U)
        Y = Yt + noise * scale * rng.standard_normal(Yt.shape) * 10
        y0 = obs0(x0) + noise * scale * rng.standard_normal(3) * 10
        Y[:, 2] = np.clip(Y[:, 2], 0, 1)
        runs.append(Run(spec.id, f"{split}{i}", y0, U, Y, Ytrue=Yt, tags={"split": split}))
    return spec, runs
