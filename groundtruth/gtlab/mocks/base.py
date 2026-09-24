"""Mock simulators = the grey-box ODE family modules run with a secret "true" theta.

MockSystem(system_id, mech=('A','B'), theta=None, noise=0.03, seed=0)
    .reset(rng=None) -> noisy y0      (true y0 drawn in Y0_RANGE; hidden state = x0(true y0))
    .step(u)         -> noisy y       (u: dict {control: value} or array in CTRL order)
    .last_truth      -> noiseless y after the last step (or the true y0 after reset)
    .simulate(y0_true, U) -> (Ytrue, Ynoisy)   batch helper, independent of reset/step state
Noise: y * (1 + noise*N(0,1)) + add*N(0,1), add = 0.5% of the family's typical magnitude
(per-observable overrides, e.g. frequency gets additive-only noise), clipped to output bounds.
"""
from __future__ import annotations

import importlib

import numpy as np

from gtlab.ode import core

# Plausible TRUE initial-observation ranges per family (OBS order). Guesses: the real
# reset distribution is learned from free resets.
Y0_RANGE = {
    "power_grid": [(650.0, 900.0), (49.9, 50.1), (0.15, 0.4)],
    "hospital_queue": [(2.0, 15.0), (10.0, 80.0), (1.0, 4.0)],
    "epidemic": [(50.0, 1500.0), (20.0, 400.0)],
    "market": [(0.8, 1.25), (0.0, 1.0), (5.0, 15.0)],
    "traffic": [(0.0, 5.0), (0.0, 5.0), (40.0, 60.0), (40.0, 60.0)],
    "supply_chain": [(0.0, 5.0), (100.0, 400.0), (50.0, 200.0)],
    "wildlife": [(200.0, 800.0), (20.0, 80.0), (200.0, 800.0), (20.0, 80.0)],
    "reservoir": [(400.0, 800.0), (3.0, 8.0), (1.0, 4.0), (0.6, 0.9)],
    "ad_auction": [(0.2, 0.6), (5.0, 20.0), (0.5, 3.0)],
    "social_contagion": [(100.0, 1000.0), (50.0, 800.0)],
}

# (multiplicative, additive) noise overrides per observable
NOISE_OVERRIDE = {
    "power_grid": {"frequency": (0.0, 0.005), "renewable_share": (0.03, 0.003)},
    "reservoir": {"quality": (0.01, 0.005)},
    "ad_auction": {"win_rate": (0.02, 0.005)},
}

UNIT_BOUNDED = {
    "power_grid": {"renewable_share"},
    "reservoir": {"quality"},
    "ad_auction": {"win_rate"},
}


def ode_module(system_id):
    return importlib.import_module(f"gtlab.ode.{system_id}")


def _to_z(mod, vec):
    z = []
    for (name, init, lo, hi, lg), v in zip(mod.PARAMS, vec):
        if lg:
            z.append((np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)))
        else:
            z.append((v - lo) / (hi - lo) if hi > lo else 0.0)
    return np.array(z)


def _from_z(mod, z):
    out = []
    for (name, init, lo, hi, lg), zz in zip(mod.PARAMS, z):
        zz = float(np.clip(zz, 0.0, 1.0))
        if lg:
            out.append(float(np.exp(np.log(lo) + zz * (np.log(hi) - np.log(lo)))))
        else:
            out.append(float(lo + zz * (hi - lo)))
    return np.array(out)


def true_theta(mod, seed=0, spread=0.25):
    """PARAMS init perturbed deterministically by seed (log params by exp(U(-s,s)),
    linear params by U(-s,s)*|init| or 10% of range when init is 0), clipped to bounds."""
    rng = np.random.default_rng(10_000 + int(seed))
    out = []
    for (name, init, lo, hi, lg) in mod.PARAMS:
        r = rng.uniform(-1.0, 1.0)
        if lg:
            v = init * np.exp(spread * r)
        else:
            scale = abs(init) * spread if init != 0 else 0.1 * (hi - lo)
            if name.startswith("f_nom"):
                scale = 0.0
            v = init + scale * r
        out.append(float(np.clip(v, lo, hi)))
    return np.array(out)


class MockSystem:
    def __init__(self, system_id, mech=("A", "B"), theta=None, noise=0.03, seed=0, n_sub=None):
        self.system_id = system_id
        self.mod = ode_module(system_id)
        self.mech = frozenset(mech)
        assert len(self.mech) == 2 and self.mech <= set("ABC")
        vec = true_theta(self.mod, seed) if theta is None else np.asarray(
            [theta[p[0]] for p in self.mod.PARAMS] if isinstance(theta, dict) else theta, float)
        self.theta_vec = vec
        self.th = core.theta_dict(self.mod, vec)
        self.noise = float(noise)
        self.n_sub = int(n_sub or getattr(self.mod, "N_SUB", 4))
        self.rng = np.random.default_rng(seed)
        self.y0_range = np.array(Y0_RANGE[system_id], float)
        self.scale = np.maximum(np.abs(self.y0_range).mean(axis=1), 1e-6)
        ov = NOISE_OVERRIDE.get(system_id, {})
        self.mult = np.array([ov.get(o, (self.noise, None))[0] for o in self.mod.OBS])
        self.add = np.array([ov.get(o, (None, 0.005 * s))[1] if ov.get(o) else 0.005 * s
                             for o, s in zip(self.mod.OBS, self.scale)])
        unit = UNIT_BOUNDED.get(system_id, set())
        self.out_hi = np.array([1.0 if o in unit else np.inf for o in self.mod.OBS])
        self.x = None
        self.last_truth = None
        self.last_u = None

    # -- helpers --
    def _u(self, u):
        if isinstance(u, dict):
            return np.array([float(u[c]) for c in self.mod.CTRL])
        return np.asarray(u, float).reshape(-1)

    def _noisy(self, y):
        n = y.shape
        e1 = self.rng.standard_normal(n)
        e2 = self.rng.standard_normal(n)
        yn = y * (1.0 + self.mult * e1) + self.add * e2
        return np.clip(yn, 0.0, self.out_hi)

    def sample_y0(self, rng=None):
        rng = self.rng if rng is None else rng
        return rng.uniform(self.y0_range[:, 0], self.y0_range[:, 1])

    def _tick(self, x, u):
        mod, th, mech = self.mod, self.th, self.mech
        dt = 1.0 / self.n_sub
        for _ in range(self.n_sub):
            k1 = mod.f(x, u, th, mech)
            k2 = mod.f(core._clip(mod, x + 0.5 * dt * k1), u, th, mech)
            k3 = mod.f(core._clip(mod, x + 0.5 * dt * k2), u, th, mech)
            k4 = mod.f(core._clip(mod, x + dt * k3), u, th, mech)
            x = core._clip(mod, x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=0.0)
        return x

    # -- gateway-like surface --
    def reset(self, rng=None, y0_true=None):
        y0 = self.sample_y0(rng) if y0_true is None else np.asarray(y0_true, float)
        self.y0_true = y0
        self.x = core._clip(self.mod, np.asarray(self.mod.x0(y0, self.th, self.mech), float))
        self.last_truth = y0.copy()
        return self._noisy(y0)

    def step(self, u):
        if self.x is None:
            raise RuntimeError("reset() first")
        u = self._u(u)
        self.x = self._tick(self.x, u)
        y = np.asarray(self.mod.h(self.x, u, self.th, self.mech), float)
        self.last_truth = y
        self.last_u = u
        return self._noisy(y)

    def simulate(self, y0_true, U):
        U = np.asarray(U, float)
        Ytrue = core.rollout(self.mod, np.asarray(y0_true, float), U, self.th, self.mech, n_sub=self.n_sub)
        return Ytrue, self._noisy(Ytrue)
