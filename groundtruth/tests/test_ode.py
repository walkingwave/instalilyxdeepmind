"""Grey-box ODE families, batched fitter and mocks.

Fast tests: interface, OBS/CTRL parity with systems.py, 4000-step stress rollouts at
bound extremes / recovery-pulse schedules for every mechanism pair, runtime budget,
batched-rollout parity, mock surface.
Slow tests (identifiability on power_grid + hospital_queue, ~2-4 min each): skipped when
GT_FAST=1.
"""
import ast
import os
import time
from pathlib import Path

import numpy as np
import pytest

from gtlab import metric, systems
from gtlab.data import Run
from gtlab.mocks import Y0_RANGE, get_mock, ode_module
from gtlab.ode import core, fit

FAMS = systems.SYSTEM_IDS
PAIRS = [frozenset(p) for p in ("AB", "AC", "BC")]
ODE_DIR = Path(__file__).resolve().parent.parent / "gtlab" / "ode"
RUNTIME_LIMIT_S = 1.2        # per 4000-step rollout (=> 40 x 4000 well under 60 s)


def _acts(spec):
    rec = np.array([spec.recovery[c] for c in spec.controls], float)
    pul = np.array([spec.pulse[c] for c in spec.controls], float)
    return rec, pul, np.array(spec.lo(), float), np.array(spec.hi(), float)


def eval_like(spec, T, kind, rng):
    """Eval-like schedules: holds, recovery with pulses (alpha ~ U(.7,1) per control),
    random corners / interior levels, and plain extremes."""
    rec, pul, lo, hi = _acts(spec)
    m = len(rec)
    if kind == "hold_lo":
        return np.tile(lo, (T, 1))
    if kind == "hold_hi":
        return np.tile(hi, (T, 1))
    if kind == "hold_pulse":
        return np.tile(pul, (T, 1))
    if kind == "hold_rec":
        return np.tile(rec, (T, 1))
    U = np.tile(rec, (T, 1)).astype(float)
    t = 0
    while t < T:
        if kind == "pulses":
            t += int(np.exp(rng.uniform(np.log(5), np.log(300))))
            L = int(rng.integers(3, 41))
            U[t:t + L] = rec + rng.uniform(0.7, 1.0, m) * (pul - rec)
            t += L
        else:  # "mixed": corners, interior, recovery, pulse
            L = int(np.exp(rng.uniform(np.log(3), np.log(200))))
            r = rng.random()
            if r < 0.3:
                lvl = np.where(rng.random(m) < 0.5, lo, hi)
            elif r < 0.7:
                lvl = rng.uniform(lo, hi)
            elif r < 0.85:
                lvl = rec
            else:
                lvl = rec + rng.uniform(0.7, 1.0, m) * (pul - rec)
            U[t:t + L] = lvl
            t += L
    return np.clip(U, lo, hi)


# ------------------------------------------------------------------ interface
@pytest.mark.parametrize("fam", FAMS)
def test_interface(fam):
    mod = ode_module(fam)
    spec = systems.get(fam)
    assert mod.FAMILY == fam
    assert list(mod.OBS) == list(spec.observables)
    assert list(mod.CTRL) == list(spec.controls)
    assert set(mod.MECHS) == {"A", "B", "C"}
    n = len(mod.STATE)
    assert mod.STATE_LO is None or len(mod.STATE_LO) == n
    assert mod.STATE_HI is None or len(mod.STATE_HI) == n
    names = [p[0] for p in mod.PARAMS]
    assert len(set(names)) == len(names)
    for name, init, lo, hi, lg in mod.PARAMS:
        assert lo <= init <= hi, name
        if lg:
            assert lo > 0, name
    assert isinstance(mod.N_SUB, int) and 1 <= mod.N_SUB <= 4
    th = core.theta_dict(mod)
    y0 = np.array(Y0_RANGE[fam], float).mean(axis=1)
    for mech in PAIRS:
        x = mod.x0(y0, th, mech)
        assert x.shape == (n,)
        dx = mod.f(core._clip(mod, x), _acts(spec)[0], th, mech)
        assert dx.shape == (n,) and np.all(np.isfinite(dx))
        y = mod.h(core._clip(mod, x), _acts(spec)[0], th, mech)
        assert y.shape == (len(mod.OBS),)


@pytest.mark.parametrize("fam", FAMS)
def test_numpy_math_only(fam):
    """Family files ship verbatim: only numpy and math may be imported."""
    tree = ast.parse((ODE_DIR / f"{fam}.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name in ("numpy", "math") for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module in ("numpy", "math", "__future__")


# ------------------------------------------------------------------ stress + runtime
@pytest.mark.parametrize("fam", FAMS)
def test_stress_4000(fam):
    mod = ode_module(fam)
    spec = systems.get(fam)
    unit = {o for o in spec.observables if spec.output_bounds(o)[1] == 1.0}
    rng = np.random.default_rng(123)
    kinds = ["hold_lo", "hold_hi", "hold_pulse", "hold_rec", "pulses", "mixed"]
    for k_i, mech in enumerate(PAIRS):
        mk = get_mock(fam, mech=tuple(sorted(mech)), seed=k_i)
        for kind in kinds:
            U = eval_like(spec, 4000, kind, rng)
            y0 = mk.sample_y0(rng)
            Y = core.rollout(mod, y0, U, mk.th, mech, n_sub=mod.N_SUB)
            assert Y.shape == (4000, len(mod.OBS))
            assert np.all(np.isfinite(Y)), (fam, mech, kind)
            assert np.all(Y >= 0.0), (fam, mech, kind, Y.min(axis=0))
            for j, o in enumerate(mod.OBS):
                if o in unit:
                    assert np.all(Y[:, j] <= 1.0), (fam, o)
            # bounded: no blow-up relative to the plausible initial scale
            scale = np.abs(np.array(Y0_RANGE[fam])).max(axis=1) + 1.0
            assert np.all(Y.max(axis=0) < 1e4 * scale), (fam, mech, kind)


@pytest.mark.parametrize("fam", FAMS)
def test_runtime(fam):
    mod = ode_module(fam)
    spec = systems.get(fam)
    th = core.theta_dict(mod)
    U = eval_like(spec, 4000, "mixed", np.random.default_rng(0))
    y0 = np.array(Y0_RANGE[fam], float).mean(axis=1)
    core.rollout(mod, y0, U[:50], th, frozenset("AB"), n_sub=mod.N_SUB)
    best = np.inf
    for mech in PAIRS:
        t0 = time.perf_counter()
        core.rollout(mod, y0, U, th, mech, n_sub=mod.N_SUB)
        best = min(best, time.perf_counter() - t0)
    print(f"{fam}: n_sub={mod.N_SUB} {best:.3f}s per 4000 steps -> {40 * best:.1f}s for 40 episodes")
    assert best < RUNTIME_LIMIT_S
    assert 40 * best < 60.0


# ------------------------------------------------------------------ batched fitter parity
@pytest.mark.parametrize("fam", FAMS)
def test_batch_matches_core(fam):
    mod = ode_module(fam)
    spec = systems.get(fam)
    rng = np.random.default_rng(1)
    mk = get_mock(fam, mech=("A", "C"), seed=2)
    U1 = eval_like(spec, 120, "mixed", rng)
    U2 = eval_like(spec, 120, "pulses", rng)
    y1, y2 = mk.sample_y0(rng), mk.sample_y0(rng)
    th2 = core.theta_dict(mod)
    TH = {k: np.array([mk.th[k], th2[k]]) for k in th2}
    Yb = fit.rollout_batch(mod, np.stack([y1, y2]), np.stack([U1, U2], axis=1), TH, mk.mech, mod.N_SUB)
    Ya = core.rollout(mod, y1, U1, mk.th, mk.mech, n_sub=mod.N_SUB)
    Yc = core.rollout(mod, y2, U2, th2, mk.mech, n_sub=mod.N_SUB)
    np.testing.assert_allclose(Yb[:, 0], Ya, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(Yb[:, 1], Yc, rtol=1e-9, atol=1e-9)


def test_param_transform_roundtrip():
    for fam in FAMS:
        mod = ode_module(fam)
        v = np.array([p[1] for p in mod.PARAMS])
        np.testing.assert_allclose(fit.from_z(mod, fit.to_z(mod, v)), v, rtol=1e-9, atol=1e-12)


# ------------------------------------------------------------------ mocks
@pytest.mark.parametrize("fam", FAMS)
def test_mock_surface(fam):
    spec = systems.get(fam)
    mk = get_mock(fam, mech=("B", "C"), seed=3)
    y0 = mk.reset(np.random.default_rng(0))
    assert y0.shape == (len(spec.observables),) and np.all(np.isfinite(y0))
    rec = dict(spec.recovery)
    ys, truths = [], []
    for t in range(30):
        ys.append(mk.step(rec if t % 2 else [spec.pulse[c] for c in spec.controls]))
        truths.append(mk.last_truth.copy())
    ys = np.array(ys)
    assert np.all(np.isfinite(ys)) and np.all(ys >= 0)
    U = np.array([[spec.pulse[c] for c in spec.controls] if t % 2 == 0 else
                  [spec.recovery[c] for c in spec.controls] for t in range(30)])
    Yt, Yn = mk.simulate(mk.y0_true, U)
    np.testing.assert_allclose(Yt, np.array(truths), rtol=1e-9, atol=1e-9)
    assert Yn.shape == Yt.shape


# ------------------------------------------------------------------ identifiability (slow)
def _ident(fam, budget):
    spec = systems.get(fam)
    mk = get_mock(fam, mech=("A", "B"), seed=1)
    rng = np.random.default_rng(7)
    runs = []
    for T, kind in [(120, "hold_rec"), (120, "pulses"), (450, "pulses"), (180, "mixed"),
                    (65, "mixed"), (65, "pulses"), (250, "mixed"), (250, "pulses")]:
        mk.reset(rng)
        y0n = mk._noisy(mk.y0_true)
        U = eval_like(spec, T, kind, rng)
        Yt, Yn = mk.simulate(mk.y0_true, U)
        runs.append(Run(fam, kind, y0n, U, Yn, Ytrue=Yt))
    assert sum(r.T for r in runs) == 1500
    sigma = metric.sigma_proxy(runs)
    theta, info = fit.fit_ode(mk.mod, runs, sigma, frozenset("AB"), n_starts=4, max_nfev=30,
                              early_T=150, n_polish=1, time_budget=budget, seed=0)
    th = core.theta_dict(mk.mod, theta)
    s_fit, s_per = [], []
    for kind in ("pulses", "mixed", "pulses", "mixed"):
        y0t = mk.sample_y0(rng)
        y0n = mk._noisy(y0t)
        U = eval_like(spec, 4000, kind, rng)
        Yt, _ = mk.simulate(y0t, U)
        Yh = core.rollout(mk.mod, y0n, U, th, frozenset("AB"), n_sub=mk.mod.N_SUB)
        s_fit.append(metric.score(Yh, Yt, sigma))
        s_per.append(metric.score(np.tile(y0n, (4000, 1)), Yt, sigma))
    return float(np.mean(s_fit)), float(np.mean(s_per)), info


@pytest.mark.skipif(os.environ.get("GT_FAST") == "1", reason="slow identifiability test")
@pytest.mark.parametrize("fam", ["power_grid", "hospital_queue"])
def test_identifiability(fam):
    s_fit, s_per, info = _ident(fam, budget=150)
    print(f"{fam}: fitted {s_fit:.3f} vs persistence {s_per:.3f} (fit {info['time']:.0f}s)")
    assert s_fit > s_per + 0.15
    assert s_fit > 0.8
