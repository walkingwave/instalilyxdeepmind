"""Infrastructure tests: ledger/resume, spend guard, drift, gating, design, mock collection.

Everything runs against MockGateway. RealGateway can never be constructed here.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from gtlab import budget as budgetmod
from gtlab import design, systems
from gtlab.collect import collect_phase, phase_cost
from gtlab.gateway import (CapError, DriftError, GatewayError, MockGateway, RealGateway,
                           SimulatedCrash, SpendError, SpendGuard)
from gtlab.ledger import Ledger, LedgerError, LockedError, data_dir, load_runs, req_id


# --------------------------------------------------------------------------- mock factory
class StandInMock:
    """Tiny stand-in with the gtlab.mocks.MockSystem API, used only if the real one is missing."""

    def __init__(self, system_id, mech=("A", "B"), theta=None, noise=0.03, seed=0):
        self.spec = systems.get(system_id)
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        p, m = self.spec.p, self.spec.m
        g = np.random.default_rng(sum(map(ord, system_id)))
        self.W = g.normal(size=(p, m))
        self.lo, self.hi = np.array(self.spec.lo()), np.array(self.spec.hi())
        self.x = None
        self.last_truth = None

    def reset(self, rng):
        self.x = rng.uniform(1.0, 3.0, size=self.spec.p)
        self.last_truth = self.x.copy()
        return self.x * (1 + self.noise * rng.normal(size=self.x.size))

    def step(self, u):
        un = (np.asarray(u, float) - self.lo) / np.maximum(self.hi - self.lo, 1e-12)
        target = 2.0 + np.tanh(self.W @ un)
        self.x = self.x + 0.1 * (target - self.x)
        self.last_truth = self.x.copy()
        return self.x * (1 + self.noise * self.rng.normal(size=self.x.size))


def _real_mock_available(system_id):
    try:
        from gtlab.mocks import MockSystem
        MockSystem(system_id)
        return True
    except Exception:
        return False


REAL_MOCKS = {s: _real_mock_available(s) for s in systems.SYSTEM_IDS}


def _factory(system_id, mech, theta, noise, world_seed, run_seed):
    """Real gtlab.mocks where that family exists, the stand-in otherwise."""
    if REAL_MOCKS[system_id]:
        import gtlab.gateway as g
        return g.__dict__["_real_default_factory"](system_id, mech, theta, noise, world_seed, run_seed)
    return StandInMock(system_id, mech=mech, theta=theta, noise=noise, seed=run_seed)


def make_gw(system_id="power_grid", **kw):
    return MockGateway(system_id, factory=_factory, **kw)


@pytest.fixture(autouse=True)
def _patch_default_factory(monkeypatch):
    import gtlab.gateway as g
    monkeypatch.setitem(g.__dict__, "_real_default_factory", g._default_factory)
    monkeypatch.setattr(g, "_default_factory", _factory)


def small_phase(spec, phase="p1", T=(20, 25, 15), seed=0):
    rng = np.random.default_rng(seed)
    ex = [
        design._exp(spec, f"{phase}.hold", design.hold(spec, design.rec(spec), T[0]), "sustained"),
        design._exp(spec, f"{phase}.train", design.pulse_train(spec, rng, n=3, lead=2, tail=2, T=T[1]), "recovery"),
        design._exp(spec, f"{phase}.order", design.eval_like(spec, "order", T[2], rng), "order",
                    shop={"n": 3, "mode": "match", "match_exp": f"{phase}.hold"}),
    ]
    return design.phase_plan(spec, phase, seed, ex)


def mk_ledger(tmp_path, sys_id="power_grid", mock=True):
    return Ledger(data_dir(sys_id, mock, tmp_path), sys_id, mock=mock)


def bplan(sys_id="power_grid"):
    return budgetmod.default_plan(sys_id)


# --------------------------------------------------------------------------- gating
def test_realgateway_cannot_be_constructed_in_tests(monkeypatch):
    monkeypatch.setenv("GT_ALLOW_SPEND", "1")
    monkeypatch.setenv("GT_ALLOW_REAL", "1")
    monkeypatch.setenv("GROUNDTRUTH_KEY", "not-a-real-key")
    monkeypatch.setenv("GROUNDTRUTH_GATEWAY_URL", "http://127.0.0.1:9")
    for spend in (True, False):
        with pytest.raises(SpendError):
            RealGateway(spend=spend)


def test_cli_spend_requires_flags(tmp_path, monkeypatch):
    from gtlab.cli import main
    base = ["--data-root", str(tmp_path), "collect", "power_grid", "--phase", "p1"]
    monkeypatch.delenv("GT_ALLOW_SPEND", raising=False)
    assert main(base + ["--spend", "--yes"]) == 2                       # no --max-steps
    assert main(base + ["--spend", "--max-steps", "1000", "--yes"]) == 2  # no GT_ALLOW_SPEND
    monkeypatch.setenv("GT_ALLOW_SPEND", "1")
    monkeypatch.setenv("GROUNDTRUTH_KEY", "x")
    monkeypatch.setenv("GROUNDTRUTH_GATEWAY_URL", "http://127.0.0.1:9")
    # the RealGateway constructor refuses under pytest -> reported as a failed system
    assert main(base + ["--spend", "--max-steps", "1000", "--yes"]) == 1
    led = Ledger(data_dir("power_grid", False, tmp_path), "power_grid", mock=False)
    assert led.steps_charged() == 0 and not led.pending_intents()


def test_cli_real_free_endpoints_gated(tmp_path, monkeypatch):
    from gtlab.cli import main
    monkeypatch.delenv("GT_ALLOW_REAL", raising=False)
    assert main(["--data-root", str(tmp_path), "docs", "power_grid", "--real"]) == 2
    assert main(["--data-root", str(tmp_path), "resets", "power_grid", "--real", "--n", "2"]) == 2


def test_dry_run_makes_zero_calls(tmp_path, monkeypatch):
    spec = systems.get("power_grid")
    led = mk_ledger(tmp_path)
    gw = make_gw()
    ph = small_phase(spec)
    r = collect_phase(spec, gw, led, ph, bplan(), dry_run=True, log=lambda *_: None)
    assert r["needed"] == 60 and r["bought"] == 0
    assert sum(gw.calls.values()) == 0
    # the CLI dry run with --spend never constructs any gateway
    import gtlab.cli as cli

    def boom(*a, **k):
        raise AssertionError("gateway constructed during dry run")
    monkeypatch.setattr(cli, "_real_gateway", boom)
    monkeypatch.setattr(cli, "_mock_gateway", boom)
    monkeypatch.setenv("GT_ALLOW_SPEND", "1")
    assert cli.main(["--data-root", str(tmp_path), "collect", "--all", "--phase", "p1", "--dry-run",
                     "--spend", "--max-steps", "5"]) == 0


# --------------------------------------------------------------------------- ledger / resume
def test_req_ids_deterministic():
    a = req_id("power_grid", "p1.hold", 0, 5)
    assert a == req_id("power_grid", "p1.hold", 0, 5)
    assert a != req_id("power_grid", "p1.hold", 0, 6) != req_id("market", "p1.hold", 0, 5)


def test_mock_and_real_dirs_never_mix(tmp_path):
    with pytest.raises(LedgerError):
        Ledger(data_dir("power_grid", False, tmp_path), "power_grid", mock=True)
    with pytest.raises(LedgerError):
        Ledger(data_dir("power_grid", True, tmp_path), "power_grid", mock=False)
    d = tmp_path / "elsewhere" / "power_grid"
    Ledger(d, "power_grid", mock=True)
    with pytest.raises(LedgerError):
        Ledger(d, "power_grid", mock=False)          # meta row pins the kind


def test_lockfile_blocks_second_writer(tmp_path):
    a, b = mk_ledger(tmp_path), mk_ledger(tmp_path)
    with a:
        with pytest.raises(LockedError):
            b.lock()
    b.lock(); b.unlock()


def _check_final(spec, tmp_path, gw, ph):
    led = mk_ledger(tmp_path)
    planned = ph["steps"]
    assert led.steps_charged() == planned
    assert gw.charged == planned and gw.remaining == 2000 - planned
    steps = [r for r in led.rows if r["t"] == "step"]
    keys = [(r["exp"], r["tick"]) for r in steps]
    assert len(keys) == len(set(keys)) == planned, "duplicate or missing ticks"
    assert len({r["req"] for r in steps}) == planned
    for e in ph["experiments"]:
        st = led.run_state(e["exp_id"])
        assert st["done"] == list(range(e["T"]))
        U = np.array([[r["action"][c] for c in spec.controls]
                      for r in sorted((r for r in steps if r["exp"] == e["exp_id"]), key=lambda r: r["tick"])])
        np.testing.assert_array_equal(U, np.asarray(e["U"]))
        runs_ids = {r["run_id"] for r in steps if r["exp"] == e["exp_id"]}
        assert runs_ids == {st["reset"]["run_id"]}
    assert not led.pending_intents()
    return led


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_kill_at_random_ticks_then_resume(tmp_path, seed):
    spec = systems.get("power_grid")
    ph = small_phase(spec, seed=seed)
    gw = make_gw(seed=seed)             # the "server" survives the client's crashes
    rng = np.random.default_rng(seed)
    kills = sorted(set(rng.integers(1, 70, size=6).tolist()))
    gw.crash_before_send = set(kills[::2])
    gw.crash_after_charge = set(kills[1::2])
    crashes = 0
    for _ in range(50):
        led = mk_ledger(tmp_path)           # fresh process: reload from disk
        try:
            collect_phase(spec, gw, led, ph, bplan(), max_steps=ph["steps"], every=7, log=lambda *_: None)
            break
        except SimulatedCrash:
            crashes += 1
    assert crashes >= 1
    _check_final(spec, tmp_path, gw, ph)


def test_torn_ledger_line_resumes_without_rebuy(tmp_path):
    spec = systems.get("power_grid")
    ph = small_phase(spec)
    gw = make_gw()
    gw.crash_before_send = {31}
    with pytest.raises(SimulatedCrash):
        collect_phase(spec, gw, mk_ledger(tmp_path), ph, bplan(), max_steps=60, log=lambda *_: None)
    # the last line is the (never sent) intent; drop it, then tear the step row before it in
    # half, as if the process died while writing that step row
    p = data_dir("power_grid", True, tmp_path) / "ledger.jsonl"
    lines = p.read_bytes().splitlines(keepends=True)
    assert b'"t":"intent"' in lines[-1] and b'"t":"step"' in lines[-2]
    lines = lines[:-1]
    lines[-1] = lines[-1][: len(lines[-1]) // 2]
    p.write_bytes(b"".join(lines))
    assert mk_ledger(tmp_path).steps_charged() == 29
    collect_phase(spec, gw, mk_ledger(tmp_path), ph, bplan(), max_steps=60, log=lambda *_: None)
    _check_final(spec, tmp_path, gw, ph)


def test_completed_phase_is_not_rebought(tmp_path):
    spec = systems.get("power_grid")
    ph = small_phase(spec)
    gw = make_gw()
    collect_phase(spec, gw, mk_ledger(tmp_path), ph, bplan(), max_steps=60, log=lambda *_: None)
    n = gw.calls["step"]
    r = collect_phase(spec, gw, mk_ledger(tmp_path), ph, bplan(), max_steps=60, log=lambda *_: None)
    assert r["bought"] == 0 and gw.calls["step"] == n


# --------------------------------------------------------------------------- caps
def test_phase_cap_raises_before_sending(tmp_path):
    spec = systems.get("power_grid")
    ph = small_phase(spec)
    gw = make_gw()
    bp = bplan()
    bp["phases"]["p1"] = 59
    with pytest.raises(CapError):
        collect_phase(spec, gw, mk_ledger(tmp_path), ph, bp, max_steps=100, log=lambda *_: None)
    with pytest.raises(CapError):
        collect_phase(spec, gw, mk_ledger(tmp_path), ph, bplan(), max_steps=59, log=lambda *_: None)
    with pytest.raises(CapError):
        collect_phase(spec, gw, mk_ledger(tmp_path), ph, bplan(), max_steps=None, log=lambda *_: None)
    assert gw.calls["step"] == 0 and gw.calls["reset"] == 0 and gw.remaining == 2000


def test_server_budget_too_small_raises(tmp_path):
    spec = systems.get("power_grid")
    ph = small_phase(spec)
    gw = make_gw(budget=50)
    bp = bplan(); bp["total"] = 50; bp["phases"] = {"p1": 50}
    with pytest.raises(CapError):
        collect_phase(spec, gw, mk_ledger(tmp_path), ph, bp, max_steps=100, log=lambda *_: None)
    assert gw.calls["step"] == 0


def test_guard_per_step_cap_and_bounds():
    spec = systems.get("power_grid")
    gw = make_gw()
    run = gw.reset("power_grid", "r")["run_id"]
    g = SpendGuard(gw, spec, base_charged=0, phase_cap_left=3, max_steps=100)
    act = dict(spec.recovery)
    for k in range(3):
        g.step(run, act, f"k{k}")
    with pytest.raises(CapError):
        g.step(run, act, "k3")
    assert gw.calls["step"] == 3
    g2 = SpendGuard(gw, spec, base_charged=0, phase_cap_left=100, max_steps=100)
    bad = dict(act); bad[spec.controls[0]] = spec.bounds[spec.controls[0]][1] * 2 + 1
    with pytest.raises(SpendError):
        g2.step(run, bad, "b0")
    with pytest.raises(SpendError):
        g2.step(run, {spec.controls[0]: 0.0}, "b1")
    with pytest.raises(SpendError):
        g2.step(run, act, None)
    assert gw.calls["step"] == 3
    # total budget: base_charged 1999 of 2000 -> exactly one more allowed
    g3 = SpendGuard(gw, spec, base_charged=1999, phase_cap_left=100, max_steps=100)
    g3.step(run, act, "t0")
    with pytest.raises(CapError):
        g3.step(run, act, "t1")


def test_mock_idempotency_and_free_refusal():
    spec = systems.get("power_grid")
    gw = make_gw()
    r1 = gw.reset("power_grid", "same")
    r2 = gw.reset("power_grid", "same")
    assert r1 == r2 and gw.n_resets == 1
    run = r1["run_id"]
    a = gw.step(run, dict(spec.recovery), "s0")
    b = gw.step(run, dict(spec.recovery), "s0")
    assert a == b and gw.remaining == 1999
    bad = dict(spec.recovery); bad[spec.controls[0]] = 1e9
    with pytest.raises(GatewayError):
        gw.step(run, bad, "s1")
    with pytest.raises(GatewayError):
        gw.step(run, {"nope": 1.0}, "s2")
    assert gw.remaining == 1999
    bud = gw.budget("power_grid")
    assert bud["remaining"] == bud["simulator_steps_remaining"] == 1999
    assert set(r1) >= {"run_id", "observation"} and set(r1["observation"]) == set(spec.observables)
    assert "interventions" in gw.brief("power_grid")


# --------------------------------------------------------------------------- drift
def test_preflight_drift_detected_and_reconcile(tmp_path):
    spec = systems.get("power_grid")
    ph = small_phase(spec)
    gw = make_gw()
    gw.remaining -= 5                                  # steps spent outside this tool
    with pytest.raises(DriftError):
        collect_phase(spec, gw, mk_ledger(tmp_path), ph, bplan(), max_steps=60, log=lambda *_: None)
    assert gw.calls["step"] == 0
    collect_phase(spec, gw, mk_ledger(tmp_path), ph, bplan(), max_steps=60, reconcile=True,
                  log=lambda *_: None)
    led = mk_ledger(tmp_path)
    assert led.steps_charged() == 65
    r = budgetmod.reconcile(gw.budget("power_grid"), led)
    assert r["ok"] and r["drift"] == 0


def test_periodic_drift_aborts(tmp_path):
    spec = systems.get("power_grid")
    ph = small_phase(spec)
    gw = make_gw()
    orig = gw.step

    def leaky(run_id, intervention, request_id=None):
        resp = orig(run_id, intervention, request_id)
        if gw.charged == 12:
            gw.remaining -= 1                          # someone else spends one step
        return resp
    gw.step = leaky
    with pytest.raises(DriftError):
        collect_phase(spec, gw, mk_ledger(tmp_path), ph, bplan(), max_steps=60, every=10,
                      log=lambda *_: None)
    assert gw.charged <= 20


# --------------------------------------------------------------------------- design
@pytest.mark.parametrize("sys_id", systems.SYSTEM_IDS)
def test_design_generators_respect_bounds(sys_id):
    spec = systems.get(sys_id)
    lo, hi = np.array(spec.lo()), np.array(spec.hi())
    rng = np.random.default_rng(1)
    outs = [design.hold(spec, design.level(spec, k, rng), 7) for k in ("recovery", "pulse", "mid", "uniform", "corner")]
    outs += [design.pulse_train(spec, rng, T=300, lead=10, tail=20),
             *design.order_pair(spec, rng), design.single_vs_joint(spec, rng),
             design.multilevel(spec, rng, 250)]
    for cat in design.CATEGORIES + ("mixed",):
        for T in (40, 125, 4000):
            U = design.eval_like(spec, cat, T, rng)
            assert U.shape == (T, spec.m), (cat, T)
            outs.append(U)
    for U in outs:
        assert U.ndim == 2 and U.shape[1] == spec.m
        assert np.all(np.isfinite(U)) and np.all(U >= lo - 1e-12) and np.all(U <= hi + 1e-12)


@pytest.mark.parametrize("sys_id", systems.SYSTEM_IDS)
def test_pulse_train_uses_alpha_in_range(sys_id):
    spec = systems.get(sys_id)
    r, p = design.rec(spec), design.pulse_level(spec, 1.0)
    U = design.pulse_train(spec, np.random.default_rng(3), n=10, T=450, lead=40, tail=100)
    assert U.shape == (450, spec.m)
    assert np.allclose(U[:40], r) and np.allclose(U[-100:], r)
    d = p - r
    mv = np.abs(d) > 1e-12
    rows = U[~np.all(np.isclose(U, r), axis=1)]
    assert len(rows) > 0
    alpha = (rows[:, mv] - r[mv]) / d[mv]
    assert np.all(alpha >= 0.7 - 1e-9) and np.all(alpha <= 1.0 + 1e-9)


@pytest.mark.parametrize("sys_id", systems.SYSTEM_IDS)
def test_default_plans_match_budget(sys_id, tmp_path):
    spec = systems.get(sys_id)
    bp = budgetmod.default_plan(sys_id)
    assert sum(bp["phases"].values()) == 2000
    for phase in ("p1", "p2", "val"):
        ph = design.phase_plan(spec, phase, seed=0)
        assert ph["steps"] == bp["phases"][phase], (phase, ph["steps"])
        assert ph == design.phase_plan(spec, phase, seed=0)          # deterministic
        for e in ph["experiments"]:
            assert e["exp_id"].startswith(phase + ".") and len(e["U"]) == e["T"]
    long_T = [e for e in design.phase_plan(spec, "p1")["experiments"] if e["exp_id"] == "p1.long_train"][0]["T"]
    assert long_T == (700 if sys_id == "reservoir" else 450)


def test_materialize_once(tmp_path):
    spec = systems.get("market")
    d = tmp_path / "data_mock" / "market"
    a = design.materialize(spec, d, "p1", seed=0)
    b = design.materialize(spec, d, "p1", seed=5)                      # ignored: already materialized
    assert a["hash"] == b["hash"] and b["seed"] == 0
    led = Ledger(d, "market", mock=True)
    led.append({"t": "reset", "exp": "p1.hold_rec", "run_idx": 0, "run_id": "x", "req": "r", "obs": {}})
    design.materialize(spec, d, "p1", seed=5, force=True, ledger=led)   # hold_rec unchanged: ok
    led.append({"t": "reset", "exp": "p1.long_train", "run_idx": 0, "run_id": "y", "req": "r2", "obs": {}})
    with pytest.raises(ValueError):
        design.materialize(spec, d, "p1", seed=9, force=True, ledger=led)


def test_shop_choice():
    C = [[1.0, 1.0], [5.0, 5.0], [2.0, 2.0]]
    assert design.shop_choice(C, bought=[[1.0, 1.0]]) == 1
    assert design.shop_choice(C, mode="match", target=[2.1, 2.1]) == 2
    assert design.shop_choice(C) == 2                                   # most typical
    assert design.shop_choice([[3.0, 3.0]]) == 0


# --------------------------------------------------------------------------- end to end (mock)
@pytest.mark.parametrize("sys_id", systems.SYSTEM_IDS)
def test_mock_collect_end_to_end(sys_id, tmp_path):
    spec = systems.get(sys_id)
    ph = small_phase(spec, T=(20, 25, 15))
    gw = make_gw(sys_id, seed=2)
    d = data_dir(sys_id, True, tmp_path)
    assert d.parent.name == "data_mock"
    r = collect_phase(spec, gw, Ledger(d, sys_id, mock=True), ph, budgetmod.default_plan(sys_id),
                      max_steps=60, log=lambda *_: None)
    assert r["bought"] == 60 and gw.remaining == 1940
    assert not (tmp_path / "data").exists()
    led = Ledger(d, sys_id, mock=True)
    runs = load_runs(spec, led)
    assert [x.exp for x in runs] == [e["exp_id"] for e in ph["experiments"]]
    for run, e in zip(runs, ph["experiments"]):
        assert run.U.shape == (e["T"], spec.m) and run.Y.shape == (e["T"], spec.p)
        np.testing.assert_array_equal(run.U, np.asarray(e["U"]))
        steps = sorted((x for x in led.rows if x["t"] == "step" and x["exp"] == e["exp_id"]), key=lambda x: x["tick"])
        np.testing.assert_array_equal(run.Y, [[s["obs"][o] for o in spec.observables] for s in steps])
        reset = led.run_state(e["exp_id"])["reset"]
        np.testing.assert_array_equal(run.y0, [reset["obs"][o] for o in spec.observables])
        assert np.all(np.isfinite(run.Y))
        assert run.tags["split"] == "train" and run.tags["phase"] == "p1"
        if run.Ytrue is not None:
            assert run.Ytrue.shape == run.Y.shape
    # order experiment shopped to match the hold run's initial observation
    assert 5 + 5 + 3 <= len(led.reset_pool()) <= 3 * (5 + 5 + 3)   # extra free resets: only the latest reset is live
    assert (d / "backup").exists()


def test_cli_mock_collect_and_status(tmp_path, capsys):
    from gtlab.cli import main
    root = ["--data-root", str(tmp_path)]
    assert main(root + ["init", "--mock"]) == 0
    assert main(root + ["plan", "hospital_queue", "--phase", "val"]) == 0
    assert main(root + ["collect", "hospital_queue", "--phase", "val", "--dry-run"]) == 0
    assert "TOTAL steps to buy: 250" in capsys.readouterr().out
    assert main(root + ["mock-collect", "hospital_queue", "--phase", "val"]) == 0
    assert main(root + ["mock-collect", "hospital_queue", "--phase", "val"]) == 0  # nothing to buy
    assert main(root + ["resets", "hospital_queue", "--n", "3"]) == 0
    assert main(root + ["docs", "hospital_queue"]) == 0
    capsys.readouterr()
    assert main(root + ["status", "--mock"]) == 0
    out = capsys.readouterr().out
    assert "hospital_queue" in out and "250/250" in out
    assert not (tmp_path / "data" / "hospital_queue" / "ledger.jsonl").exists()
    d = data_dir("hospital_queue", True, tmp_path)
    assert json.loads((d / "docs.json").read_text())["budget"]["remaining"] == 1750


@pytest.mark.parametrize("sys_id", ["power_grid", "market"])
def test_mock_gateway_save_load_replays_exactly(sys_id, tmp_path):
    spec = systems.get(sys_id)
    ph = small_phase(spec)
    gw = make_gw(sys_id, seed=4)
    gw.crash_after_charge = {33}
    with pytest.raises(SimulatedCrash):
        collect_phase(spec, gw, mk_ledger(tmp_path, sys_id), ph, bplan(sys_id), max_steps=60, log=lambda *_: None)
    gw.save(tmp_path / "srv.json")
    gw2 = MockGateway.load(tmp_path / "srv.json", factory=_factory)
    assert gw2.remaining == gw.remaining and gw2.cache == gw.cache
    run = max(gw.runs, key=lambda r: gw.runs[r]["tick"])
    act = dict(spec.recovery)
    assert gw.step(run, act, "cmp")["observation"] == gw2.step(run, act, "cmp")["observation"]
    gw3 = MockGateway.load(tmp_path / "srv.json", factory=_factory)   # resume in a "new process"
    collect_phase(spec, gw3, mk_ledger(tmp_path, sys_id), ph, bplan(sys_id), max_steps=60, log=lambda *_: None)
    led = mk_ledger(tmp_path, sys_id)
    assert led.steps_charged() == 60 and gw3.charged == 60 and not led.pending_intents()


def test_plan_from_experiments_file(tmp_path):
    from gtlab.cli import main
    spec = systems.get("market")
    U = design.hold(spec, design.pulse_level(spec, 1.0), 30).tolist()
    f = tmp_path / "p2.json"
    f.write_text(json.dumps({"market": [{"exp_id": "tax_then_rate", "U": U, "shop": {"n": 2, "mode": "spread"}}]}))
    assert main(["--data-root", str(tmp_path), "plan", "market", "--phase", "p2", "--experiments", str(f)]) == 0
    ph = design.load_plan_file(data_dir("market", True, tmp_path))["phases"]["p2"]
    assert [e["exp_id"] for e in ph["experiments"]] == ["p2.tax_then_rate"] and ph["steps"] == 30
