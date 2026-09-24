"""Gateways: the kit Client surface (reset/step/budget/brief/documents), real and mock, plus
SpendGuard, the only thing that is allowed to call a paid step().

RealGateway construction is gated (see its docstring) and always raises under pytest.
MockGateway keeps its own budget and honors idempotency keys like the real server.
"""
from __future__ import annotations

import copy
import math
import os
import json
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from gtlab import systems

ROOT = Path(__file__).resolve().parent.parent


class GatewayError(RuntimeError):
    """The gateway refused a request (bad action, unknown run, no budget). Nothing was charged."""


class SpendError(RuntimeError):
    """A spend rule would be broken. Raised before anything is sent."""


class CapError(SpendError):
    pass


class DriftError(SpendError):
    pass


@runtime_checkable
class Gateway(Protocol):
    def reset(self, system_id: str, request_id: str | None = None) -> dict: ...
    def step(self, run_id: str, intervention: dict, request_id: str | None = None) -> dict: ...
    def budget(self, system_id: str) -> dict: ...
    def brief(self, system_id: str) -> dict: ...
    def documents(self, system_id: str): ...


def server_remaining(resp: dict) -> int:
    v = resp.get("simulator_steps_remaining", resp.get("remaining"))
    if v is None:
        raise GatewayError(f"budget response has no remaining field: {resp}")
    return int(v)


def _under_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ or "_pytest" in sys.modules or "pytest" in sys.modules


# --------------------------------------------------------------------------- real
class RealGateway:
    """Wraps kit/client.py Client.

    spend=True  (paid steps allowed): needs env GT_ALLOW_SPEND=1 and GROUNDTRUTH_KEY.
    spend=False (free endpoints only: reset/budget/brief/documents): needs GT_ALLOW_REAL=1 and
                GROUNDTRUTH_KEY; step() raises.
    Under pytest the constructor always raises.
    """

    def __init__(self, spend: bool = False, base_url: str | None = None, transport=None):
        if _under_pytest():
            raise SpendError("RealGateway cannot be constructed under pytest")
        key = os.environ.get("GROUNDTRUTH_KEY")
        if not key:
            raise SpendError("GROUNDTRUTH_KEY not set")
        if spend is True:
            if os.environ.get("GT_ALLOW_SPEND") != "1":
                raise SpendError("paid gateway needs GT_ALLOW_SPEND=1 in the environment")
        elif spend is False:
            if os.environ.get("GT_ALLOW_REAL") != "1" and os.environ.get("GT_ALLOW_SPEND") != "1":
                raise SpendError("real gateway (free endpoints) needs GT_ALLOW_REAL=1")
        else:
            raise SpendError("spend must be exactly True or False")
        base_url = base_url or os.environ.get("GROUNDTRUTH_GATEWAY_URL")
        if not base_url:
            raise SpendError("GROUNDTRUTH_GATEWAY_URL not set")
        sys.path.insert(0, str(ROOT / "kit"))
        try:
            from client import Client          # kit/client.py
        finally:
            sys.path.pop(0)
        self.spend = spend
        self._c = Client(base_url, key, transport=transport)

    def reset(self, system_id, request_id=None):
        return self._c.reset(system_id, request_id)

    def step(self, run_id, intervention, request_id=None):
        if not self.spend:
            raise SpendError("this RealGateway was built for free endpoints only")
        if not request_id:
            raise SpendError("paid steps must carry a deterministic request id")
        return self._c.step(run_id, intervention, request_id)

    def budget(self, system_id):
        return self._c.budget(system_id)

    def brief(self, system_id):
        return self._c.brief(system_id)

    def documents(self, system_id):
        return self._c.documents(system_id)

    def close(self):
        self._c.close()


# --------------------------------------------------------------------------- mock
def _default_factory(system_id, mech, theta, noise, world_seed, run_seed):
    """One simulator instance per run. world_seed fixes the secret true theta (same physics for
    every run of this gateway); run_seed only drives the observation noise."""
    from gtlab.mocks import MockSystem      # lazy: written by another engineer
    m = MockSystem(system_id, mech=mech, theta=theta, noise=noise, seed=world_seed)
    m.rng = np.random.default_rng(run_seed)
    return m


class MockGateway:
    """Offline stand-in for the research gateway, for one system.

    Responses: reset -> {"run_id","observation"}; step -> {"observation","run_id","tick"};
    budget -> {"remaining","simulator_steps_remaining"}. Mock-only extra: "mock_truth" (noiseless).
    Same request id -> same response, charged once. Bad actions are refused for free.

    Fault injection for tests: crash_before_send / crash_after_charge are sets of 1-based
    call counters of step(); the mock raises SimulatedCrash at that point.
    """

    def __init__(self, system_id, mech=("A", "B"), seed=0, budget=2000, noise=0.03, theta=None,
                 factory=None):
        self.system_id = system_id
        self.spec = systems.get(system_id)
        self.mech = tuple(mech)
        self.seed = int(seed)
        self.total = int(budget)
        self.remaining = int(budget)
        self.noise = noise
        self.theta = theta
        self.factory = factory or _default_factory
        self.runs = {}            # run_id -> {"sys": MockSystem, "tick": int}
        self.cache = {}           # request_id -> response
        self.n_resets = 0
        self.calls = {"reset": 0, "step": 0, "budget": 0, "brief": 0, "documents": 0}
        self.charged = 0
        self.crash_before_send: set[int] = set()
        self.crash_after_charge: set[int] = set()

    def _check_sys(self, system_id):
        if system_id != self.system_id:
            raise GatewayError(f"this mock serves {self.system_id}, not {system_id}")

    def reset(self, system_id, request_id=None):
        self.calls["reset"] += 1
        self._check_sys(system_id)
        if request_id and request_id in self.cache:
            return copy.deepcopy(self.cache[request_id])
        k = self.n_resets
        self.n_resets += 1
        run_id, m, y0 = self._new_run(k)
        resp = {"run_id": run_id, "observation": {o: float(v) for o, v in zip(self.spec.observables, y0)}}
        lt = getattr(m, "last_truth", None)
        if lt is not None:
            resp["mock_truth"] = {o: float(v) for o, v in zip(self.spec.observables, np.asarray(lt, float))}
        if request_id:
            self.cache[request_id] = copy.deepcopy(resp)
        return resp

    def _new_run(self, k):
        run_seed = (self.seed * 1_000_003 + k * 7919 + 12345) % (2 ** 31)
        m = self.factory(self.system_id, self.mech, self.theta, self.noise, self.seed, run_seed)
        y0 = np.asarray(m.reset(np.random.default_rng(run_seed)), float)
        run_id = f"mock-{self.system_id}-{self.seed}-{k:05d}"
        self.runs[run_id] = {"sys": m, "tick": 0, "k": k, "actions": []}
        return run_id, m, y0

    def _validate(self, intervention):
        if not isinstance(intervention, dict) or set(intervention) != set(self.spec.controls):
            raise GatewayError(f"intervention keys must be exactly {self.spec.controls}")
        u = []
        for c in self.spec.controls:
            v = intervention[c]
            if isinstance(v, bool) or not isinstance(v, (int, float, np.floating, np.integer)):
                raise GatewayError(f"{c} is not a number")
            v = float(v)
            lo, hi = self.spec.bounds[c]
            if not math.isfinite(v) or v < lo or v > hi:
                raise GatewayError(f"{c}={v} outside [{lo}, {hi}]")
            u.append(v)
        return np.array(u)

    def step(self, run_id, intervention, request_id=None):
        self.calls["step"] += 1
        n = self.calls["step"]
        if n in self.crash_before_send:
            raise SimulatedCrash(f"crash before send (call {n})")
        if request_id and request_id in self.cache:
            return copy.deepcopy(self.cache[request_id])
        if run_id not in self.runs:
            raise GatewayError(f"unknown run {run_id}")
        u = self._validate(intervention)
        if self.remaining <= 0:
            raise GatewayError("budget exhausted")
        self.remaining -= 1
        self.charged += 1
        r = self.runs[run_id]
        r["actions"].append(u.tolist())
        y = np.asarray(r["sys"].step(u), float)
        tick = r["tick"]
        r["tick"] += 1
        resp = {"run_id": run_id, "tick": tick,
                "observation": {o: float(v) for o, v in zip(self.spec.observables, y)}}
        lt = getattr(r["sys"], "last_truth", None)
        if lt is not None:
            resp["mock_truth"] = {o: float(v) for o, v in zip(self.spec.observables, np.asarray(lt, float))}
        if request_id:
            self.cache[request_id] = copy.deepcopy(resp)
        if n in self.crash_after_charge:
            raise SimulatedCrash(f"crash after charge (call {n})")
        return resp

    def budget(self, system_id):
        self.calls["budget"] += 1
        self._check_sys(system_id)
        return {"system_id": system_id, "remaining": self.remaining,
                "simulator_steps_remaining": self.remaining}

    def brief(self, system_id):
        self.calls["brief"] += 1
        self._check_sys(system_id)
        s = self.spec
        return {"system_id": s.id, "description": s.brief, "observables": list(s.observables),
                "interventions": {c: list(s.bounds[c]) for c in s.controls},
                "recovery": dict(s.recovery), "pulse": dict(s.pulse), "mock": True}

    def documents(self, system_id):
        self.calls["documents"] += 1
        self._check_sys(system_id)
        return [{"id": "mock-notes", "title": "Mock operating notes", "text": "Offline mock; no real documents."}]

    # persistence so CLI mock collection can resume across processes. Saved as JSON and
    # rebuilt by replaying every run's actions (deterministic seeds), so no pickled modules.
    def save(self, path):
        path = Path(path)
        state = {"system_id": self.system_id, "mech": list(self.mech), "seed": self.seed,
                 "total": self.total, "remaining": self.remaining, "noise": self.noise,
                 "theta": self.theta, "n_resets": self.n_resets, "charged": self.charged,
                 "cache": self.cache,
                 "runs": {rid: {"k": r["k"], "actions": r["actions"]} for rid, r in self.runs.items()}}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        os.replace(tmp, path)

    @staticmethod
    def load(path, factory=None):
        st = json.loads(Path(path).read_text())
        g = MockGateway(st["system_id"], mech=st["mech"], seed=st["seed"], budget=st["total"],
                        noise=st["noise"], theta=st["theta"], factory=factory)
        g.remaining, g.charged, g.n_resets, g.cache = st["remaining"], st["charged"], st["n_resets"], st["cache"]
        for rid, r in sorted(st["runs"].items(), key=lambda kv: kv[1]["k"]):
            rid2, m, _ = g._new_run(r["k"])
            assert rid2 == rid
            for u in r["actions"]:
                m.step(np.asarray(u, float))
            g.runs[rid]["actions"] = list(r["actions"])
            g.runs[rid]["tick"] = len(r["actions"])
        return g


class SimulatedCrash(Exception):
    """Test-only: a process kill in the middle of a call."""


# --------------------------------------------------------------------------- guard
class SpendGuard:
    """The only caller of a paid step(). Enforces caps before sending and reconciles with the
    free server budget every `every` steps.

    base_charged: steps the ledger says were already bought (all phases, incl. external spend).
    pending: intents without a step row (possibly charged on the server, possibly not).
    phase_cap_left / max_steps: hard caps for this session. total/floor from budget.json.
    """

    def __init__(self, gateway, spec, *, base_charged, phase_cap_left, max_steps, total=2000,
                 floor=0, pending=0, every=50, ledger=None):
        self.g = gateway
        self.spec = spec
        self.base = int(base_charged)
        self.phase_cap_left = int(phase_cap_left)
        self.max_steps = int(max_steps)
        self.total = int(total)
        self.floor = int(floor)
        self.pending = int(pending)
        self.every = int(every)
        self.ledger = ledger
        self.sent = 0                 # paid step calls sent this session (retries included)
        self.completed = 0            # steps that returned (== new step rows)
        self.server_cached = None

    def _read_budget(self, where):
        rem = server_remaining(self.g.budget(self.spec.id))
        expected = self.total - self.base - self.completed
        if self.ledger is not None:
            self.ledger.append({"t": "budget", "remaining": rem, "expected": expected, "where": where})
        return rem, expected

    def preflight(self, planned: int, reconcile: bool = False):
        rem, expected = self._read_budget("preflight")
        # a dangling intent may or may not have been charged by the server
        if not (expected - self.pending <= rem <= expected):
            drift = expected - rem
            if reconcile and self.ledger is not None and drift > 0:
                self.ledger.append({"t": "external_spend", "n": drift, "note": "reconciled at preflight"})
                self.base += drift
            else:
                raise DriftError(f"{self.spec.id}: server remaining {rem}, ledger expects {expected} "
                                 f"(pending {self.pending}); pass --reconcile if steps were spent elsewhere")
        self.server_cached = rem
        if planned > rem - self.floor:
            raise CapError(f"{self.spec.id}: plan needs {planned}, server has {rem} (floor {self.floor})")
        if planned > self.max_steps:
            raise CapError(f"{self.spec.id}: plan needs {planned} > --max-steps {self.max_steps}")
        if planned > self.phase_cap_left:
            raise CapError(f"{self.spec.id}: plan needs {planned} > phase cap left {self.phase_cap_left}")
        return rem

    def _check_action(self, action):
        if set(action) != set(self.spec.controls):
            raise SpendError(f"action keys {sorted(action)} != controls {self.spec.controls}")
        for c, v in action.items():
            lo, hi = self.spec.bounds[c]
            if not (isinstance(v, (int, float)) and math.isfinite(v) and lo <= v <= hi):
                raise SpendError(f"action {c}={v!r} outside [{lo}, {hi}]")

    def step(self, run_id, action, request_id):
        if not request_id:
            raise SpendError("paid step without request id")
        self._check_action(action)
        n = self.sent + 1
        left_total = self.total - self.base - self.sent
        cached = self.server_cached if self.server_cached is not None else left_total
        limit = min(self.phase_cap_left, self.max_steps, left_total, cached - self.floor)
        if n > min(self.phase_cap_left, self.max_steps) or 1 > min(left_total, cached - self.floor):
            raise CapError(f"{self.spec.id}: step {n} would exceed cap (limit {limit})")
        self.sent += 1
        resp = self.g.step(run_id, action, request_id=request_id)
        self.completed += 1
        if self.server_cached is not None:
            self.server_cached -= 1
        if self.completed % self.every == 0:
            self.reconcile("periodic")
        return resp

    def reconcile(self, where="periodic"):
        rem, expected = self._read_budget(where)
        if not (expected - self.pending <= rem <= expected):
            raise DriftError(f"{self.spec.id}: budget drift: server {rem}, expected {expected} "
                             f"(pending {self.pending})")
        self.server_cached = rem
        return rem
