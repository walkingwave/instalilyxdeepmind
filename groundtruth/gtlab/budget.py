"""Per-system budget plan (data/<sys>/budget.json) and reconciliation vs the server budget."""
from __future__ import annotations

import json
from pathlib import Path

TOTAL = 2000


def default_plan(system_id: str) -> dict:
    if system_id == "reservoir":
        # long seasonal run (700 instead of 450) is taken out of p2
        phases = {"p1": 1250, "p2": 250, "val": 250, "reserve": 250}
    else:
        phases = {"p1": 1000, "p2": 500, "val": 250, "reserve": 250}
    assert sum(phases.values()) == TOTAL
    return {"system": system_id, "total": TOTAL, "phases": phases, "floor": 0}


def path(directory) -> Path:
    return Path(directory) / "budget.json"


def load(directory, system_id: str | None = None) -> dict:
    p = path(directory)
    if p.exists():
        plan = json.loads(p.read_text())
    elif system_id is not None:
        plan = default_plan(system_id)
    else:
        raise FileNotFoundError(p)
    validate(plan)
    return plan


def write(directory, plan: dict, overwrite=False) -> Path:
    validate(plan)
    p = path(directory)
    if p.exists() and not overwrite:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(plan, indent=1) + "\n")
    return p


def validate(plan: dict):
    if sum(plan["phases"].values()) > plan["total"]:
        raise ValueError(f"phase caps {plan['phases']} exceed total {plan['total']}")
    if plan["total"] > TOTAL:
        raise ValueError("total above 2000")
    if any(v < 0 for v in plan["phases"].values()) or plan.get("floor", 0) < 0:
        raise ValueError("negative cap")


def phase_left(plan: dict, ledger, phase: str) -> int:
    """Steps this phase may still buy: its cap minus steps already in the ledger for it, and
    never more than what is left of the total."""
    if phase not in plan["phases"]:
        raise KeyError(f"phase {phase} not in budget.json")
    cap_left = plan["phases"][phase] - ledger.phase_charged(phase)
    total_left = plan["total"] - ledger.steps_charged() - plan.get("floor", 0)
    return max(0, min(cap_left, total_left))


def reconcile(server_resp: dict, ledger, total: int = TOTAL) -> dict:
    from gtlab.gateway import server_remaining
    rem = server_remaining(server_resp)
    charged = ledger.steps_charged()
    pending = len(ledger.pending_intents())
    expected = total - charged
    return {"server_remaining": rem, "ledger_charged": charged, "expected_remaining": expected,
            "pending_intents": pending, "drift": expected - rem,
            "ok": expected - pending <= rem <= expected}


def summary(plan: dict, ledger) -> dict:
    return {ph: {"cap": cap, "spent": ledger.phase_charged(ph)} for ph, cap in plan["phases"].items()}
