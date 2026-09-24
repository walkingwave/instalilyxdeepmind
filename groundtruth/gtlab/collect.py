"""Execute one plan phase through a Gateway, via the write-ahead ledger.

Completed experiments are skipped, partial ones resume at the next tick on their stored
run_id, and a dangling intent is re-sent with the same idempotency key (never re-bought).
"""
from __future__ import annotations

import math

import numpy as np

from gtlab import budget as budgetmod
from gtlab.design import shop_choice
from gtlab.gateway import CapError, GatewayError, SpendGuard
from gtlab.ledger import LedgerError, req_id


def phase_cost(ledger, phase_doc) -> list[dict]:
    rows = []
    for e in phase_doc["experiments"]:
        st = ledger.run_state(e["exp_id"], e.get("run_idx", 0))
        done = min(st["next"], e["T"])
        rows.append({"exp_id": e["exp_id"], "T": e["T"], "done": done, "todo": e["T"] - done,
                     "started": st["reset"] is not None, "pending": st["pending"] is not None})
    return rows


def format_cost(system_id, phase, rows) -> str:
    out = [f"{system_id} {phase}:"]
    for r in rows:
        flag = "done" if r["todo"] == 0 else ("resume" if r["started"] else "new")
        out.append(f"  {r['exp_id']:<22} T={r['T']:>4}  done={r['done']:>4}  todo={r['todo']:>4}  {flag}"
                   + ("  (dangling intent: re-send same key)" if r["pending"] else ""))
    out.append(f"  TOTAL steps to buy: {sum(r['todo'] for r in rows)}")
    return "\n".join(out)


def _check_obs(spec, resp):
    obs = resp.get("observation") if isinstance(resp, dict) else None
    if not isinstance(obs, dict) or any(o not in obs for o in spec.observables):
        raise GatewayError(f"malformed observation: {resp!r}")
    for o in spec.observables:
        v = obs[o]
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise GatewayError(f"non-finite observation {o}={v!r}")
    return {o: float(obs[o]) for o in spec.observables}


def _do_reset(spec, gateway, ledger, e, log):
    exp, idx = e["exp_id"], e.get("run_idx", 0)
    shop = e.get("shop") or {"n": 1, "mode": "spread"}
    n = max(1, int(shop.get("n", 1)))
    cands = []
    for k in range(n):
        rq = req_id(spec.id, exp, idx, f"reset{k}")
        resp = gateway.reset(spec.id, request_id=rq)
        obs = _check_obs(spec, resp)
        ledger.log_reset_candidate({"exp": exp, "run_idx": idx, "k": k, "req": rq,
                                    "run_id": resp["run_id"], "obs": obs, "raw": resp})
        cands.append((rq, resp, obs))
    target = None
    if shop.get("mode") == "match":
        ref = ledger.run_state(shop["match_exp"], 0)["reset"]
        if ref is not None:
            target = [ref["obs"][o] for o in spec.observables]
    bought = [[b[o] for o in spec.observables] for b in ledger.bought_initials()]

    def choose():
        C = [[c[2][o] for o in spec.observables] for c in cands]
        if target is not None:
            return shop_choice(C, mode="match", target=target)
        return shop_choice(C, bought=bought, mode="spread")

    # The real gateway keeps only the LATEST reset alive ("a new run replaces the old one"),
    # so we may only buy steps on the most recent candidate. Keep resetting (free) until the
    # latest candidate is the best seen so far, up to 2n extra resets; then take the latest.
    k = choose()
    extra = 0
    while k != len(cands) - 1 and extra < 2 * n:
        kk = len(cands)
        rq = req_id(spec.id, exp, idx, f"reset{kk}")
        resp = gateway.reset(spec.id, request_id=rq)
        obs = _check_obs(spec, resp)
        ledger.log_reset_candidate({"exp": exp, "run_idx": idx, "k": kk, "req": rq,
                                    "run_id": resp["run_id"], "obs": obs, "raw": resp})
        cands.append((rq, resp, obs))
        extra += 1
        k = choose()
    k = len(cands) - 1
    n = len(cands)
    rq, resp, obs = cands[k]
    row = ledger.append({"t": "reset", "exp": exp, "run_idx": idx, "run_id": resp["run_id"],
                         "req": rq, "obs": obs, "raw": resp, "shop_k": k, "shop_n": n})
    log(f"  reset {exp}: picked candidate {k}/{n} run_id={resp['run_id']}")
    return row


def collect_phase(spec, gateway, ledger, phase_doc, budget_plan, *, max_steps=None, dry_run=False,
                  reconcile=False, every=50, log=print, backup=True):
    """Run a phase. Returns {"bought": n, "cost": rows}. dry_run never touches the gateway."""
    phase = phase_doc["phase"]
    rows = phase_cost(ledger, phase_doc)
    needed = sum(r["todo"] for r in rows)
    log(format_cost(spec.id, phase, rows))
    if dry_run or needed == 0:
        return {"bought": 0, "cost": rows, "needed": needed, "dry_run": dry_run}
    with ledger:
        cap_left = budgetmod.phase_left(budget_plan, ledger, phase)
        if needed > cap_left:
            raise CapError(f"{spec.id} {phase}: needs {needed}, phase cap left {cap_left}")
        if max_steps is None:
            raise CapError("max_steps is required")
        if needed > max_steps:
            raise CapError(f"{spec.id} {phase}: needs {needed} > max_steps {max_steps}")
        guard = SpendGuard(gateway, spec, base_charged=ledger.steps_charged(), phase_cap_left=cap_left,
                           max_steps=max_steps, total=budget_plan["total"],
                           floor=budget_plan.get("floor", 0), pending=len(ledger.pending_intents()),
                           every=every, ledger=ledger)
        guard.preflight(needed, reconcile=reconcile)
        bought = 0
        for e in phase_doc["experiments"]:
            exp, idx, T = e["exp_id"], e.get("run_idx", 0), e["T"]
            st = ledger.run_state(exp, idx)
            if st["next"] >= T:
                continue
            reset = st["reset"] or _do_reset(spec, gateway, ledger, e, log)
            run_id = reset["run_id"]
            U = np.asarray(e["U"], float)
            first = st["next"]
            for tick in range(first, T):
                action = {c: float(U[tick, j]) for j, c in enumerate(spec.controls)}
                rq = req_id(spec.id, exp, idx, tick)
                pend = st["pending"] if tick == first else None
                if pend is not None:
                    if pend["req"] != rq or pend["action"] != action:
                        raise LedgerError(f"{exp} tick {tick}: dangling intent differs from plan; "
                                          "the schedule changed after it was started")
                else:
                    ledger.append({"t": "intent", "exp": exp, "run_idx": idx, "run_id": run_id,
                                   "tick": tick, "req": rq, "action": action})
                resp = guard.step(run_id, action, rq)
                if pend is not None:
                    guard.pending = max(0, guard.pending - 1)
                obs = _check_obs(spec, resp)
                ledger.append({"t": "step", "exp": exp, "run_idx": idx, "run_id": run_id,
                               "tick": tick, "req": rq, "action": action, "obs": obs, "raw": resp})
                bought += 1
            log(f"  {exp}: complete ({T} ticks)")
        guard.reconcile("end")
        if backup:
            ledger.backup(phase)
    return {"bought": bought, "cost": rows, "needed": needed, "dry_run": False}
