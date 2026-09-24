# gt: GroundTruth Toronto 26 workbench

Tooling for the GroundTruth hackathon: buy simulator data carefully, fit grey-box/ODE models, package
forecasters and check them locally before upload. Full design is in `PLAN.md`.

Each system gets 2,000 paid steps, and they never come back. Most of this repo exists so we can't waste them.

## Spending safety

- **Mock by default.** Every command runs against `MockGateway` and writes to `data_mock/` unless you pass `--spend`.
  Real data only ever goes to `data/`, mock data only to `data_mock/`. The ledger refuses to open the wrong one.
- **Paid steps need all of these:** `--spend`, `--max-steps N` (no default), `--yes` or typing `SPEND` at the
  prompt, and `GT_ALLOW_SPEND=1` in the env. The key alone does nothing.
- **Caps before sending.** `data/<sys>/budget.json` has per-phase caps (p1 1000 / p2 500 / val 250 / reserve 250;
  reservoir is 1250 / 250 / 250 / 250 because of the long seasonal run). A phase that needs more than its cap,
  `--max-steps`, or what the server says is left gets refused before any call.
- **Reconciliation.** Before spending we read the free `budget` endpoint and check it against the ledger. Every
  50 steps we check again. On a mismatch we abort. `--reconcile` records the gap as external spend if you
  really did spend elsewhere (e.g. you ran the kit's `collect.py`, so don't).
- **Write-ahead ledger + deterministic idempotency keys.** `intent` is fsynced before each step, `step` after.
  Request ids are `uuid5(sys/exp/run/tick)`, so after a crash the same key is re-sent and the server dedupes it.
  Rerunning a command skips finished experiments and resumes partial ones on the same `run_id`.
- **Plans are materialized once** into `plan.json` (seeded, hashed). Changing a started experiment needs a new id.
- One collector per system at a time (`.lock`). The ledger is backed up to `data/<sys>/backup/` after each phase.
  Commit `data/`: it is irreplaceable.
- `gt docs --real` and `gt resets --real` hit only free endpoints (brief, documents, budget, reset). They need
  `GT_ALLOW_REAL=1`. They still talk to the real gateway, so run them on purpose.
- Tests can never build a real gateway (`RealGateway` raises under pytest).

## Setup

Python 3.12, same pins as the scoring sandbox:

```sh
uv venv --python 3.12 .venv312
uv pip install --python .venv312 numpy==2.3.5 scipy==1.16.3 scikit-learn==1.7.2 joblib==1.5.2 httpx pytest
uv pip install --python .venv312 -e .          # gives the `gt` command
.venv312/bin/python -m pytest -q
```

Env for real runs (keep the key out of files and zips):

```sh
export GROUNDTRUTH_GATEWAY_URL='https://gt-gateway-...'
export GROUNDTRUTH_KEY='...'
export GT_ALLOW_REAL=1      # free endpoints (docs/resets)
export GT_ALLOW_SPEND=1     # only when you actually mean to buy steps
```

## Daily commands

```sh
gt init                                     # systems.json + data/<sys>/budget.json
gt docs --all --real                        # free: brief + documents -> data/<sys>/docs.json
gt resets --all --n 40 --real               # free: initial-observation pool -> data/<sys>/resets.jsonl

gt plan power_grid --phase p1 --real-dir    # write the phase into data/<sys>/plan.json, print cost
gt collect --all --phase p1 --dry-run --spend --max-steps 1250    # exact step cost, zero calls
gt collect --all --phase p1 --spend --max-steps 1250 --yes        # buys steps (resumable)

gt mock-collect --all --phase p1            # same thing against the mock, into data_mock/
gt status [--mock]                          # spent per phase, last server budget, dangling intents

gt fit SYS [--real-dir] | gt validate SYS | gt package --all --tag x | gt check submissions/<tag>
gt pipeline --all --phase p1 --spend --max-steps 1250 --yes       # collect -> fit -> package -> check
gt mock-bench --all
```

`--max-steps` is per system. 1000 is enough for p1 everywhere except reservoir (1250). If one system fails
(cap, drift, network), the others still run and the failed one is listed at the end; rerun to resume it.

p2 in `design.py` is a placeholder. After looking at p1, write the real p2 experiments (PLAN §7) as JSON
(`[{"exp_id": "...", "U": [[...], ...], "shop": {"n": 5, "mode": "spread"}}]`, controls in brief order) and
`gt plan SYS --phase p2 --real-dir --experiments p2.json [--force]`. `--force` is refused once a changed
experiment has started.

First upload (no data needed): `gt package --persistence` then `gt check submissions/<dir>`.

## Upload checklist

1. `gt check <zip>` passes (import whitelist, secrets scan, 40x4000 sandbox run, timing) and `check_report.json` is next to the zip.
2. All 10 system folders are at the zip root. No enclosing folder, no venv, no keys, no `__pycache__`.
3. Max 3 accepted uploads per system per Toronto day, shared between public and final. Keep one spare.
4. Keep every zip you upload under `submissions/`. The latest accepted upload counts, even if worse.
5. **The Final tab must be uploaded separately, Sep 28 12:00 to Sep 30 12:00 (Toronto).** Public uploads do
   not carry over. Submit all 10 early on Sep 28 as insurance, then confirm receipts before 12:00 Sep 30.
