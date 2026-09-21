"""Batch runner: prepare -> solve -> grade -> scorecard.

This is the instrument. Every change to prompts or orchestration is judged by
re-running the same instances and comparing scorecards; without that, "it feels
better" is all anyone has.

    python -m eval.run --task toy --mode single
    python -m eval.run --task toy --mode none      # grade the untouched repo
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from core.agent import Agent, AgentResult
from core.config import load_env
from core.budget import Budget
from core.llm import get_provider
from eval.grade import Scorecard, grade
from tasks.base import TaskInstance
from tracing.logger import TraceLogger

WORKSPACES = Path(".workspaces")
RESULTS = Path("eval/results")
PROMPTS = Path("prompts")


def load_task(name: str):
    if name == "toy":
        from tasks.toy import ToyTask

        return ToyTask()
    if name == "swebench":
        from tasks.swebench import SweBenchTask

        return SweBenchTask()
    raise SystemExit(f"unknown task {name!r} (expected: toy, swebench)")


def git_diff(workspace: Path) -> str:
    proc = subprocess.run(
        ["git", "diff"], cwd=workspace, capture_output=True, text=True, check=False
    )
    return proc.stdout


def build_prompt(instance: TaskInstance) -> str:
    return (
        f"## Defect report\n\n{instance.problem_statement}\n\n"
        f"## Reported failing test(s)\n\n"
        + "\n".join(f"- {t}" for t in instance.fail_to_pass)
        + "\n\nThe repository is your working directory. Fix it."
    )


def solve_single(instance: TaskInstance, *, trace, budget: Budget) -> AgentResult:
    system = (PROMPTS / "single_agent.md").read_text(encoding="utf-8")
    agent = Agent(
        "repair",
        system,
        workspace=instance.workspace,
        tier="strong",
        provider=get_provider(),
        budget=budget,
        trace=trace,
    )
    return agent.run(build_prompt(instance))


def run_instance(task, instance_id: str, mode: str, trace: TraceLogger) -> dict[str, Any]:
    workspace = WORKSPACES / trace.run_id / instance_id.replace("/", "_")
    instance = task.prepare(instance_id, workspace)
    budget = Budget()
    started = time.monotonic()

    agent_result: AgentResult | None = None
    if mode == "single":
        agent_result = solve_single(instance, trace=trace, budget=budget)
    elif mode == "multi":
        from orchestrator import solve_multi

        agent_result = solve_multi(instance, trace=trace, budget=budget)
    elif mode != "none":
        raise SystemExit(f"unknown mode {mode!r} (expected: none, single, multi)")

    card: Scorecard = grade(instance)
    record = {
        "instance_id": instance_id,
        "mode": mode,
        **card.as_dict(),
        "wall_clock_s": round(time.monotonic() - started, 1),
        "budget": budget.as_dict(),
        "agent": agent_result.as_dict() if agent_result else None,
        "diff": git_diff(instance.workspace),
    }
    trace.event("instance_graded", **{k: v for k, v in record.items() if k != "diff"})
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description="Run and grade the agent harness.")
    ap.add_argument("--task", default="toy")
    ap.add_argument("--mode", default="single", choices=["none", "single", "multi"])
    ap.add_argument("--n", type=int, default=None, help="max instances")
    ap.add_argument("--instance", default=None, help="run one specific instance id")
    args = ap.parse_args()

    load_env()
    task = load_task(args.task)
    ids = [args.instance] if args.instance else task.instance_ids(limit=args.n)

    trace = TraceLogger(task=f"{args.task}:{args.mode}")
    records = [run_instance(task, iid, args.mode, trace) for iid in ids]

    total = len(records)
    avg = sum(r["score"] for r in records) / total if total else 0.0
    resolved = sum(1 for r in records if r["resolved"])
    cost = sum(r["budget"]["cost_usd"] for r in records)

    summary = {
        "run_id": trace.run_id,
        "task": args.task,
        "mode": args.mode,
        "instances": total,
        "avg_score": round(avg, 1),
        "resolved": resolved,
        "resolve_rate": round(resolved / total, 3) if total else 0.0,
        "total_cost_usd": round(cost, 4),
        "records": records,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{trace.run_id}-{args.task}-{args.mode}.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\n=== {args.task} / {args.mode} ===")
    for r in records:
        flag = "RESOLVED" if r["resolved"] else "         "
        h = r["hidden"]
        print(
            f"  {flag}  score {r['score']:>3}/100  "
            f"target={'ok' if r['target']['all_green'] else 'FAIL'}  "
            f"regress={'ok' if r['regression']['all_green'] else 'BROKE'}  "
            f"hidden={h['passed']}/{h['passed'] + h['failed'] + h['errors']}  "
            f"{r['instance_id']}"
        )
        for note in r["notes"]:
            print(f"            - {note}")
    print(
        f"\n  avg score {summary['avg_score']}/100   "
        f"resolved {resolved}/{total}   "
        f"cost ${summary['total_cost_usd']}"
    )
    print(f"  scorecard: {out}")
    print(f"  trace:     {trace.path}")


if __name__ == "__main__":
    main()
