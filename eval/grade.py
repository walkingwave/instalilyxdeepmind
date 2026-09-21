"""Scoring, modelled on the stated hackathon rubric.

"Every task is scored out of 100, weighted to fixes that survive tests you have
not seen." So: the held-out suite carries the largest single weight, the
regression guard carries as much as the target fix, and passing only the
visible test is worth well under half.

  target      30   the reported failure now passes
  regression  30   nothing that used to pass broke
  hidden      40   held-out tests, prorated by how many pass

A patch that games the visible test scores 60. One that restores the actual
contract scores 100. That gap is the whole game.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tasks.base import TaskInstance

WEIGHT_TARGET = 30
WEIGHT_REGRESSION = 30
WEIGHT_HIDDEN = 40

_COUNT_RX = {
    "passed": re.compile(r"(\d+) passed"),
    "failed": re.compile(r"(\d+) failed"),
    "error": re.compile(r"(\d+) error"),
}


@dataclass
class SuiteResult:
    passed: int = 0
    failed: int = 0
    errors: int = 0
    returncode: int = 0
    output: str = ""

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors

    @property
    def all_green(self) -> bool:
        return self.total > 0 and self.failed == 0 and self.errors == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "all_green": self.all_green,
        }


@dataclass
class Scorecard:
    instance_id: str
    target: SuiteResult = field(default_factory=SuiteResult)
    regression: SuiteResult = field(default_factory=SuiteResult)
    hidden: SuiteResult = field(default_factory=SuiteResult)
    score: int = 0
    resolved: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "score": self.score,
            "resolved": self.resolved,
            "target": self.target.as_dict(),
            "regression": self.regression.as_dict(),
            "hidden": self.hidden.as_dict(),
            "notes": self.notes,
        }


def run_pytest(workspace: Path, targets: list[str], timeout: int = 300) -> SuiteResult:
    if not targets:
        return SuiteResult()
    cmd = ["python", "-m", "pytest", "-q", "--no-header", "--tb=no", *targets]
    try:
        proc = subprocess.run(
            cmd, cwd=workspace, capture_output=True, text=True, timeout=timeout
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        return SuiteResult(errors=len(targets), returncode=-1, output="timed out")

    counts = {}
    for key, rx in _COUNT_RX.items():
        m = rx.search(out)
        counts[key] = int(m.group(1)) if m else 0
    return SuiteResult(
        passed=counts["passed"],
        failed=counts["failed"],
        errors=counts["error"],
        returncode=rc,
        output=out[-4000:],
    )


def grade(instance: TaskInstance) -> Scorecard:
    card = Scorecard(instance_id=instance.id)

    card.target = run_pytest(instance.workspace, instance.fail_to_pass)
    card.regression = run_pytest(instance.workspace, instance.pass_to_pass)

    installed = instance.install_hidden_tests()
    try:
        card.hidden = run_pytest(instance.workspace, installed)
    finally:
        instance.remove_hidden_tests(installed)

    score = 0
    if card.target.all_green:
        score += WEIGHT_TARGET
    else:
        card.notes.append("target failure not fixed")

    if card.regression.all_green:
        score += WEIGHT_REGRESSION
    else:
        card.notes.append(f"regressions: {card.regression.failed} previously-passing tests broke")

    if card.hidden.total:
        ratio = card.hidden.passed / card.hidden.total
        score += round(WEIGHT_HIDDEN * ratio)
        if ratio < 1.0:
            card.notes.append(
                f"hidden suite {card.hidden.passed}/{card.hidden.total} "
                "-- patch did not generalize"
            )

    card.score = score
    # "Resolved" is the strict bar: everything green, including held-out tests.
    card.resolved = (
        card.target.all_green and card.regression.all_green and card.hidden.all_green
    )
    return card
