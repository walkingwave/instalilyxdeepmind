"""The toy task: one planted bug, one visible failure, five held-out tests.

Deliberately constructed so the lazy fix passes everything the agent can see
and still loses most of the hidden points. Measured baseline for the buggy
code plus a sort-only patch: visible 5/5, hidden 2/5.

This is the smoke test for the whole harness -- fast, free, and offline.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from tasks.base import TaskInstance

FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "fixtures"
REPO_SRC = FIXTURE_ROOT / "toy_repo"
HIDDEN_SRC = FIXTURE_ROOT / "toy_hidden"

PROBLEM_STATEMENT = """\
`merge_intervals` in `toybug/intervals.py` returns the wrong result when the
caller passes intervals that are not already sorted by start.

Reproduction:

    >>> from toybug import merge_intervals
    >>> merge_intervals([(5, 7), (1, 3), (2, 4)])
    [(5, 7)]

Expected: [(1, 4), (5, 7)]

The failing test is tests/test_intervals.py::test_unsorted_input.

Read the docstring on `merge_intervals` -- it states the contract the function
is supposed to honour. Fix the implementation so it honours that contract.
"""


class ToyTask:
    name = "toy"

    def instance_ids(self, limit: int | None = None) -> list[str]:
        return ["toy-intervals"]

    def prepare(self, instance_id: str, dest: Path) -> TaskInstance:
        dest = Path(dest)
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(REPO_SRC, dest)

        # A git checkpoint per workspace gives us free diff extraction and
        # rollback between candidate patches.
        subprocess.run(["git", "init", "-q"], cwd=dest, check=False)
        subprocess.run(["git", "add", "-A"], cwd=dest, check=False)
        subprocess.run(
            ["git", "-c", "user.email=harness@local", "-c", "user.name=harness",
             "commit", "-q", "-m", "base"],
            cwd=dest,
            check=False,
        )

        return TaskInstance(
            id=instance_id,
            problem_statement=PROBLEM_STATEMENT,
            workspace=dest,
            fail_to_pass=["tests/test_intervals.py::test_unsorted_input"],
            pass_to_pass=[
                "tests/test_intervals.py::test_empty",
                "tests/test_intervals.py::test_single",
                "tests/test_intervals.py::test_nested_interval_is_absorbed",
                "tests/test_intervals.py::test_sorted_overlapping",
            ],
            hidden_sources=[HIDDEN_SRC / "test_hidden.py"],
        )
