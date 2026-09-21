"""Task interface.

Every task -- the toy fixture, SWE-bench-Lite, and whatever is revealed at
kickoff -- reduces to a prepared workspace plus three test sets:

  fail_to_pass  the reported failure(s); must go green
  pass_to_pass  the pre-existing suite; must stay green (regression guard)
  hidden        held out, copied in only at grading time

Keeping `hidden` physically out of the workspace during the run is the point:
it is the local stand-in for "tests you have not seen".
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class TaskInstance:
    id: str
    problem_statement: str
    workspace: Path
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    hidden_sources: list[Path] = field(default_factory=list)
    hidden_dest: str = "tests"

    def install_hidden_tests(self) -> list[str]:
        """Copy held-out tests in. Called by the grader, never during the run."""
        dest_dir = self.workspace / self.hidden_dest
        dest_dir.mkdir(parents=True, exist_ok=True)
        installed: list[str] = []
        for src in self.hidden_sources:
            dest = dest_dir / src.name
            shutil.copy2(src, dest)
            installed.append(str(dest.relative_to(self.workspace)).replace("\\", "/"))
        return installed

    def remove_hidden_tests(self, installed: list[str]) -> None:
        for rel in installed:
            target = self.workspace / rel
            if target.exists():
                target.unlink()


class Task(Protocol):
    name: str

    def instance_ids(self, limit: int | None = None) -> list[str]:
        """Which instances this task can produce."""

    def prepare(self, instance_id: str, dest: Path) -> TaskInstance:
        """Materialize a fresh workspace for one instance."""
