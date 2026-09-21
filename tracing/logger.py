"""Append-only JSONL trace of everything an agent does.

Written from the first LLM call onward, because a six-agent system is not
debuggable any other way. The same file is the cost ledger and the Demo Day
artifact (see tracing/viewer.py).
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

RUNS_DIR = Path(os.environ.get("TRACE_DIR", "tracing/runs"))


class TraceLogger:
    def __init__(self, run_id: str | None = None, *, task: str = "") -> None:
        self.run_id = run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        self.path = RUNS_DIR / f"{self.run_id}.jsonl"
        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        self.event("run_start", task=task, run_id=self.run_id)

    def event(self, kind: str, **fields: Any) -> None:
        record = {
            "t": round(time.monotonic() - self._t0, 3),
            "kind": kind,
            **fields,
        }
        line = json.dumps(record, default=str, ensure_ascii=False)
        # Agents run in parallel; one lock keeps lines from interleaving.
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def child(self, role: str) -> "RoleTrace":
        return RoleTrace(self, role)


class RoleTrace:
    """A view of the trace tagged with which role is speaking."""

    def __init__(self, logger: TraceLogger, role: str) -> None:
        self._logger = logger
        self.role = role

    def event(self, kind: str, **fields: Any) -> None:
        self._logger.event(kind, role=self.role, **fields)


class NullTrace:
    """No-op trace, for unit tests and one-off scripts."""

    role = "null"

    def event(self, kind: str, **fields: Any) -> None:  # noqa: D102
        return

    def child(self, role: str) -> "NullTrace":
        return self
