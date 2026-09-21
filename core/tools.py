"""The toolbox every role draws from.

All file and shell access is confined to a workspace root: paths coming back
from a model are untrusted, so each one is resolved to its canonical form and
rejected if it escapes the root.

Tool descriptions state *when* to call the tool, not only what it does --
trigger conditions are the part that actually moves call rate.
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core.types import ToolCall, ToolResult, ToolSpec

MAX_OUTPUT_CHARS = 8000
DEFAULT_TIMEOUT = 120


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - limit
    return f"{text[:half]}\n\n... [{omitted} chars omitted] ...\n\n{text[-half:]}"


class PathEscape(ValueError):
    """Raised when a model-supplied path resolves outside the workspace."""


@dataclass
class Toolbox:
    root: Path
    timeout: int = DEFAULT_TIMEOUT

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()

    # ---- path safety -------------------------------------------------

    def _resolve(self, rel: str) -> Path:
        candidate = (self.root / rel).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise PathEscape(f"path {rel!r} resolves outside the workspace")
        return candidate

    def _rel(self, p: Path) -> str:
        return str(p.relative_to(self.root)).replace("\\", "/")

    # ---- tool implementations ---------------------------------------

    def read_file(self, path: str, start: int | None = None, end: int | None = None) -> str:
        target = self._resolve(path)
        if not target.is_file():
            raise FileNotFoundError(f"{path} is not a file")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        lo = max((start or 1) - 1, 0)
        hi = min(end or len(lines), len(lines))
        width = len(str(hi))
        body = "\n".join(f"{i + 1:>{width}}\t{lines[i]}" for i in range(lo, hi))
        return _truncate(body) or "(empty file)"

    def write_file(self, path: str, content: str) -> str:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {self._rel(target)}"

    def edit_file(self, path: str, old: str, new: str) -> str:
        target = self._resolve(path)
        text = target.read_text(encoding="utf-8")
        hits = text.count(old)
        if hits == 0:
            raise ValueError(
                f"`old` not found in {path}. Read the file again and copy the exact text, "
                "including indentation."
            )
        if hits > 1:
            raise ValueError(
                f"`old` matches {hits} times in {path}; include surrounding lines "
                "to make it unique."
            )
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
        return f"edited {self._rel(target)}"

    def glob(self, pattern: str, limit: int = 200) -> str:
        matches = [
            self._rel(p)
            for p in sorted(self.root.rglob("*"))
            if p.is_file() and fnmatch.fnmatch(self._rel(p), pattern)
        ]
        if not matches:
            return f"no files match {pattern!r}"
        return "\n".join(matches[:limit])

    def grep(self, pattern: str, path_glob: str = "**/*.py", limit: int = 100) -> str:
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc
        out: list[str] = []
        for p in sorted(self.root.rglob("*")):
            if not p.is_file():
                continue
            rel = self._rel(p)
            if not fnmatch.fnmatch(rel, path_glob):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for n, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    out.append(f"{rel}:{n}: {line.strip()}")
                    if len(out) >= limit:
                        return "\n".join(out) + f"\n... (stopped at {limit} matches)"
        return "\n".join(out) if out else f"no matches for {pattern!r}"

    def run_bash(self, command: str) -> str:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        body = (proc.stdout or "") + (proc.stderr or "")
        return _truncate(f"exit code: {proc.returncode}\n{body}")

    def run_tests(self, targets: str = "", extra_args: str = "") -> str:
        cmd = f"python -m pytest -q --no-header {extra_args} {targets}".strip()
        return self.run_bash(cmd)

    # ---- dispatch ----------------------------------------------------

    def specs(self, names: list[str] | None = None) -> list[ToolSpec]:
        chosen = names or list(_SPECS)
        return [_SPECS[n] for n in chosen]

    def execute(self, call: ToolCall) -> ToolResult:
        fn: Callable[..., str] | None = getattr(self, call.name, None)
        if fn is None or call.name not in _SPECS:
            return ToolResult(call.id, call.name, f"unknown tool {call.name!r}", is_error=True)
        try:
            out = fn(**call.args)
        except subprocess.TimeoutExpired:
            return ToolResult(
                call.id, call.name, f"command timed out after {self.timeout}s", is_error=True
            )
        except Exception as exc:  # surfaced to the model so it can adapt
            return ToolResult(call.id, call.name, f"{type(exc).__name__}: {exc}", is_error=True)
        return ToolResult(call.id, call.name, out)


def _obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required}


_SPECS: dict[str, ToolSpec] = {
    "read_file": ToolSpec(
        "read_file",
        "Read a file from the workspace, with line numbers. Call this before editing any "
        "file so your `old` string matches exactly, and whenever a traceback names a file "
        "you have not yet looked at. Use start/end to page through large files.",
        _obj(
            {
                "path": {"type": "string", "description": "Path relative to the workspace root"},
                "start": {"type": "integer", "description": "First line (1-indexed)"},
                "end": {"type": "integer", "description": "Last line (inclusive)"},
            },
            ["path"],
        ),
    ),
    "write_file": ToolSpec(
        "write_file",
        "Create a new file or fully replace an existing one. Use this for new test files. "
        "For changes to existing source, prefer edit_file -- a full rewrite risks dropping "
        "code you did not intend to touch.",
        _obj(
            {
                "path": {"type": "string"},
                "content": {"type": "string", "description": "Complete file contents"},
            },
            ["path", "content"],
        ),
    ),
    "edit_file": ToolSpec(
        "edit_file",
        "Replace one exact occurrence of `old` with `new` in a file. This is the default way "
        "to change source. `old` must appear exactly once -- include surrounding lines if the "
        "snippet would otherwise be ambiguous.",
        _obj(
            {
                "path": {"type": "string"},
                "old": {
                    "type": "string",
                    "description": "Exact existing text, including indentation",
                },
                "new": {"type": "string", "description": "Replacement text"},
            },
            ["path", "old", "new"],
        ),
    ),
    "glob": ToolSpec(
        "glob",
        "List workspace files matching a glob such as **/test_*.py. Call this first when "
        "orienting in an unfamiliar repository, before grepping.",
        _obj(
            {
                "pattern": {"type": "string", "description": "Glob relative to workspace root"},
                "limit": {"type": "integer"},
            },
            ["pattern"],
        ),
    ),
    "grep": ToolSpec(
        "grep",
        "Regex search file contents, returning path:line: match. Call this to find where a "
        "symbol is defined or used when a traceback or test name gives you an identifier but "
        "not a location.",
        _obj(
            {
                "pattern": {"type": "string", "description": "Python regular expression"},
                "path_glob": {
                    "type": "string",
                    "description": "Restrict to files matching this glob",
                },
                "limit": {"type": "integer"},
            },
            ["pattern"],
        ),
    ),
    "run_bash": ToolSpec(
        "run_bash",
        "Run a shell command in the workspace root and return exit code plus combined output. "
        "Use it for git, for installing a missing dependency, or for anything the other tools "
        "do not cover. Prefer run_tests for running the test suite.",
        _obj({"command": {"type": "string"}}, ["command"]),
    ),
    "run_tests": ToolSpec(
        "run_tests",
        "Run pytest and return the result. Call this to reproduce the reported failure before "
        "changing anything, and again after every edit. A fix you have not run is not a fix. "
        "Leave `targets` empty to run the whole suite and catch regressions.",
        _obj(
            {
                "targets": {
                    "type": "string",
                    "description": (
                        "Space-separated pytest node ids or paths; empty runs everything"
                    ),
                },
                "extra_args": {"type": "string", "description": "Additional pytest flags"},
            },
            [],
        ),
    ),
}
