"""Append-only JSONL write-ahead ledger (one per system).

Rows (every row also gets "ts"):
  {"t":"meta","system":..,"mock":bool,"version":1}                       first row, written once
  {"t":"reset","exp":..,"run_idx":i,"run_id":..,"req":..,"obs":{..},"raw":{..}}
  {"t":"intent","exp":..,"run_idx":i,"run_id":..,"tick":k,"req":R,"action":{..}}   BEFORE sending
  {"t":"step","exp":..,"run_idx":i,"run_id":..,"tick":k,"req":R,"action":{..},"obs":{..},"raw":{..}}
  {"t":"budget","remaining":n,"expected":n,"where":..}                     free reconciliation reads
  {"t":"external_spend","n":k,"note":..}                                   steps spent outside this tool

Real data lives in data/<sys>/, mock data in data_mock/<sys>/. The meta row pins which one a
ledger is, and opening it the other way raises. Every append is flushed and fsynced.
"""
from __future__ import annotations

try:
    import fcntl
except ImportError:          # Windows
    fcntl = None
    import msvcrt
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NS = uuid.UUID("6f1d3b8e-2b0c-5d7a-9a41-47a0c1e2d026")   # fixed namespace, never change


class LedgerError(RuntimeError):
    pass


class LockedError(LedgerError):
    pass


def data_dir(system_id: str, mock: bool, base: Path | str | None = None) -> Path:
    base = Path(base) if base is not None else Path(os.environ.get("GT_DATA_ROOT", ROOT))
    return base / ("data_mock" if mock else "data") / system_id


def req_id(system_id: str, exp_id: str, run_idx: int, tick) -> str:
    """Deterministic idempotency key. tick=-1 is the reset; shop resets use 'reset<k>'."""
    return str(uuid.uuid5(NS, f"{system_id}/{exp_id}/{run_idx}/{tick}"))


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


class Ledger:
    def __init__(self, directory: Path | str, system_id: str, mock: bool):
        self.dir = Path(directory)
        self.system = system_id
        self.mock = bool(mock)
        kind = self.dir.parent.name
        if kind in ("data", "data_mock") and (kind == "data_mock") != self.mock:
            raise LedgerError(f"refusing to open a {'mock' if mock else 'real'} ledger in {self.dir}")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "ledger.jsonl"
        self.resets_path = self.dir / "resets.jsonl"
        self._lock_fh = None
        self.rows: list[dict] = []
        self._load()
        if not self.rows:
            self.append({"t": "meta", "system": system_id, "mock": self.mock, "version": 1})
        else:
            meta = self.rows[0]
            if meta.get("t") != "meta" or meta.get("system") != system_id or bool(meta.get("mock")) != self.mock:
                raise LedgerError(f"ledger meta mismatch in {self.path}: {meta}")

    # ---------------------------------------------------------------- io
    def _load(self):
        self.rows = []
        self._index = {}          # (exp, run_idx) -> {"reset": row, "steps": {tick: row}, "intents": {tick: row}}
        self._reqs = set()
        if not self.path.exists():
            return
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            # torn last line from a crash: terminate it so the next append starts clean
            with open(self.path, "ab") as fh:
                fh.write(b"\n")
                fh.flush()
                os.fsync(fh.fileno())
        for line in raw.decode("utf-8", "replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue          # torn line; the intent before it makes resume safe
            self._ingest(row)

    def _ingest(self, row):
        self.rows.append(row)
        t = row.get("t")
        if t in ("reset", "intent", "step"):
            e = self._index.setdefault((row["exp"], int(row.get("run_idx", 0))),
                                       {"reset": None, "steps": {}, "intents": {}})
            if t == "reset":
                e["reset"] = row
            elif t == "intent":
                e["intents"][int(row["tick"])] = row
            else:
                e["steps"].setdefault(int(row["tick"]), row)   # first one wins; dupes ignored
                if row.get("req") in self._reqs:
                    return
                self._reqs.add(row.get("req"))

    @staticmethod
    def _write_line(path: Path, row: dict):
        line = json.dumps(_jsonable(row), separators=(",", ":"), allow_nan=False) + "\n"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

    def append(self, row: dict) -> dict:
        row = dict(row)
        row.setdefault("ts", time.time())
        self._write_line(self.path, row)
        self._ingest(_jsonable(row))
        return row

    def log_reset_candidate(self, row: dict):
        """Free shop resets go to resets.jsonl (initial-observation pool + noise estimate)."""
        row = dict(row)
        row.setdefault("ts", time.time())
        row.setdefault("system", self.system)
        self._write_line(self.resets_path, row)

    def reset_pool(self) -> list[dict]:
        if not self.resets_path.exists():
            return []
        out = []
        for line in self.resets_path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return out

    # ---------------------------------------------------------------- lock
    def lock(self):
        fh = open(self.dir / ".lock", "a+")
        try:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        except (BlockingIOError, OSError):
            fh.close()
            raise LockedError(f"another collector holds {self.dir / '.lock'}")
        self._lock_fh = fh
        return self

    def unlock(self):
        if self._lock_fh is not None:
            if fcntl is not None:
                fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_UN)
            else:
                try:
                    self._lock_fh.seek(0)
                    msvcrt.locking(self._lock_fh.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            self._lock_fh.close()
            self._lock_fh = None

    def __enter__(self):
        return self.lock()

    def __exit__(self, *a):
        self.unlock()

    # ---------------------------------------------------------------- queries
    def run_state(self, exp: str, run_idx: int = 0) -> dict:
        """{"reset": row|None, "done": sorted ticks, "next": first missing tick, "pending": intent|None}"""
        e = self._index.get((exp, run_idx))
        if e is None:
            return {"reset": None, "done": [], "next": 0, "pending": None}
        done = sorted(e["steps"])
        nxt = 0
        while nxt in e["steps"]:
            nxt += 1
        pend = e["intents"].get(nxt)
        if pend is not None and e["reset"] is not None and pend.get("run_id") != e["reset"]["run_id"]:
            pend = None
        return {"reset": e["reset"], "done": done, "next": nxt, "pending": pend}

    def pending_intents(self) -> list[dict]:
        out = []
        for (exp, i), e in self._index.items():
            for k, row in e["intents"].items():
                if k not in e["steps"]:
                    out.append(row)
        return out

    def steps_charged(self, prefix: str | None = None) -> int:
        """Distinct completed paid steps (+ external spend when prefix is None)."""
        n = 0
        for (exp, _), e in self._index.items():
            if prefix is None or exp.startswith(prefix):
                n += len(e["steps"])
        if prefix is None:
            n += sum(int(r.get("n", 0)) for r in self.rows if r.get("t") == "external_spend")
        return n

    def phase_charged(self, phase: str) -> int:
        return self.steps_charged(prefix=phase + ".")

    def last_budget(self) -> dict | None:
        for r in reversed(self.rows):
            if r.get("t") == "budget":
                return r
        return None

    def bought_initials(self) -> list[dict]:
        return [e["reset"]["obs"] for e in self._index.values() if e["reset"] is not None]

    def backup(self, tag: str = "") -> Path:
        bdir = self.dir / "backup"
        bdir.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        dst = bdir / f"ledger.{ts}{('.' + tag) if tag else ''}.jsonl"
        shutil.copy2(self.path, dst)
        if self.resets_path.exists():
            shutil.copy2(self.resets_path, bdir / f"resets.{ts}.jsonl")
        return dst


def load_runs(spec, ledger: Ledger):
    """Ledger -> list[gtlab.data.Run]. Mock ledgers also get Ytrue from raw['mock_truth']."""
    from gtlab.data import runs_from_ledger
    rows = [r for r in ledger.rows if r.get("t") in ("reset", "step")]
    runs = runs_from_ledger(spec, rows)
    if ledger.mock:
        truth = {}
        for r in rows:
            if r.get("t") == "step":
                mt = (r.get("raw") or {}).get("mock_truth")
                truth.setdefault(r["run_id"], {}).setdefault(int(r["tick"]), mt)
        for run in runs:
            d = truth.get(run.tags.get("run_id"), {})
            ticks = sorted(d)
            if ticks and all(d[k] is not None for k in ticks) and len(ticks) == run.T:
                run.Ytrue = np.array([[d[k][o] for o in spec.observables] for k in ticks], float)
    exp_run = {e["reset"]["run_id"]: (exp, i) for (exp, i), e in ledger._index.items() if e["reset"]}
    for run in runs:
        exp, i = exp_run.get(run.tags.get("run_id"), (run.exp, 0))
        run.tags["run_idx"] = i
        run.tags["phase"] = exp.split(".", 1)[0]
    return runs
