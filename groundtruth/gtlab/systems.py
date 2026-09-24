"""Parse kit/briefs.md into a structured system spec. No network."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRIEFS = ROOT / "kit" / "briefs.md"
SYSTEM_IDS = [
    "epidemic", "market", "traffic", "power_grid", "supply_chain",
    "wildlife", "reservoir", "ad_auction", "social_contagion", "hospital_queue",
]

# Hard output bounds we are confident about from the briefs. (lo, hi); None = open.
HARD_BOUNDS = {
    "power_grid": {"renewable_share": (0.0, 1.0)},
    "reservoir": {"quality": (0.0, 1.0)},
    "ad_auction": {"win_rate": (0.0, 1.0)},
}


@dataclass
class SystemSpec:
    id: str
    brief: str
    observables: list[str]
    controls: list[str]
    bounds: dict[str, tuple[float, float]]
    recovery: dict[str, float]
    pulse: dict[str, float]
    hard_bounds: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)

    @property
    def m(self) -> int:
        return len(self.controls)

    @property
    def p(self) -> int:
        return len(self.observables)

    def lo(self):
        return [self.bounds[c][0] for c in self.controls]

    def hi(self):
        return [self.bounds[c][1] for c in self.controls]

    def output_bounds(self, name):
        """Every observable is non-negative; some also have an upper bound."""
        lo, hi = self.hard_bounds.get(name, (0.0, None))
        return (0.0 if lo is None else lo, hi)

    def context(self, documents=None, brief_obj=None) -> dict:
        """The context dict exactly as the scoring runner passes it (7 keys)."""
        return {
            "protocol": "local", "revision": "local", "family": self.id,
            "observables": list(self.observables),
            "intervention_bounds": {c: list(self.bounds[c]) for c in self.controls},
            "brief": brief_obj if brief_obj is not None else self.brief,
            "documents": documents or [],
        }

    def to_json(self):
        return asdict(self)


def _parse(text: str) -> dict[str, SystemSpec]:
    sections = re.split(r"^## ", text, flags=re.M)[1:]
    out = {}
    for sec in sections:
        name, body = sec.split("\n", 1)
        name = name.strip()
        if name not in SYSTEM_IDS:
            continue
        obs = re.search(r"Observables:\s*(.+?)\.\s*$", body, flags=re.M).group(1)
        observables = [o.strip() for o in obs.split(",")]
        bounds = {}
        for row in re.finditer(r"^\|\s*([a-z_]+)\s*\|\s*([-\d.]+)\s*\|\s*([-\d.]+)\s*\|", body, flags=re.M):
            bounds[row.group(1)] = (float(row.group(2)), float(row.group(3)))
        rec = json.loads(re.search(r"Reference recovery action:\s*`(\{.*?\})`", body).group(1))
        pul = json.loads(re.search(r"Reference pulse action:\s*`(\{.*?\})`", body).group(1))
        brief = body.split("Observables:")[0].strip()
        out[name] = SystemSpec(
            id=name, brief=brief, observables=observables, controls=list(bounds),
            bounds=bounds, recovery=rec, pulse=pul,
            hard_bounds=HARD_BOUNDS.get(name, {}),
        )
    missing = set(SYSTEM_IDS) - set(out)
    if missing:
        raise ValueError(f"briefs.md missing systems: {missing}")
    return out


@lru_cache(maxsize=1)
def load_all() -> dict[str, SystemSpec]:
    return _parse(BRIEFS.read_text())


def get(system_id: str) -> SystemSpec:
    return load_all()[system_id]


if __name__ == "__main__":
    for s in load_all().values():
        print(s.id, s.observables, s.controls)
