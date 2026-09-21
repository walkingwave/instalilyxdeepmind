"""Per-run ceilings on tokens, dollars, wall-clock, and agent steps.

Seven days, solo, finite credits: every agent run is bounded, and hitting a
ceiling raises rather than silently producing a truncated result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from core.types import BudgetExceeded, Usage


@dataclass
class Budget:
    max_usd: float = 2.00
    max_steps: int = 40
    max_seconds: float = 900.0

    usage: Usage = field(default_factory=Usage)
    steps: int = 0
    _t0: float = field(default_factory=time.monotonic)

    def charge(self, usage: Usage) -> None:
        self.usage = self.usage + usage
        self.steps += 1
        self.check()

    def check(self) -> None:
        if self.usage.cost_usd > self.max_usd:
            raise BudgetExceeded(
                f"cost ${self.usage.cost_usd:.3f} exceeded ceiling ${self.max_usd:.2f}"
            )
        if self.steps > self.max_steps:
            raise BudgetExceeded(f"step count {self.steps} exceeded ceiling {self.max_steps}")
        if self.elapsed > self.max_seconds:
            raise BudgetExceeded(
                f"elapsed {self.elapsed:.0f}s exceeded ceiling {self.max_seconds:.0f}s"
            )

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    def as_dict(self) -> dict:
        return {
            **self.usage.as_dict(),
            "steps": self.steps,
            "elapsed_s": round(self.elapsed, 1),
        }
