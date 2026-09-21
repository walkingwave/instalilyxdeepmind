"""The single agent primitive.

There is exactly one agent loop in this codebase. A "role" is a system prompt,
a subset of tools, and a model tier handed to this same loop -- not a subclass.
That is what makes repointing the whole system at a different task cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.budget import Budget
from core.llm import Provider, get_provider
from core.tools import Toolbox
from core.types import BudgetExceeded, Tier, Usage
from tracing.logger import NullTrace


@dataclass
class AgentResult:
    role: str
    text: str
    steps: int = 0
    usage: Usage = field(default_factory=Usage)
    stopped: str = "end_turn"  # end_turn | budget | error
    error: str = ""
    tool_calls_made: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.stopped == "end_turn"

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "text": self.text,
            "steps": self.steps,
            "usage": self.usage.as_dict(),
            "stopped": self.stopped,
            "error": self.error,
            "tool_calls_made": self.tool_calls_made,
        }


class Agent:
    """A role: system prompt + tool subset + tier, driven by a tool-calling loop."""

    def __init__(
        self,
        role: str,
        system: str,
        *,
        workspace: Path,
        tools: list[str] | None = None,
        tier: Tier = "strong",
        provider: Provider | None = None,
        budget: Budget | None = None,
        trace: Any = None,
        max_tokens: int = 16000,
    ) -> None:
        self.role = role
        self.system = system
        self.toolbox = Toolbox(workspace)
        self.tool_names = tools
        self.tier = tier
        self.provider = provider or get_provider()
        self.budget = budget or Budget()
        base_trace = trace or NullTrace()
        self.trace = base_trace.child(role) if hasattr(base_trace, "child") else base_trace
        self.max_tokens = max_tokens

    def run(self, prompt: str) -> AgentResult:
        specs = self.toolbox.specs(self.tool_names)
        session = self.provider.session(
            system=self.system, tools=specs, tier=self.tier, max_tokens=self.max_tokens
        )
        result = AgentResult(role=self.role, text="")
        self.trace.event(
            "agent_start",
            tier=self.tier,
            model=getattr(session, "model", ""),
            tools=[s.name for s in specs],
            prompt_chars=len(prompt),
        )

        try:
            response = session.send_text(prompt)
            while True:
                self.budget.charge(response.usage)
                result.steps += 1
                self.trace.event(
                    "llm_turn",
                    step=result.steps,
                    text=response.text,
                    tool_calls=[{"name": c.name, "args": c.args} for c in response.tool_calls],
                    usage=response.usage.as_dict(),
                    stop_reason=response.stop_reason,
                )

                if not response.wants_tools:
                    result.text = response.text
                    break

                results = []
                for call in response.tool_calls:
                    tool_result = self.toolbox.execute(call)
                    result.tool_calls_made.append(call.name)
                    self.trace.event(
                        "tool_result",
                        step=result.steps,
                        tool=call.name,
                        args=call.args,
                        is_error=tool_result.is_error,
                        output=tool_result.content,
                    )
                    results.append(tool_result)

                response = session.send_tool_results(results)

        except BudgetExceeded as exc:
            result.stopped = "budget"
            result.error = str(exc)
            self.trace.event("agent_budget_exceeded", error=str(exc))
        except Exception as exc:  # noqa: BLE001 - one bad role must not kill the run
            result.stopped = "error"
            result.error = f"{type(exc).__name__}: {exc}"
            self.trace.event("agent_error", error=result.error)

        result.usage = session.total_usage
        self.trace.event("agent_end", **result.as_dict())
        return result
