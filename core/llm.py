"""Provider-agnostic LLM interface.

Call sites ask for a *tier* ("cheap" / "strong"), never a model name. Swapping
providers, or downgrading under budget pressure, is then a config change rather
than a code change.

The unit of interaction is an `LLMSession`: one agent's conversation with one
model. Providers differ in how they carry history -- Anthropic resends the full
transcript on every call, Gemini's Interactions API chains server-side via
`previous_interaction_id` -- so history management lives behind this interface
and the agent loop stays identical for both.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from core.types import LLMResponse, Tier, ToolResult, ToolSpec, Usage

# Price per 1M tokens, (input, output). Keep in sync with provider pricing pages;
# these drive the budget ceiling, so being roughly right matters more than exact.
PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # Gemini (introductory rates)
    "gemini-3.1-pro": (2.00, 12.00),
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.5-flash-lite": (0.10, 0.40),
}

DEFAULT_MODELS: dict[str, dict[Tier, str]] = {
    "anthropic": {"strong": "claude-opus-5", "cheap": "claude-haiku-4-5"},
    "gemini": {"strong": "gemini-3.1-pro", "cheap": "gemini-3.8-flash"},
}


def estimate_cost(model: str, usage: Usage) -> float:
    """Dollar cost of a turn. Unknown models cost 0 rather than crashing a run."""
    rates = PRICING.get(model)
    if rates is None:
        return 0.0
    inp, out = rates
    return (usage.input_tokens * inp + usage.output_tokens * out) / 1_000_000


class LLMSession(ABC):
    """One agent's conversation with one model.

    Implementations own their own history representation. The agent loop only
    ever calls `send_text` once, then `send_tool_results` in a loop.
    """

    model: str

    def __init__(self) -> None:
        self.total_usage = Usage()

    @abstractmethod
    def send_text(self, text: str) -> LLMResponse:
        """Send a user-role text turn."""

    @abstractmethod
    def send_tool_results(self, results: list[ToolResult]) -> LLMResponse:
        """Return results for every tool call from the previous turn, in one turn."""

    def _record(self, response: LLMResponse) -> LLMResponse:
        response.usage.cost_usd = estimate_cost(response.model, response.usage)
        self.total_usage = self.total_usage + response.usage
        return response


class Provider(ABC):
    name: str

    @abstractmethod
    def session(
        self,
        *,
        system: str,
        tools: list[ToolSpec],
        tier: Tier = "strong",
        max_tokens: int = 16000,
    ) -> LLMSession:
        """Open a new conversation."""

    def model_for(self, tier: Tier) -> str:
        env_key = f"{self.name.upper()}_MODEL_{tier.upper()}"
        return os.environ.get(env_key) or DEFAULT_MODELS[self.name][tier]


def get_provider(name: str | None = None) -> Provider:
    """Resolve a provider by name, defaulting to $LLM_PROVIDER then anthropic."""
    name = (name or os.environ.get("LLM_PROVIDER") or "anthropic").lower()
    if name == "anthropic":
        from core.providers.anthropic import AnthropicProvider

        return AnthropicProvider()
    if name == "gemini":
        from core.providers.gemini import GeminiProvider

        return GeminiProvider()
    raise ValueError(f"unknown provider {name!r}; expected 'anthropic' or 'gemini'")
