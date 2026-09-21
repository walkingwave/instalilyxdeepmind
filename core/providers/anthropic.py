"""Anthropic adapter.

Stateless provider: the full transcript is resent on every call, so the session
owns a local `messages` list. Assistant content blocks are appended back
*verbatim* -- thinking blocks in particular must round-trip unmodified or the
API rejects the turn.
"""

from __future__ import annotations

import os
from typing import Any

from core.llm import LLMSession, Provider
from core.types import LLMResponse, Tier, ToolCall, ToolResult, ToolSpec, Usage

# Models taking `thinking={"type": "adaptive"}` + `output_config.effort`.
# Older models (haiku-4-5, sonnet-4-5) reject both and are called plainly.
_ADAPTIVE_MODELS = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
)


def _is_adaptive(model: str) -> bool:
    return model.startswith(_ADAPTIVE_MODELS)


class AnthropicSession(LLMSession):
    def __init__(
        self,
        client: Any,
        *,
        model: str,
        system: str,
        tools: list[ToolSpec],
        max_tokens: int,
        effort: str,
    ) -> None:
        super().__init__()
        self._client = client
        self.model = model
        self._system = system
        self._max_tokens = max_tokens
        self._effort = effort
        self._messages: list[dict[str, Any]] = []
        self._tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]

    def send_text(self, text: str) -> LLMResponse:
        self._messages.append({"role": "user", "content": text})
        return self._complete()

    def send_tool_results(self, results: list[ToolResult]) -> LLMResponse:
        # Every tool_use id from the prior turn needs exactly one tool_result,
        # and they all go in a single user message.
        blocks = [
            {
                "type": "tool_result",
                "tool_use_id": r.call_id,
                "content": r.content,
                "is_error": r.is_error,
            }
            for r in results
        ]
        self._messages.append({"role": "user", "content": blocks})
        return self._complete()

    def _request_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "messages": self._messages,
        }
        if self._system:
            # A cache breakpoint on the system prompt: it is byte-identical
            # across every turn of every task, so it is the natural prefix.
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": self._system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        if self._tools:
            kwargs["tools"] = self._tools
        if _is_adaptive(self.model):
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": self._effort}
        return kwargs

    def _complete(self) -> LLMResponse:
        raw = self._client.messages.create(**self._request_kwargs())

        # Append verbatim so thinking blocks round-trip unmodified.
        self._messages.append({"role": "assistant", "content": raw.content})

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in raw.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, args=dict(block.input)))

        usage = Usage(
            input_tokens=getattr(raw.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(raw.usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(raw.usage, "cache_read_input_tokens", 0) or 0,
        )
        return self._record(
            LLMResponse(
                text="\n".join(text_parts).strip(),
                tool_calls=calls,
                stop_reason=raw.stop_reason or "end_turn",
                usage=usage,
                model=self.model,
            )
        )


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - environment problem
            raise RuntimeError(
                "anthropic SDK not installed. Run: pip install -e ."
            ) from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example)")
        self._client = anthropic.Anthropic()

    def session(
        self,
        *,
        system: str,
        tools: list[ToolSpec],
        tier: Tier = "strong",
        max_tokens: int = 16000,
    ) -> LLMSession:
        model = self.model_for(tier)
        return AnthropicSession(
            self._client,
            model=model,
            system=system,
            tools=tools,
            max_tokens=max_tokens,
            effort="high" if tier == "strong" else "low",
        )
