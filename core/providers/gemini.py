"""Gemini adapter (Interactions API).

Stateful provider: history lives server-side and is chained via
`previous_interaction_id`, so this session sends only the *new* input each turn
rather than resending a transcript.

Parameter and response shapes were read off google-genai 2.24.0:
`interactions.create` takes input/model/tools/system_instruction/
previous_interaction_id, with max_output_tokens and thinking_level nested in
`generation_config`; responses carry output_text, steps, usage, id. Token
counters are still read defensively so an SDK bump cannot zero out the budget
ledger silently.
"""

from __future__ import annotations

import json
import os
from typing import Any

from core.llm import LLMSession, Provider
from core.types import LLMResponse, Tier, ToolCall, ToolResult, ToolSpec, Usage


def _first_attr(obj: Any, names: tuple[str, ...], default: int = 0) -> int:
    for n in names:
        v = getattr(obj, n, None)
        if isinstance(v, int):
            return v
    return default


def _as_args(raw: Any) -> dict[str, Any]:
    """Tool arguments arrive as either a mapping or a JSON string."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"value": raw}
    if raw is None:
        return {}
    return dict(raw)


class GeminiSession(LLMSession):
    def __init__(
        self,
        client: Any,
        *,
        model: str,
        system: str,
        tools: list[ToolSpec],
        max_tokens: int,
        thinking_level: str,
    ) -> None:
        super().__init__()
        self._client = client
        self.model = model
        self._system = system
        self._max_tokens = max_tokens
        self._thinking_level = thinking_level
        self._previous_id: str | None = None
        self._tools = [
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            }
            for t in tools
        ]

    def send_text(self, text: str) -> LLMResponse:
        return self._create(text)

    def send_tool_results(self, results: list[ToolResult]) -> LLMResponse:
        items = [
            {
                "type": "function_result",
                "name": r.name,
                "call_id": r.call_id,
                "result": [{"type": "text", "text": r.content}],
                "is_error": r.is_error,
            }
            for r in results
        ]
        return self._create(items)

    def _create(self, payload: Any) -> LLMResponse:
        generation_config: dict[str, Any] = {"max_output_tokens": self._max_tokens}
        if self._thinking_level:
            generation_config["thinking_level"] = self._thinking_level

        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": payload,
            "generation_config": generation_config,
        }
        if self._tools:
            kwargs["tools"] = self._tools
        if self._previous_id:
            # Chained: history is server-side, so don't resend it.
            kwargs["previous_interaction_id"] = self._previous_id
        elif self._system:
            # System instruction belongs on the first turn of the chain.
            kwargs["system_instruction"] = self._system
        raw = self._client.interactions.create(**kwargs)
        self._previous_id = getattr(raw, "id", None)

        calls: list[ToolCall] = []
        for step in getattr(raw, "steps", None) or []:
            if getattr(step, "type", None) == "function_call":
                calls.append(
                    ToolCall(
                        id=getattr(step, "id", "") or getattr(step, "call_id", ""),
                        name=step.name,
                        args=_as_args(getattr(step, "arguments", None)),
                    )
                )

        raw_usage = getattr(raw, "usage", None)
        usage = Usage(
            input_tokens=_first_attr(
                raw_usage, ("total_input_tokens", "input_tokens", "prompt_token_count")
            ),
            output_tokens=_first_attr(
                raw_usage, ("total_output_tokens", "output_tokens", "candidates_token_count")
            ),
            cache_read_tokens=_first_attr(
                raw_usage, ("total_cached_tokens", "cached_content_token_count")
            ),
        )

        return self._record(
            LLMResponse(
                text=(getattr(raw, "output_text", "") or "").strip(),
                tool_calls=calls,
                stop_reason="tool_use" if calls else "end_turn",
                usage=usage,
                model=self.model,
            )
        )


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self) -> None:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - environment problem
            raise RuntimeError(
                "google-genai SDK not installed. Run: pip install -e ."
            ) from exc
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise RuntimeError("GEMINI_API_KEY is not set (see .env.example)")
        self._client = genai.Client()

    def session(
        self,
        *,
        system: str,
        tools: list[ToolSpec],
        tier: Tier = "strong",
        max_tokens: int = 16000,
    ) -> LLMSession:
        return GeminiSession(
            self._client,
            model=self.model_for(tier),
            system=system,
            tools=tools,
            max_tokens=max_tokens,
            thinking_level="high" if tier == "strong" else "low",
        )
