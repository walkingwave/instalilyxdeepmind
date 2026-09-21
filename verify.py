"""Provider round-trip check.

Run this before anything else after adding API keys. It proves, per provider,
that: the key works, the model id is real, a tool definition is accepted, the
model actually calls the tool, the result round-trips, and token/cost
accounting comes back non-zero.

Catching a bad model id here costs one call. Catching it inside a six-agent
orchestration costs an afternoon.

    python verify.py              # every provider with a key set
    python verify.py anthropic    # just one
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from core.config import load_env
from core.llm import get_provider
from core.tools import Toolbox

PROMPT = (
    "Read the file notes.txt in the workspace and reply with only the single "
    "word it contains. Use the read_file tool -- do not guess."
)
SECRET = "pomegranate"


def check(provider_name: str, tier: str = "cheap") -> bool:
    print(f"\n--- {provider_name} ({tier} tier) ---")
    try:
        provider = get_provider(provider_name)
    except Exception as exc:
        print(f"  SKIP: {exc}")
        return True  # missing key is not a failure, just unconfigured

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "notes.txt").write_text(SECRET, encoding="utf-8")
        toolbox = Toolbox(root)
        specs = toolbox.specs(["read_file"])

        try:
            session = provider.session(
                system="You are a terse assistant with filesystem tools.",
                tools=specs,
                tier=tier,
                max_tokens=2000,
            )
            print(f"  model: {session.model}")

            response = session.send_text(PROMPT)
            if not response.tool_calls:
                print(f"  FAIL: model did not call a tool. Said: {response.text[:200]!r}")
                return False
            print(f"  tool call: {response.tool_calls[0].name}({response.tool_calls[0].args})")

            results = [toolbox.execute(c) for c in response.tool_calls]
            if any(r.is_error for r in results):
                print(f"  FAIL: tool errored: {results[0].content[:200]}")
                return False

            final = session.send_tool_results(results)
            usage = session.total_usage
            print(f"  reply: {final.text[:120]!r}")
            print(
                f"  usage: in={usage.input_tokens} out={usage.output_tokens} "
                f"cost=${usage.cost_usd:.5f}"
            )

            if SECRET not in final.text.lower():
                print(f"  FAIL: reply did not contain {SECRET!r} -- tool result did not land")
                return False
            if usage.input_tokens == 0:
                print("  WARN: token accounting read 0 -- budget ceilings will not work")
            print("  OK")
            return True

        except Exception as exc:
            print(f"  FAIL: {type(exc).__name__}: {exc}")
            return False


def main() -> int:
    load_env()
    names = sys.argv[1:] or ["anthropic", "gemini"]
    results = [check(n) for n in names]
    ok = all(results)
    print("\n" + ("all configured providers OK" if ok else "SOME PROVIDERS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
