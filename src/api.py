"""Single public entry point used by UI, evaluators, and tests.

Dispatches to baseline or crew while keeping the contract identical
(both return ``AnswerResult``).  Nothing else should call
``src.baseline.answer`` or ``src.crew.answer`` directly — go through here.
"""

from __future__ import annotations

from typing import Literal

from src import baseline, crew
from src.llm import AnswerResult

Architecture = Literal["baseline", "crew"]


def answer(
    query: str,
    history: list[dict] | None = None,
    arch: Architecture = "crew",
) -> AnswerResult:
    """Run one user turn through the chosen architecture."""
    if arch == "baseline":
        return baseline.answer(query, history)
    if arch == "crew":
        return crew.answer(query, history)
    raise ValueError(f"Unknown architecture: {arch!r}")


if __name__ == "__main__":
    # CLI: uv run python -m src.api <arch> "<query>"
    import sys

    arch: Architecture = sys.argv[1] if len(sys.argv) > 1 else "crew"  # type: ignore[assignment]
    query = sys.argv[2] if len(sys.argv) > 2 else "Скільки витратив на каву минулого тижня?"

    res = answer(query, arch=arch)
    print(f"\n=== {res.architecture.upper()} ===")
    print(res.answer)
    print(f"\nlatency={res.latency_ms}ms  tokens={res.total_tokens}  "
          f"cost=${res.total_cost_usd:.5f}  tools_used={len(res.tool_calls)}")
    if res.architecture == "crew":
        print(f"intent={res.metadata.get('intent')}")
        print(f"cost_by_agent={res.cost_by_agent()}")
