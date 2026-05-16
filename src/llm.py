"""Shared LLM infrastructure: OpenAI client, model config, cost accounting,
and the common ``AnswerResult`` contract returned by both architectures.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

# Load .env from the homework root regardless of where the process was started.
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)

# Use the Langfuse-patched OpenAI client when Langfuse keys are present so
# every Chat Completions call is auto-traced.  Falls back to plain openai
# when keys are missing, so the project still runs without Langfuse.
if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
    from langfuse.openai import OpenAI  # noqa: E402  (auto-instrumented)
    LANGFUSE_ENABLED = True
else:
    from openai import OpenAI  # noqa: E402
    LANGFUSE_ENABLED = False

# Public OpenAI list prices, USD per 1M tokens, as published on
# https://openai.com/api/pricing/. Update if pricing changes.
MODEL_COSTS_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o":        (2.50, 10.00),
    "gpt-4o-mini":   (0.15,  0.60),
    "gpt-4.1":       (2.00,  8.00),
    "gpt-4.1-mini":  (0.40,  1.60),
}

MODEL_SMART = os.getenv("OPENAI_MODEL_SMART", "gpt-4o")
MODEL_CHEAP = os.getenv("OPENAI_MODEL_CHEAP", "gpt-4o-mini")


_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Singleton OpenAI client built from ``OPENAI_API_KEY``."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY missing. Copy .env.example to .env and fill it in."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def compute_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD using the public per-1M-token list price."""
    if model not in MODEL_COSTS_PER_1M:
        return 0.0
    in_cost, out_cost = MODEL_COSTS_PER_1M[model]
    return (input_tokens * in_cost + output_tokens * out_cost) / 1_000_000


# ───────────────────────────────────────── shared result types ──────────────


@dataclass
class ToolCallTrace:
    tool: str
    args: dict
    result: dict
    latency_ms: int


@dataclass
class ModelCall:
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    agent: str | None = None   # e.g. "router", "data_analyst" — set by crew
    cost_usd: float = 0.0


@dataclass
class AnswerResult:
    """Unified result returned by both ``baseline.answer`` and ``crew.answer``."""

    architecture: Literal["baseline", "crew"]
    answer: str
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    model_calls: list[ModelCall] = field(default_factory=list)
    latency_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_input_tokens(self) -> int:
        return sum(m.input_tokens for m in self.model_calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(m.output_tokens for m in self.model_calls)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost_usd(self) -> float:
        return sum(m.cost_usd for m in self.model_calls)

    def cost_by_agent(self) -> dict[str, float]:
        """For crew traces: USD per agent."""
        out: dict[str, float] = {}
        for m in self.model_calls:
            key = m.agent or "main"
            out[key] = out.get(key, 0.0) + m.cost_usd
        return out

    def tokens_by_agent(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for m in self.model_calls:
            key = m.agent or "main"
            out[key] = out.get(key, 0) + m.input_tokens + m.output_tokens
        return out
