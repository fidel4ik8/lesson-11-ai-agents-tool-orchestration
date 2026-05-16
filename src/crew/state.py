"""Shared LangGraph state for the multi-agent crew."""

from __future__ import annotations

from operator import add
from typing import Annotated, Literal, TypedDict

from src.llm import ModelCall, ToolCallTrace

Intent = Literal["stats", "advice", "multi_step", "fraud", "out_of_scope"]


class CrewState(TypedDict, total=False):
    # Inputs
    query: str
    history: list[dict]

    # Router output
    intent: Intent
    intent_rationale: str

    # DataAnalyst output (consumed by Advisor)
    findings: str

    # Final answer (set by the terminal node for each branch)
    answer: str

    # Accumulated across all nodes (Annotated[..., add] merges lists).
    tool_calls: Annotated[list[ToolCallTrace], add]
    model_calls: Annotated[list[ModelCall], add]
