"""LangGraph wiring + public ``answer()`` entry point for the crew."""

from __future__ import annotations

import time

from langgraph.graph import END, StateGraph

from src.crew.nodes import (
    advisor_node,
    data_analyst_node,
    oos_node,
    router_node,
    safety_node,
)
from src.crew.state import CrewState
from src.llm import LANGFUSE_ENABLED, AnswerResult

if LANGFUSE_ENABLED:
    from langfuse import observe
else:
    def observe(*args, **kwargs):
        def _wrap(fn):
            return fn
        return _wrap if args and not callable(args[0]) else (args[0] if args else _wrap)


def _route_from_router(state: CrewState) -> str:
    intent = state.get("intent", "stats")
    if intent == "fraud":
        return "safety"
    if intent == "out_of_scope":
        return "oos"
    return "data_analyst"


def _route_from_data_analyst(state: CrewState) -> str:
    # Simple stats: data analyst's output IS the final answer → end.
    if state.get("intent") == "stats":
        return END
    return "advisor"


def build_graph():
    g = StateGraph(CrewState)
    g.add_node("router", router_node)
    g.add_node("data_analyst", data_analyst_node)
    g.add_node("advisor", advisor_node)
    g.add_node("safety", safety_node)
    g.add_node("oos", oos_node)

    g.set_entry_point("router")
    g.add_conditional_edges(
        "router", _route_from_router,
        {"safety": "safety", "oos": "oos", "data_analyst": "data_analyst"},
    )
    g.add_conditional_edges(
        "data_analyst", _route_from_data_analyst,
        {END: END, "advisor": "advisor"},
    )
    g.add_edge("advisor", END)
    g.add_edge("safety", END)
    g.add_edge("oos", END)
    return g.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


@observe(name="crew_answer")
def answer(query: str, history: list[dict] | None = None) -> AnswerResult:
    """Run the crew for one user turn and return a unified AnswerResult."""
    graph = get_graph()
    initial_state: CrewState = {
        "query": query,
        "history": history or [],
        "tool_calls": [],
        "model_calls": [],
    }
    t0 = time.perf_counter()
    final_state = graph.invoke(initial_state)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    return AnswerResult(
        architecture="crew",
        answer=final_state.get("answer", ""),
        tool_calls=final_state.get("tool_calls", []),
        model_calls=final_state.get("model_calls", []),
        latency_ms=latency_ms,
        metadata={
            "intent": final_state.get("intent"),
            "intent_rationale": final_state.get("intent_rationale"),
        },
    )


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "Де можу зекономити $200 цього місяця?"
    res = answer(q)
    print("\n=== ANSWER ===")
    print(res.answer)
    print(f"\n=== META === intent={res.metadata['intent']}  "
          f"rationale={res.metadata['intent_rationale']}")
    print("\n=== TOOL CALLS ===")
    for tc in res.tool_calls:
        print(f"  → {tc.tool}({tc.args}) [{tc.latency_ms}ms]")
    print("\n=== MODEL CALLS ===")
    for m in res.model_calls:
        print(f"  · {m.agent:14s} {m.model:14s} in={m.input_tokens:>5d} "
              f"out={m.output_tokens:>4d} {m.latency_ms:>4d}ms ${m.cost_usd:.5f}")
    print(f"\nlatency={res.latency_ms}ms  tokens={res.total_tokens}  "
          f"cost=${res.total_cost_usd:.5f}")
    print(f"cost_by_agent={res.cost_by_agent()}")
