"""LangGraph nodes for the multi-agent crew.

Every node:
  * accepts a ``CrewState`` dict
  * returns a partial state update (LangGraph merges it)
  * appends its ``ModelCall`` records tagged with ``agent="..."`` so
    we can compute cost_breakdown_by_agent later.
"""

from __future__ import annotations

import json
import time
from typing import Any

from src.crew.prompts import (
    ADVISOR_PROMPT,
    DATA_ANALYST_PROMPT,
    OOS_PROMPT,
    ROUTER_PROMPT,
    SAFETY_PROMPT,
)
from src.crew.state import CrewState
from src.llm import (
    LANGFUSE_ENABLED,
    MODEL_CHEAP,
    MODEL_SMART,
    ModelCall,
    ToolCallTrace,
    compute_cost_usd,
    get_client,
)
from src.tools import execute_tool, get_openai_tools

if LANGFUSE_ENABLED:
    from langfuse import get_client as _lf_client
    from langfuse import observe
else:
    _lf_client = None  # type: ignore[assignment]

    def observe(*args, **kwargs):  # no-op
        def _wrap(fn):
            return fn
        return _wrap if args and not callable(args[0]) else (args[0] if args else _wrap)


def _attach_span_meta(**fields):
    if not LANGFUSE_ENABLED or _lf_client is None:
        return
    try:
        _lf_client().update_current_span(metadata=fields)
    except Exception:
        pass


MAX_DATA_ANALYST_ITERATIONS = 6


# ───────────────────────────────────────── helpers ──────────────────────────


def _llm_call(
    *,
    model: str,
    messages: list[dict],
    agent: str,
    tools: list[dict] | None = None,
    response_format: dict | None = None,
    temperature: float = 0.3,
) -> tuple[Any, ModelCall]:
    """Single LLM round-trip + ModelCall record (tagged by agent)."""
    client = get_client()
    t0 = time.perf_counter()
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if response_format:
        kwargs["response_format"] = response_format
    resp = client.chat.completions.create(**kwargs)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    usage = resp.usage
    cost = compute_cost_usd(model, usage.prompt_tokens, usage.completion_tokens)
    call = ModelCall(
        model=model,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        latency_ms=latency_ms,
        agent=agent,
        cost_usd=cost,
    )
    return resp, call


def _serialize_assistant(msg) -> dict:
    out: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return out


# ───────────────────────────────────────── 1. router ────────────────────────


@observe(name="router")
def router_node(state: CrewState) -> dict:
    messages = [
        {"role": "system", "content": ROUTER_PROMPT},
        {"role": "user", "content": state["query"]},
    ]
    # Include short history if present (only last 4 turns to keep cheap)
    if state.get("history"):
        history = state["history"][-4:]
        messages = (
            [{"role": "system", "content": ROUTER_PROMPT}]
            + history
            + [{"role": "user", "content": state["query"]}]
        )

    resp, call = _llm_call(
        model=MODEL_CHEAP,
        messages=messages,
        agent="router",
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
        intent = parsed.get("intent", "stats")
        rationale = parsed.get("rationale", "")
    except json.JSONDecodeError:
        intent = "stats"
        rationale = "router_parse_failed"

    if intent not in {"stats", "advice", "multi_step", "fraud", "out_of_scope"}:
        intent = "stats"

    _attach_span_meta(intent=intent, rationale=rationale)
    return {
        "intent": intent,
        "intent_rationale": rationale,
        "model_calls": [call],
    }


# ───────────────────────────────────────── 2. data_analyst ──────────────────


@observe(name="data_analyst")
def data_analyst_node(state: CrewState) -> dict:
    tools = get_openai_tools()
    messages: list[dict] = [{"role": "system", "content": DATA_ANALYST_PROMPT}]
    if state.get("history"):
        messages.extend(state["history"][-6:])
    messages.append({"role": "user", "content": state["query"]})

    tool_traces: list[ToolCallTrace] = []
    model_calls: list[ModelCall] = []
    final_text = ""

    for _ in range(MAX_DATA_ANALYST_ITERATIONS):
        resp, call = _llm_call(
            model=MODEL_SMART,
            messages=messages,
            agent="data_analyst",
            tools=tools,
            temperature=0.2,
        )
        model_calls.append(call)
        msg = resp.choices[0].message
        messages.append(_serialize_assistant(msg))

        if not msg.tool_calls:
            final_text = msg.content or ""
            break

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            t0 = time.perf_counter()
            result = execute_tool(tc.function.name, args)
            tlat = int((time.perf_counter() - t0) * 1000)
            tool_traces.append(ToolCallTrace(
                tool=tc.function.name, args=args, result=result, latency_ms=tlat,
            ))
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })
    else:
        final_text = "Не вдалося завершити збір даних — забагато ітерацій."

    _attach_span_meta(
        tools_called=[tc.tool for tc in tool_traces],
        iterations=len(model_calls),
    )
    return {
        "findings": final_text,
        # If the route ends here (stats branch), this becomes the answer too.
        "answer": final_text,
        "tool_calls": tool_traces,
        "model_calls": model_calls,
    }


# ───────────────────────────────────────── 3. advisor ───────────────────────


@observe(name="advisor")
def advisor_node(state: CrewState) -> dict:
    user_block = (
        f"Запит користувача:\n{state['query']}\n\n"
        f"Findings від Data Analyst (використовуй лише ці числа):\n"
        f"{state.get('findings', '')}"
    )
    messages = [
        {"role": "system", "content": ADVISOR_PROMPT},
        {"role": "user", "content": user_block},
    ]
    resp, call = _llm_call(
        model=MODEL_SMART,
        messages=messages,
        agent="advisor",
        temperature=0.4,
    )
    answer = resp.choices[0].message.content or ""
    return {"answer": answer, "model_calls": [call]}


# ───────────────────────────────────────── 4. safety ────────────────────────


@observe(name="safety")
def safety_node(state: CrewState) -> dict:
    messages = [
        {"role": "system", "content": SAFETY_PROMPT},
        {"role": "user", "content": state["query"]},
    ]
    resp, call = _llm_call(
        model=MODEL_CHEAP,
        messages=messages,
        agent="safety",
        temperature=0.3,
    )
    return {"answer": resp.choices[0].message.content or "", "model_calls": [call]}


# ───────────────────────────────────────── 5. out_of_scope ──────────────────


@observe(name="out_of_scope")
def oos_node(state: CrewState) -> dict:
    messages = [
        {"role": "system", "content": OOS_PROMPT},
        {"role": "user", "content": state["query"]},
    ]
    resp, call = _llm_call(
        model=MODEL_CHEAP,
        messages=messages,
        agent="out_of_scope",
        temperature=0.3,
    )
    return {"answer": resp.choices[0].message.content or "", "model_calls": [call]}
