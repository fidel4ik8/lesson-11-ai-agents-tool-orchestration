"""Streamlit demo for the Personal Finance Crew.

Two tabs:
  * Chat — interactive multi-turn against baseline or crew (sidebar switch).
  * Eval — load / refresh golden set results, compare both architectures.

Run:
    uv run streamlit run app.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

from src.api import answer as run_answer
from src.llm import LANGFUSE_ENABLED


HOMEWORK_DIR = Path(__file__).resolve().parent
RESULTS_PATH = HOMEWORK_DIR / "evals" / "results.json"
SUMMARY_PATH = HOMEWORK_DIR / "evals" / "summary.md"
GOLDEN_PATH = HOMEWORK_DIR / "evals" / "golden_set.json"


# ───────────────────────────────────────── page setup ──────────────────────


st.set_page_config(
    page_title="Personal Finance Crew",
    page_icon="💸",
    layout="wide",
)


def _init_state():
    st.session_state.setdefault("history", [])      # for the chat tab
    st.session_state.setdefault("traces", [])       # parallel: trace per assistant turn
    st.session_state.setdefault("arch", "crew")
    st.session_state.setdefault("results", None)


_init_state()


# ───────────────────────────────────────── sidebar ─────────────────────────


with st.sidebar:
    st.markdown("### Architecture")
    arch = st.radio(
        "Pick one",
        options=["crew", "baseline"],
        index=0 if st.session_state.arch == "crew" else 1,
        label_visibility="collapsed",
        captions=["LangGraph · multi-agent", "OpenAI tool_calls · single LLM"],
    )
    st.session_state.arch = arch

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.traces = []
        st.rerun()

    st.divider()
    st.caption(f"Langfuse tracing: **{'on' if LANGFUSE_ENABLED else 'off'}**")
    st.caption(f"Dataset: 842 txns · 2024-12-01 → 2025-11-30")


# ───────────────────────────────────────── tabs ────────────────────────────


tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Eval"])


# ─── Chat tab ──────────────────────────────────────────────────────────────


with tab_chat:
    st.markdown(f"### Personal Finance Coach — `{arch}`")

    # Render history
    for i, msg in enumerate(st.session_state.history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Assistant turns may have a paired trace
            if msg["role"] == "assistant":
                idx = i // 2  # i: 0=user,1=assistant,2=user,3=assistant...
                if idx < len(st.session_state.traces):
                    trace = st.session_state.traces[idx]
                    with st.expander(
                        f"Trace · {trace['arch']} · "
                        f"{trace['latency_ms']} ms · "
                        f"${trace['cost_usd']:.5f} · "
                        f"{trace['total_tokens']} tokens",
                        expanded=False,
                    ):
                        if trace["arch"] == "crew" and trace.get("intent"):
                            st.caption(
                                f"**intent:** `{trace['intent']}` "
                                f"({trace.get('intent_rationale', '')})"
                            )

                        if trace["tool_calls"]:
                            st.markdown("**Tool calls**")
                            for tc in trace["tool_calls"]:
                                st.markdown(
                                    f"- `{tc['tool']}` ({tc['latency_ms']} ms)"
                                )
                                st.code(json.dumps(tc["args"], ensure_ascii=False, indent=2),
                                        language="json")
                                st.caption("result preview:")
                                preview = json.dumps(tc["result"], ensure_ascii=False, indent=2)
                                if len(preview) > 1500:
                                    preview = preview[:1500] + "\n…"
                                st.code(preview, language="json")
                        else:
                            st.caption("No tool calls.")

                        st.markdown("**Model calls**")
                        for m in trace["model_calls"]:
                            st.markdown(
                                f"- `{m['agent'] or 'baseline'}` · {m['model']} · "
                                f"in={m['input_tokens']} out={m['output_tokens']} "
                                f"· {m['latency_ms']} ms · ${m['cost_usd']:.5f}"
                            )

                        if trace.get("cost_by_agent"):
                            st.markdown("**Cost by agent**")
                            st.dataframe(
                                pd.DataFrame(
                                    [(k, f"${v:.5f}") for k, v in trace["cost_by_agent"].items()],
                                    columns=["agent", "cost_usd"],
                                ),
                                hide_index=True,
                                use_container_width=False,
                            )

    # Input
    user_query = st.chat_input("Запитай про свої витрати…")
    if user_query:
        st.session_state.history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Prepare history slice without the just-added user msg for the API call;
        # API expects history of prior user/assistant pairs, then accepts the new query.
        api_history = st.session_state.history[:-1]
        with st.chat_message("assistant"):
            with st.spinner(f"{arch} thinking…"):
                try:
                    res = run_answer(user_query, history=api_history, arch=arch)
                except Exception as e:
                    st.error(f"Error: {e}")
                    res = None

            if res is not None:
                st.markdown(res.answer)
                st.session_state.history.append(
                    {"role": "assistant", "content": res.answer}
                )
                st.session_state.traces.append({
                    "arch": res.architecture,
                    "latency_ms": res.latency_ms,
                    "cost_usd": res.total_cost_usd,
                    "total_tokens": res.total_tokens,
                    "intent": res.metadata.get("intent"),
                    "intent_rationale": res.metadata.get("intent_rationale"),
                    "tool_calls": [
                        {
                            "tool": tc.tool,
                            "args": tc.args,
                            "result": tc.result,
                            "latency_ms": tc.latency_ms,
                        }
                        for tc in res.tool_calls
                    ],
                    "model_calls": [asdict(m) for m in res.model_calls],
                    "cost_by_agent": res.cost_by_agent() if res.architecture == "crew" else {},
                })
                st.rerun()


# ─── Eval tab ──────────────────────────────────────────────────────────────


def _load_results() -> list[dict] | None:
    if not RESULTS_PATH.exists():
        return None
    return json.loads(RESULTS_PATH.read_text())


def _aggregate(records: list[dict], arch: str) -> dict:
    subset = [r for r in records if r["architecture"] == arch]
    if not subset:
        return {}
    n = len(subset)
    latencies = sorted(r["latency_ms"] for r in subset)
    costs = [r["cost_usd"] for r in subset]
    tokens = [r["total_tokens"] for r in subset]
    scores: dict[str, list[float]] = {}
    for r in subset:
        for k, v in r["scores"].items():
            scores.setdefault(k, []).append(v)
    return {
        "n": n,
        "latency_p50_ms": latencies[n // 2],
        "latency_p95_ms": latencies[int(n * 0.95)] if n > 1 else latencies[0],
        "cost_per_task": sum(costs) / n,
        "tokens_per_task": sum(tokens) / n,
        "scores": {k: sum(v) / len(v) for k, v in scores.items()},
    }


with tab_eval:
    st.markdown("### Golden set evaluation")
    st.caption(
        f"Golden set: **{len(json.loads(GOLDEN_PATH.read_text()))} tasks** · "
        "5 stats / 5 advice / 3 multi-step / 2 fraud / 1 OOS"
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔁 Re-run experiments", use_container_width=True, type="primary"):
            with st.spinner("Running golden set on both architectures…"):
                import subprocess
                proc = subprocess.run(
                    ["uv", "run", "python", "-m", "evals.run_experiments"],
                    cwd=str(HOMEWORK_DIR), capture_output=True, text=True,
                )
                if proc.returncode != 0:
                    st.error(proc.stderr or "experiment failed")
                else:
                    st.success("Done.")
                    st.code(proc.stdout[-2000:])
                    st.rerun()
    with col2:
        if RESULTS_PATH.exists():
            mtime = RESULTS_PATH.stat().st_mtime
            st.caption(f"Last run: {pd.Timestamp.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')}")

    records = _load_results()
    if not records:
        st.info("No results yet. Click **Re-run experiments** above.")
    else:
        agg_baseline = _aggregate(records, "baseline")
        agg_crew = _aggregate(records, "crew")

        st.markdown("#### Side-by-side metrics")
        rows = []
        for metric, label, fmt in [
            ("n", "n_tasks", "{:d}"),
            ("latency_p50_ms", "latency p50 (ms)", "{:.0f}"),
            ("latency_p95_ms", "latency p95 (ms)", "{:.0f}"),
            ("cost_per_task", "cost / task (USD)", "${:.5f}"),
            ("tokens_per_task", "tokens / task", "{:.0f}"),
        ]:
            rows.append({
                "metric": label,
                "baseline": fmt.format(agg_baseline.get(metric, 0)),
                "crew": fmt.format(agg_crew.get(metric, 0)),
            })
        score_keys = sorted(
            set(agg_baseline.get("scores", {})) | set(agg_crew.get("scores", {}))
        )
        for k in score_keys:
            rows.append({
                "metric": f"score · {k}",
                "baseline": f"{agg_baseline.get('scores', {}).get(k, float('nan')):.3f}",
                "crew": f"{agg_crew.get('scores', {}).get(k, float('nan')):.3f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        st.markdown("#### Per-task drill-down")
        df = pd.DataFrame([
            {
                "task_id": r["task_id"],
                "category": r["category"],
                "arch": r["architecture"],
                "latency_ms": r["latency_ms"],
                "cost_usd": round(r["cost_usd"], 5),
                "tokens": r["total_tokens"],
                "tools": ",".join(r["tools_called"]) or "—",
                "intent": r.get("intent") or "—",
                **{f"s_{k}": round(v, 2) for k, v in r["scores"].items()},
            }
            for r in records
        ])
        category = st.selectbox(
            "Filter category",
            ["all"] + sorted(df["category"].unique().tolist()),
            index=0,
        )
        if category != "all":
            df = df[df["category"] == category]
        st.dataframe(df, hide_index=True, use_container_width=True)
