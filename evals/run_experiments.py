"""Run the golden set against both architectures, collect metrics,
log scores to Langfuse (if enabled), and print a comparison table.

Outputs:
  evals/results.json        — raw per-task results
  evals/summary.md          — markdown table used in REPORT.md
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evals.evaluators import Score, evaluate
from src.api import answer as run_answer
from src.llm import LANGFUSE_ENABLED

if LANGFUSE_ENABLED:
    from langfuse import get_client as get_langfuse
    from langfuse import observe
else:
    get_langfuse = None  # type: ignore[assignment]

    def observe(*args, **kwargs):  # no-op
        def _wrap(fn):
            return fn
        return _wrap if args and not callable(args[0]) else (args[0] if args else _wrap)


HOMEWORK_DIR = Path(__file__).resolve().parent.parent
GOLDEN_PATH = HOMEWORK_DIR / "evals" / "golden_set.json"
RESULTS_PATH = HOMEWORK_DIR / "evals" / "results.json"
SUMMARY_PATH = HOMEWORK_DIR / "evals" / "summary.md"


@dataclass
class TaskRunRecord:
    task_id: str
    category: str
    query: str
    architecture: str
    answer: str
    latency_ms: int
    total_tokens: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    tools_called: list[str]
    intent: str | None
    cost_by_agent: dict[str, float] = field(default_factory=dict)
    tokens_by_agent: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    score_reasons: dict[str, str] = field(default_factory=dict)


def _score_dict(scores: list[Score]) -> tuple[dict[str, float], dict[str, str]]:
    values: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for s in scores:
        if not math.isnan(s.value):
            values[s.name] = s.value
        reasons[s.name] = s.reasoning
    return values, reasons


@observe(name="golden_set_task")
def run_task(task: dict, architecture: str) -> TaskRunRecord:
    res = run_answer(task["query"], arch=architecture)  # type: ignore[arg-type]
    scores = evaluate(task, res)
    values, reasons = _score_dict(scores)

    if LANGFUSE_ENABLED and get_langfuse is not None:
        try:
            lf = get_langfuse()
            lf.update_current_span(
                metadata={"task_id": task["id"], "architecture": architecture,
                          "category": task.get("category")},
            )
            for s in scores:
                if math.isnan(s.value):
                    continue
                lf.score_current_trace(
                    name=s.name,
                    value=s.value,
                    comment=s.reasoning,
                    data_type="NUMERIC",
                )
        except Exception as e:
            print(f"  ! Langfuse scoring failed: {e}")

    return TaskRunRecord(
        task_id=task["id"],
        category=task["category"],
        query=task["query"],
        architecture=architecture,
        answer=res.answer,
        latency_ms=res.latency_ms,
        total_tokens=res.total_tokens,
        input_tokens=res.total_input_tokens,
        output_tokens=res.total_output_tokens,
        cost_usd=res.total_cost_usd,
        tools_called=[tc.tool for tc in res.tool_calls],
        intent=res.metadata.get("intent"),
        cost_by_agent=res.cost_by_agent() if res.architecture == "crew" else {},
        tokens_by_agent=res.tokens_by_agent() if res.architecture == "crew" else {},
        scores=values,
        score_reasons=reasons,
    )


def aggregate(records: list[TaskRunRecord]) -> dict[str, Any]:
    if not records:
        return {}
    latencies = [r.latency_ms for r in records]
    costs = [r.cost_usd for r in records]
    tokens = [r.total_tokens for r in records]

    score_names = sorted({k for r in records for k in r.scores})
    score_means: dict[str, float] = {}
    for name in score_names:
        values = [r.scores[name] for r in records if name in r.scores]
        if values:
            score_means[name] = sum(values) / len(values)

    # crew-specific aggregates
    arch = records[0].architecture
    inter_agent_overhead_pct = None
    cost_by_agent_total: dict[str, float] = {}
    tokens_by_agent_total: dict[str, int] = {}
    if arch == "crew":
        # inter-agent overhead = % of total input tokens consumed by non-data-analyst
        # agents whose input includes context from previous agents (router gets only
        # query, advisor gets findings).  We treat advisor's input tokens as the
        # explicit "context passing" cost.
        total_input = sum(r.input_tokens for r in records)
        advisor_input = 0
        for r in records:
            for m_agent, t in r.tokens_by_agent.items():
                cost_by_agent_total[m_agent] = (
                    cost_by_agent_total.get(m_agent, 0.0) + r.cost_by_agent.get(m_agent, 0.0)
                )
                tokens_by_agent_total[m_agent] = tokens_by_agent_total.get(m_agent, 0) + t
            # we don't have per-agent input split; approximate advisor as input/2 of advisor tokens
        # Use a simpler proxy: cost of router + advisor / total cost
        router_cost = cost_by_agent_total.get("router", 0.0)
        advisor_cost = cost_by_agent_total.get("advisor", 0.0)
        total_cost = sum(costs) or 1e-9
        inter_agent_overhead_pct = (router_cost + advisor_cost) / total_cost * 100

    return {
        "n_tasks": len(records),
        "latency_p50_ms": int(statistics.median(latencies)),
        "latency_p95_ms": int(_percentile(latencies, 95)),
        "latency_mean_ms": int(statistics.mean(latencies)),
        "cost_mean_usd": round(statistics.mean(costs), 5),
        "cost_total_usd": round(sum(costs), 4),
        "tokens_mean": int(statistics.mean(tokens)),
        "tokens_total": sum(tokens),
        "score_means": {k: round(v, 3) for k, v in score_means.items()},
        "inter_agent_overhead_pct": (
            round(inter_agent_overhead_pct, 1) if inter_agent_overhead_pct is not None else None
        ),
        "cost_by_agent_total": {k: round(v, 5) for k, v in cost_by_agent_total.items()},
        "tokens_by_agent_total": tokens_by_agent_total,
    }


def _percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0
    s = sorted(values)
    k = (len(s) - 1) * pct / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def main() -> None:
    tasks = json.loads(GOLDEN_PATH.read_text())
    print(f"Loaded {len(tasks)} golden-set tasks.\n")

    records: list[TaskRunRecord] = []
    for arch in ("baseline", "crew"):
        print(f"=== Running {arch.upper()} ===")
        for i, task in enumerate(tasks, 1):
            print(f"  [{i:>2}/{len(tasks)}] {task['id']:35s} ", end="", flush=True)
            t0 = time.perf_counter()
            try:
                rec = run_task(task, arch)
                records.append(rec)
                dt = (time.perf_counter() - t0) * 1000
                scores_str = " ".join(
                    f"{k}={v:.2f}" for k, v in rec.scores.items() if k in {
                        "tool_selection_accuracy", "groundedness", "judge_success"
                    }
                )
                print(f"latency={rec.latency_ms}ms  ${rec.cost_usd:.4f}  {scores_str}")
            except Exception as e:
                print(f"FAILED: {e}")
        print()

    # Persist raw results
    RESULTS_PATH.write_text(
        json.dumps([asdict(r) for r in records], ensure_ascii=False, indent=2)
    )

    # Aggregate and print
    baseline_recs = [r for r in records if r.architecture == "baseline"]
    crew_recs = [r for r in records if r.architecture == "crew"]
    summary = {
        "baseline": aggregate(baseline_recs),
        "crew": aggregate(crew_recs),
        "by_category": {
            cat: {
                "baseline": aggregate([r for r in baseline_recs if r.category == cat]),
                "crew": aggregate([r for r in crew_recs if r.category == cat]),
            }
            for cat in sorted({r.category for r in records})
        },
    }
    SUMMARY_PATH.write_text(
        "# Experiment summary\n\n```json\n"
        + json.dumps(summary, ensure_ascii=False, indent=2)
        + "\n```\n"
    )

    print("=" * 78)
    print(f"{'metric':<28} {'baseline':>20} {'crew':>20}")
    print("-" * 78)
    rows = [
        ("n_tasks", "n_tasks"),
        ("latency_p50_ms", "latency p50 (ms)"),
        ("latency_p95_ms", "latency p95 (ms)"),
        ("cost_mean_usd", "cost / task (USD)"),
        ("cost_total_usd", "cost total (USD)"),
        ("tokens_mean", "tokens / task"),
    ]
    for key, label in rows:
        b = summary["baseline"].get(key, "—")
        c = summary["crew"].get(key, "—")
        print(f"{label:<28} {str(b):>20} {str(c):>20}")
    print("-" * 78)
    print("Score means (higher is better):")
    score_keys = sorted(
        set(summary["baseline"].get("score_means", {}))
        | set(summary["crew"].get("score_means", {}))
    )
    for k in score_keys:
        b = summary["baseline"].get("score_means", {}).get(k, "—")
        c = summary["crew"].get("score_means", {}).get(k, "—")
        print(f"  {k:<26} {str(b):>20} {str(c):>20}")
    print("-" * 78)
    if summary["crew"].get("inter_agent_overhead_pct") is not None:
        print(f"  inter_agent_overhead_pct (crew): {summary['crew']['inter_agent_overhead_pct']}%")
        print(f"  cost_by_agent (crew): {summary['crew']['cost_by_agent_total']}")
    print("=" * 78)
    print(f"\nWrote {RESULTS_PATH.relative_to(HOMEWORK_DIR)} and {SUMMARY_PATH.relative_to(HOMEWORK_DIR)}.")
    if LANGFUSE_ENABLED:
        print("Scores also pushed to Langfuse.")


if __name__ == "__main__":
    main()
