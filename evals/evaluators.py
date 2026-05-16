"""Evaluators for golden-set runs.

Each function takes (task, AnswerResult) and returns a float score in [0, 1]
plus an optional reasoning string for trace.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.llm import MODEL_CHEAP, AnswerResult, compute_cost_usd, get_client


@dataclass
class Score:
    name: str
    value: float
    reasoning: str = ""


# ───────────────────────────────────────── 1. intent_accuracy ───────────────


def intent_accuracy(task: dict, result: AnswerResult) -> Score:
    """For crew only. Compares Router's intent against expected_intent."""
    if result.architecture != "crew":
        return Score("intent_accuracy", float("nan"), "N/A for baseline")
    actual = result.metadata.get("intent")
    expected = task.get("expected_intent")
    ok = actual == expected
    return Score(
        "intent_accuracy",
        1.0 if ok else 0.0,
        f"expected={expected} actual={actual}",
    )


# ───────────────────────────────────────── 2. tool_selection_accuracy ───────


def tool_selection_accuracy(task: dict, result: AnswerResult) -> Score:
    """Score = 1.0 if all expected_tools_min are present in actual tool_calls.
    For tasks that explicitly expect NO tools (fraud/oos), full score iff 0
    tools called.
    """
    expected = set(task.get("expected_tools_min", []))
    actual = {tc.tool for tc in result.tool_calls}
    if not expected:
        # Expect NO tool calls (fraud / oos)
        ok = len(actual) == 0
        return Score(
            "tool_selection_accuracy",
            1.0 if ok else 0.0,
            f"expected=∅ actual={sorted(actual)}",
        )
    missing = expected - actual
    score = (len(expected) - len(missing)) / len(expected)
    return Score(
        "tool_selection_accuracy",
        score,
        f"expected⊆{sorted(expected)} actual={sorted(actual)} missing={sorted(missing)}",
    )


# ───────────────────────────────────────── 3. groundedness ──────────────────


_NUMBER_RE = re.compile(r"\$?-?\d+(?:[.,]\d+)?")


def _extract_numbers(text: str) -> list[float]:
    """Pull all numeric tokens out of free text. Returns floats."""
    nums = []
    for m in _NUMBER_RE.finditer(text or ""):
        s = m.group().lstrip("$").replace(",", ".")
        try:
            nums.append(float(s))
        except ValueError:
            continue
    return nums


def _numbers_from_tool_results(result: AnswerResult) -> set[float]:
    """All numeric values that appeared in tool results — they are by
    construction grounded in the database."""
    payload = json.dumps([tc.result for tc in result.tool_calls], ensure_ascii=False)
    out: set[float] = set()
    for m in _NUMBER_RE.finditer(payload):
        s = m.group().lstrip("$").replace(",", ".")
        try:
            out.add(float(s))
        except ValueError:
            continue
    return out


def groundedness(task: dict, result: AnswerResult) -> Score:
    """Fraction of substantial numbers in the response that also appear in
    tool-call results (within ±0.5 absolute or ±2 % relative tolerance).
    Ignores trivial integers (e.g. enumeration "1.", "2.").
    """
    response_nums = [n for n in _extract_numbers(result.answer) if abs(n) >= 5]
    if not response_nums:
        return Score("groundedness", 1.0, "no substantial numbers in answer")
    tool_nums = _numbers_from_tool_results(result)
    if not tool_nums:
        return Score("groundedness", 0.0, "answer has numbers but no tools called")

    matched = 0
    unmatched: list[float] = []
    for n in response_nums:
        if any(abs(n - t) <= max(0.5, abs(t) * 0.02) for t in tool_nums):
            matched += 1
        else:
            unmatched.append(n)
    score = matched / len(response_nums)
    return Score(
        "groundedness",
        score,
        f"matched {matched}/{len(response_nums)} numbers"
        + (f"; ungrounded: {unmatched}" if unmatched else ""),
    )


# ───────────────────────────────────────── 4. judge_success ─────────────────


_JUDGE_PROMPT = """\
You are evaluating a financial assistant's response against the user query
and a list of explicit criteria. Reply ONLY in JSON.

User query:
{query}

Assistant response:
{response}

Criteria (ALL must be satisfied for a passing score):
{criteria}

Return:
{{
  "pass": <true|false>,
  "satisfied": ["criterion 1 verbatim", ...],
  "violated":  ["criterion N verbatim", ...],
  "rationale": "<one sentence>"
}}
"""


def judge_success(task: dict, result: AnswerResult) -> Score:
    """Use a cheap LLM as a binary judge against task-specific criteria."""
    criteria = task.get("judge_criteria") or []
    if not criteria:
        return Score("judge_success", 1.0, "no criteria")
    crit_block = "\n".join(f"- {c}" for c in criteria)

    client = get_client()
    resp = client.chat.completions.create(
        model=MODEL_CHEAP,
        messages=[
            {"role": "system", "content": "You are a strict evaluation judge."},
            {
                "role": "user",
                "content": _JUDGE_PROMPT.format(
                    query=task["query"],
                    response=result.answer,
                    criteria=crit_block,
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return Score("judge_success", 0.0, "judge returned invalid JSON")

    if data.get("pass") is True:
        return Score("judge_success", 1.0, data.get("rationale", "pass"))
    violated = data.get("violated", [])
    score = 1.0 - (len(violated) / len(criteria)) if criteria else 0.0
    score = max(0.0, min(1.0, score))
    return Score(
        "judge_success",
        score,
        f"violated={violated[:3]} | rationale={data.get('rationale', '')}",
    )


# ───────────────────────────────────────── 5. escalation_correct ────────────


_ESCALATION_KEYWORDS = ["заблок", "блок", "підтримк", "support", "dispute", "chargeback"]


def escalation_correct(task: dict, result: AnswerResult) -> Score:
    """For fraud tasks: response MUST mention card-block and support escalation."""
    if task.get("expected_intent") != "fraud":
        return Score("escalation_correct", float("nan"), "N/A — not a fraud task")
    text = (result.answer or "").lower()
    hits = [kw for kw in _ESCALATION_KEYWORDS if kw in text]
    has_block = any(kw in text for kw in ["заблок", "блок", "block"])
    has_support = any(kw in text for kw in ["підтримк", "support", "dispute", "chargeback"])
    ok = has_block and has_support
    return Score(
        "escalation_correct",
        1.0 if ok else 0.0,
        f"keywords_hit={hits} block={has_block} support={has_support}",
    )


# ───────────────────────────────────────── public API ───────────────────────


ALL_EVALUATORS = [
    intent_accuracy,
    tool_selection_accuracy,
    groundedness,
    judge_success,
    escalation_correct,
]


def evaluate(task: dict, result: AnswerResult) -> list[Score]:
    """Run every evaluator and return the list of scores."""
    return [ev(task, result) for ev in ALL_EVALUATORS]
