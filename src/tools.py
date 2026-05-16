"""Finance tools exposed to LLM agents.

Each tool is a thin SQL-backed function with a Pydantic argument schema.
The same definitions are consumed by:
  * baseline single-agent (OpenAI tool_calls)
  * multi-agent crew (LangGraph nodes / langchain tools)

Design rules:
  * LLM never does arithmetic — every number returned to it comes from SQL.
  * Inputs validated by Pydantic before hitting the DB.
  * Parameterised SQL only (no string concat) — defends against
    prompt-injected SQL fragments.
  * Outputs are JSON-serialisable dicts with positive ``spending``
    (we flip the sign of negative amounts) so the LLM does not have
    to remember "expenses are negative".
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, field_validator

from src.db import get_db

# Dataset spans 2024-12-01 .. 2025-11-30. We treat the last date as "today"
# so the agent can interpret relative phrases like "last week".
DATASET_FIRST_DATE = "2024-12-01"
DATASET_LAST_DATE = "2025-11-30"
TODAY = DATASET_LAST_DATE

CATEGORIES = [
    "coffee", "groceries", "restaurants", "delivery", "transport",
    "entertainment", "shopping", "health", "subscriptions",
    "utilities", "salary", "credit_payment", "travel",
]
ACCOUNTS = ["main_debit", "credit_card"]


# ───────────────────────────────────────── helpers ──────────────────────────


def _iso_date(value: str) -> str:
    """Validate a YYYY-MM-DD string, raising ValueError on bad input."""
    datetime.strptime(value, "%Y-%m-%d")
    return value


def _build_filters(
    start_date: str | None,
    end_date: str | None,
    category: str | None,
    merchant: str | None,
    account: str | None,
    min_amount: float | None,
    max_amount: float | None,
) -> tuple[str, list[Any]]:
    """Return ('WHERE ...', params) suitable for parameterised queries."""
    clauses: list[str] = []
    params: list[Any] = []
    if start_date:
        clauses.append("date_only >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("date_only <= ?")
        params.append(end_date)
    if category:
        clauses.append("category = ?")
        params.append(category)
    if merchant:
        clauses.append("LOWER(merchant) = LOWER(?)")
        params.append(merchant)
    if account:
        clauses.append("account = ?")
        params.append(account)
    if min_amount is not None:
        # min_amount/max_amount refer to absolute spending value
        clauses.append("ABS(amount) >= ?")
        params.append(min_amount)
    if max_amount is not None:
        clauses.append("ABS(amount) <= ?")
        params.append(max_amount)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def _time_of_day_bucket_sql() -> str:
    """SQL CASE expression mapping hour → bucket."""
    return (
        "CASE "
        "WHEN hour BETWEEN 5  AND 11 THEN 'morning' "
        "WHEN hour BETWEEN 12 AND 17 THEN 'afternoon' "
        "WHEN hour BETWEEN 18 AND 21 THEN 'evening' "
        "ELSE 'night' "
        "END"
    )


# ───────────────────────────────────────── 1. query_transactions ────────────


class QueryTransactionsArgs(BaseModel):
    """Filter raw transactions. Use this when you need individual rows
    (e.g. last 5 on a card, all Netflix payments) rather than aggregates."""

    start_date: str | None = Field(
        None, description="Inclusive YYYY-MM-DD; omit for no lower bound."
    )
    end_date: str | None = Field(
        None, description="Inclusive YYYY-MM-DD; omit for no upper bound."
    )
    category: Literal[
        "coffee", "groceries", "restaurants", "delivery", "transport",
        "entertainment", "shopping", "health", "subscriptions",
        "utilities", "salary", "credit_payment", "travel",
    ] | None = None
    merchant: str | None = Field(None, description="Exact match, case-insensitive.")
    account: Literal["main_debit", "credit_card"] | None = None
    min_amount: float | None = Field(
        None, description="Minimum absolute amount (e.g. 100 = $100 or more)."
    )
    max_amount: float | None = None
    limit: int = Field(50, ge=1, le=200)
    sort: Literal["date_desc", "date_asc", "amount_desc", "amount_asc"] = "date_desc"

    @field_validator("start_date", "end_date")
    @classmethod
    def _v_date(cls, v):
        return _iso_date(v) if v else v


def query_transactions(args: QueryTransactionsArgs) -> dict:
    where, params = _build_filters(
        args.start_date, args.end_date, args.category,
        args.merchant, args.account, args.min_amount, args.max_amount,
    )
    order = {
        "date_desc": "date DESC",
        "date_asc": "date ASC",
        "amount_desc": "ABS(amount) DESC",
        "amount_asc": "ABS(amount) ASC",
    }[args.sort]
    sql = (
        "SELECT date, merchant, amount, category, account, recurring "
        f"FROM transactions {where} ORDER BY {order} LIMIT ?"
    )
    rows = get_db().execute(sql, [*params, args.limit]).fetchall()
    return {
        "count": len(rows),
        "transactions": [
            {
                "date": r["date"],
                "merchant": r["merchant"],
                "amount": round(r["amount"], 2),
                "category": r["category"],
                "account": r["account"],
                "recurring": bool(r["recurring"]),
            }
            for r in rows
        ],
    }


# ───────────────────────────────────────── 2. aggregate_spending ────────────

GroupBy = Literal[
    "category", "merchant", "account", "month", "weekday", "is_weekend",
    "time_of_day",
]


class AggregateSpendingArgs(BaseModel):
    """Sum and average spending grouped by a dimension. Use this for
    'how much did I spend on X', 'top N categories', 'weekday vs weekend'."""

    group_by: GroupBy = Field(
        ..., description="Dimension to group by. 'time_of_day' buckets: morning(5-11), afternoon(12-17), evening(18-21), night(22-4)."
    )
    start_date: str | None = None
    end_date: str | None = None
    category: Literal[
        "coffee", "groceries", "restaurants", "delivery", "transport",
        "entertainment", "shopping", "health", "subscriptions",
        "utilities", "salary", "credit_payment", "travel",
    ] | None = Field(None, description="Restrict to a single category before grouping.")
    top_n: int = Field(10, ge=1, le=50)
    include_income: bool = Field(
        False,
        description="If True, salary is included (positive amounts). Default: expenses only.",
    )

    @field_validator("start_date", "end_date")
    @classmethod
    def _v_date(cls, v):
        return _iso_date(v) if v else v


_GROUP_EXPR: dict[str, str] = {
    "category": "category",
    "merchant": "merchant",
    "account": "account",
    "month": "substr(date_only, 1, 7)",
    "weekday": (
        "CASE weekday "
        "WHEN 0 THEN 'Mon' WHEN 1 THEN 'Tue' WHEN 2 THEN 'Wed' "
        "WHEN 3 THEN 'Thu' WHEN 4 THEN 'Fri' WHEN 5 THEN 'Sat' "
        "WHEN 6 THEN 'Sun' END"
    ),
    "is_weekend": "CASE WHEN is_weekend=1 THEN 'weekend' ELSE 'weekday' END",
    "time_of_day": _time_of_day_bucket_sql(),
}


def aggregate_spending(args: AggregateSpendingArgs) -> dict:
    where, params = _build_filters(
        args.start_date, args.end_date, args.category, None, None, None, None,
    )
    # exclude salary unless explicitly asked for
    if not args.include_income:
        clause = "category != 'salary'"
        where = f"{where} AND {clause}" if where else f"WHERE {clause}"

    group_expr = _GROUP_EXPR[args.group_by]
    sql = (
        f"SELECT {group_expr} AS bucket, "
        "SUM(-amount) AS spending, "
        "COUNT(*) AS txn_count, "
        "AVG(-amount) AS avg_per_txn "
        f"FROM transactions {where} "
        "GROUP BY bucket "
        "ORDER BY spending DESC "
        "LIMIT ?"
    )
    rows = get_db().execute(sql, [*params, args.top_n]).fetchall()

    # Overall totals over the same filter
    total_sql = (
        "SELECT SUM(-amount) AS spending, COUNT(*) AS txn_count "
        f"FROM transactions {where}"
    )
    total_row = get_db().execute(total_sql, params).fetchone()

    return {
        "group_by": args.group_by,
        "period": {"start": args.start_date, "end": args.end_date},
        "total_spending": round(total_row["spending"] or 0.0, 2),
        "total_transactions": total_row["txn_count"],
        "groups": [
            {
                "bucket": r["bucket"],
                "spending": round(r["spending"], 2),
                "txn_count": r["txn_count"],
                "avg_per_txn": round(r["avg_per_txn"], 2),
            }
            for r in rows
        ],
    }


# ───────────────────────────────────────── 3. find_recurring_subscriptions ──


class FindRecurringArgs(BaseModel):
    """Find recurring (subscription-like) merchants. Highlights stale ones
    that may be forgotten subscriptions (last charge older than threshold)."""

    stale_after_days: int = Field(
        45,
        ge=7, le=365,
        description="A subscription is flagged 'stale' if last seen more than this many days ago (relative to dataset end 2025-11-30).",
    )
    include_non_recurring: bool = Field(
        False,
        description="If True, include merchants that appear regularly even if not flagged recurring in the data.",
    )


def find_recurring_subscriptions(args: FindRecurringArgs) -> dict:
    where = "WHERE recurring = 1" if not args.include_non_recurring else ""
    sql = (
        "SELECT merchant, "
        "COUNT(*) AS occurrences, "
        "SUM(-amount) AS total_paid, "
        "AVG(-amount) AS avg_per_occurrence, "
        "MIN(date_only) AS first_seen, "
        "MAX(date_only) AS last_seen "
        f"FROM transactions {where} "
        "GROUP BY merchant "
        "HAVING COUNT(*) >= 2 "
        "ORDER BY total_paid DESC"
    )
    rows = get_db().execute(sql).fetchall()
    today = datetime.strptime(TODAY, "%Y-%m-%d").date()
    out = []
    for r in rows:
        last = datetime.strptime(r["last_seen"], "%Y-%m-%d").date()
        days_since = (today - last).days
        out.append({
            "merchant": r["merchant"],
            "occurrences": r["occurrences"],
            "total_paid": round(r["total_paid"], 2),
            "avg_per_occurrence": round(r["avg_per_occurrence"], 2),
            "first_seen": r["first_seen"],
            "last_seen": r["last_seen"],
            "days_since_last": days_since,
            "is_stale": days_since >= args.stale_after_days,
        })
    return {
        "today": TODAY,
        "stale_threshold_days": args.stale_after_days,
        "subscriptions": out,
    }


# ───────────────────────────────────────── 4. compare_periods ───────────────


class ComparePeriodsArgs(BaseModel):
    """Compare totals between two periods (e.g. YoY, MoM). Returns
    per-category deltas with absolute and percentage change."""

    period_a_start: str = Field(..., description="YYYY-MM-DD inclusive")
    period_a_end: str = Field(..., description="YYYY-MM-DD inclusive")
    period_b_start: str = Field(..., description="YYYY-MM-DD inclusive")
    period_b_end: str = Field(..., description="YYYY-MM-DD inclusive")
    group_by: Literal["category", "merchant", "account", "none"] = "category"

    @field_validator(
        "period_a_start", "period_a_end", "period_b_start", "period_b_end"
    )
    @classmethod
    def _v_date(cls, v):
        return _iso_date(v)


def _period_totals(start: str, end: str, group_by: str) -> dict:
    sql_total = (
        "SELECT SUM(-amount) AS spending, COUNT(*) AS txn_count "
        "FROM transactions "
        "WHERE date_only BETWEEN ? AND ? AND category != 'salary'"
    )
    total = get_db().execute(sql_total, [start, end]).fetchone()
    if group_by == "none":
        groups: list[dict] = []
    else:
        expr = _GROUP_EXPR[group_by]
        sql_g = (
            f"SELECT {expr} AS bucket, SUM(-amount) AS spending "
            "FROM transactions WHERE date_only BETWEEN ? AND ? "
            "AND category != 'salary' "
            "GROUP BY bucket ORDER BY spending DESC"
        )
        rows = get_db().execute(sql_g, [start, end]).fetchall()
        groups = [{"bucket": r["bucket"], "spending": round(r["spending"], 2)} for r in rows]
    return {
        "start": start,
        "end": end,
        "total_spending": round(total["spending"] or 0.0, 2),
        "total_transactions": total["txn_count"],
        "groups": groups,
    }


def compare_periods(args: ComparePeriodsArgs) -> dict:
    a = _period_totals(args.period_a_start, args.period_a_end, args.group_by)
    b = _period_totals(args.period_b_start, args.period_b_end, args.group_by)

    total_diff = b["total_spending"] - a["total_spending"]
    total_pct = (total_diff / a["total_spending"] * 100) if a["total_spending"] else None

    deltas: list[dict] = []
    if args.group_by != "none":
        a_map = {g["bucket"]: g["spending"] for g in a["groups"]}
        b_map = {g["bucket"]: g["spending"] for g in b["groups"]}
        for bucket in sorted(set(a_map) | set(b_map)):
            av = a_map.get(bucket, 0.0)
            bv = b_map.get(bucket, 0.0)
            diff = bv - av
            pct = (diff / av * 100) if av else None
            deltas.append({
                "bucket": bucket,
                "period_a": round(av, 2),
                "period_b": round(bv, 2),
                "diff": round(diff, 2),
                "pct_change": round(pct, 1) if pct is not None else None,
            })
        deltas.sort(key=lambda d: abs(d["diff"]), reverse=True)

    return {
        "period_a": a,
        "period_b": b,
        "total_diff": round(total_diff, 2),
        "total_pct_change": round(total_pct, 1) if total_pct is not None else None,
        "deltas_by_group": deltas,
    }


# ───────────────────────────────────────── 5. detect_pattern ────────────────


class DetectPatternArgs(BaseModel):
    """Ready-made behavioural patterns. Cheaper than reasoning over raw aggregates."""

    pattern: Literal["late_night_share", "weekend_vs_weekday", "time_of_day_distribution"]
    category: Literal[
        "coffee", "groceries", "restaurants", "delivery", "transport",
        "entertainment", "shopping", "health", "subscriptions",
        "utilities", "credit_payment", "travel",
    ] | None = Field(
        None,
        description="Restrict pattern to one category (e.g. 'delivery' for late-night delivery share).",
    )
    start_date: str | None = None
    end_date: str | None = None

    @field_validator("start_date", "end_date")
    @classmethod
    def _v_date(cls, v):
        return _iso_date(v) if v else v


def detect_pattern(args: DetectPatternArgs) -> dict:
    where, params = _build_filters(
        args.start_date, args.end_date, args.category, None, None, None, None,
    )
    # always exclude salary
    where = f"{where} AND category != 'salary'" if where else "WHERE category != 'salary'"

    if args.pattern == "late_night_share":
        sql = (
            "SELECT "
            "SUM(CASE WHEN hour >= 21 OR hour < 5 THEN 1 ELSE 0 END) AS late_count, "
            "COUNT(*) AS total_count, "
            "SUM(CASE WHEN hour >= 21 OR hour < 5 THEN -amount ELSE 0 END) AS late_spending, "
            "SUM(-amount) AS total_spending "
            f"FROM transactions {where}"
        )
        r = get_db().execute(sql, params).fetchone()
        total = r["total_count"] or 0
        share = (r["late_count"] / total) if total else 0.0
        return {
            "pattern": "late_night_share",
            "category": args.category,
            "late_count": r["late_count"] or 0,
            "total_count": total,
            "late_share_pct": round(share * 100, 1),
            "late_spending": round(r["late_spending"] or 0.0, 2),
            "total_spending": round(r["total_spending"] or 0.0, 2),
        }

    if args.pattern == "weekend_vs_weekday":
        sql = (
            "SELECT is_weekend, "
            "COUNT(*) AS txn_count, "
            "AVG(-amount) AS avg_per_txn, "
            "SUM(-amount) AS spending "
            f"FROM transactions {where} GROUP BY is_weekend"
        )
        rows = get_db().execute(sql, params).fetchall()
        result = {"weekday": None, "weekend": None}
        for r in rows:
            key = "weekend" if r["is_weekend"] else "weekday"
            result[key] = {
                "txn_count": r["txn_count"],
                "avg_per_txn": round(r["avg_per_txn"], 2),
                "spending": round(r["spending"], 2),
            }
        # ratio of avg amounts
        ratio = None
        if result["weekday"] and result["weekend"] and result["weekday"]["avg_per_txn"]:
            ratio = result["weekend"]["avg_per_txn"] / result["weekday"]["avg_per_txn"]
        return {
            "pattern": "weekend_vs_weekday",
            "category": args.category,
            "weekday": result["weekday"],
            "weekend": result["weekend"],
            "weekend_avg_to_weekday_ratio": round(ratio, 2) if ratio else None,
        }

    # time_of_day_distribution
    bucket = _time_of_day_bucket_sql()
    sql = (
        f"SELECT {bucket} AS bucket, "
        "COUNT(*) AS txn_count, "
        "SUM(-amount) AS spending "
        f"FROM transactions {where} GROUP BY bucket"
    )
    rows = get_db().execute(sql, params).fetchall()
    total = sum(r["txn_count"] for r in rows) or 1
    buckets = [
        {
            "bucket": r["bucket"],
            "txn_count": r["txn_count"],
            "spending": round(r["spending"], 2),
            "share_pct": round(r["txn_count"] / total * 100, 1),
        }
        for r in rows
    ]
    return {
        "pattern": "time_of_day_distribution",
        "category": args.category,
        "buckets": buckets,
    }


# ───────────────────────────────────────── 6. get_recent_transactions ───────


class GetRecentTransactionsArgs(BaseModel):
    """Most recent transactions, optionally on a specific account/card.
    Used for fraud follow-up: 'show me recent activity on this card'."""

    n: int = Field(5, ge=1, le=50)
    account: Literal["main_debit", "credit_card"] | None = None


def get_recent_transactions(args: GetRecentTransactionsArgs) -> dict:
    where, params = _build_filters(
        None, None, None, None, args.account, None, None,
    )
    sql = (
        "SELECT date, merchant, amount, category, account "
        f"FROM transactions {where} "
        "ORDER BY date DESC LIMIT ?"
    )
    rows = get_db().execute(sql, [*params, args.n]).fetchall()
    return {
        "account": args.account,
        "count": len(rows),
        "transactions": [
            {
                "date": r["date"],
                "merchant": r["merchant"],
                "amount": round(r["amount"], 2),
                "category": r["category"],
                "account": r["account"],
            }
            for r in rows
        ],
    }


# ───────────────────────────────────────── registry ─────────────────────────


TOOL_REGISTRY: dict[str, tuple[type[BaseModel], Callable[[BaseModel], dict], str]] = {
    "query_transactions": (
        QueryTransactionsArgs, query_transactions,
        "List raw transactions with filters (date range, category, merchant, account, amount). Use when you need individual rows, not aggregates.",
    ),
    "aggregate_spending": (
        AggregateSpendingArgs, aggregate_spending,
        "Sum / average spending grouped by category, merchant, month, weekday, is_weekend, or time_of_day. Use for 'how much did I spend on X' or 'top N categories'.",
    ),
    "find_recurring_subscriptions": (
        FindRecurringArgs, find_recurring_subscriptions,
        "List recurring (subscription-like) merchants with last_seen and a 'is_stale' flag for forgotten subscriptions.",
    ),
    "compare_periods": (
        ComparePeriodsArgs, compare_periods,
        "Compare totals between two periods with per-group delta and percentage change.",
    ),
    "detect_pattern": (
        DetectPatternArgs, detect_pattern,
        "Compute ready-made behavioural patterns: late_night_share, weekend_vs_weekday, time_of_day_distribution.",
    ),
    "get_recent_transactions": (
        GetRecentTransactionsArgs, get_recent_transactions,
        "Most recent N transactions, optionally restricted to one account/card. Used for fraud follow-up.",
    ),
}


def _clean_schema(schema: dict) -> dict:
    """Remove fields the OpenAI tools API doesn't accept gracefully."""
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return schema


def get_openai_tools() -> list[dict]:
    """Return the tool list in OpenAI Chat Completions ``tools`` format."""
    tools = []
    for name, (schema_cls, _fn, description) in TOOL_REGISTRY.items():
        params = _clean_schema(schema_cls.model_json_schema())
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": params,
            },
        })
    return tools


def execute_tool(name: str, arguments: dict | str) -> dict:
    """Dispatch a tool call from the LLM. ``arguments`` may be raw JSON string."""
    if name not in TOOL_REGISTRY:
        return {"error": f"unknown_tool: {name}"}
    schema_cls, fn, _desc = TOOL_REGISTRY[name]
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError as e:
            return {"error": f"invalid_json: {e}"}
    try:
        parsed = schema_cls(**arguments)
    except Exception as e:
        return {"error": f"invalid_arguments: {e}"}
    try:
        return fn(parsed)
    except Exception as e:
        return {"error": f"tool_execution_failed: {e}"}


if __name__ == "__main__":
    # Smoke test from CLI
    out = execute_tool("aggregate_spending", {"group_by": "category", "top_n": 5})
    print(json.dumps(out, indent=2, ensure_ascii=False))
