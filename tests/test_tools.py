"""Tests for tools — they validate the SAME patterns the agents will rely on
when answering the example queries from homework/README.md.

If these tests fail, the golden set evaluation will also fail.
"""

from __future__ import annotations

import pytest

from src.db import reset_db
from src.tools import execute_tool, get_openai_tools, TOOL_REGISTRY


@pytest.fixture(autouse=True)
def _clean_db():
    reset_db()
    yield
    reset_db()


# ───────────────────────────────────────── schema / registry ────────────────


def test_all_tools_registered():
    expected = {
        "query_transactions", "aggregate_spending", "find_recurring_subscriptions",
        "compare_periods", "detect_pattern", "get_recent_transactions",
    }
    assert set(TOOL_REGISTRY) == expected


def test_openai_tools_shape():
    tools = get_openai_tools()
    assert len(tools) == 6
    for t in tools:
        assert t["type"] == "function"
        assert "name" in t["function"]
        assert "description" in t["function"]
        assert t["function"]["parameters"]["type"] == "object"


def test_unknown_tool_returns_error():
    res = execute_tool("foobar", {})
    assert "error" in res and "unknown_tool" in res["error"]


def test_invalid_arguments_return_error():
    res = execute_tool("aggregate_spending", {"group_by": "not_a_real_dimension"})
    assert "error" in res and "invalid_arguments" in res["error"]


def test_invalid_date_returns_error():
    res = execute_tool("query_transactions", {"start_date": "not-a-date"})
    assert "error" in res


# ───────────────────────────────────────── query_transactions ───────────────


def test_query_transactions_filter_by_merchant():
    res = execute_tool("query_transactions", {"merchant": "Netflix"})
    assert res["count"] > 0
    for tx in res["transactions"]:
        assert tx["merchant"] == "Netflix"
        assert tx["category"] == "subscriptions"


def test_query_transactions_date_range():
    res = execute_tool("query_transactions", {
        "start_date": "2025-06-01",
        "end_date": "2025-06-30",
        "category": "coffee",
        "limit": 200,
    })
    assert res["count"] > 0
    for tx in res["transactions"]:
        assert tx["date"].startswith("2025-06")
        assert tx["category"] == "coffee"


def test_query_transactions_sort_amount_desc():
    res = execute_tool("query_transactions", {
        "category": "shopping",
        "sort": "amount_desc",
        "limit": 5,
    })
    amounts = [abs(tx["amount"]) for tx in res["transactions"]]
    assert amounts == sorted(amounts, reverse=True)


# ───────────────────────────────────────── aggregate_spending ───────────────


def test_aggregate_spending_top_categories():
    res = execute_tool("aggregate_spending", {"group_by": "category", "top_n": 5})
    buckets = [g["bucket"] for g in res["groups"]]
    assert "salary" not in buckets  # excluded by default
    assert len(res["groups"]) == 5
    # sorted desc by spending
    spendings = [g["spending"] for g in res["groups"]]
    assert spendings == sorted(spendings, reverse=True)


def test_aggregate_spending_coffee_per_month():
    res = execute_tool("aggregate_spending", {
        "group_by": "month",
        "category": "coffee",
        "top_n": 20,
    })
    assert res["total_spending"] > 0
    # 12 months in dataset, should have ~12 buckets
    assert len(res["groups"]) >= 10


def test_aggregate_spending_weekend_vs_weekday():
    res = execute_tool("aggregate_spending", {"group_by": "is_weekend"})
    buckets = {g["bucket"]: g for g in res["groups"]}
    assert {"weekday", "weekend"} == set(buckets.keys())


def test_aggregate_spending_include_income():
    res = execute_tool("aggregate_spending", {
        "group_by": "category",
        "include_income": True,
        "top_n": 20,
    })
    buckets = [g["bucket"] for g in res["groups"]]
    assert "salary" in buckets


# ───────────────────────────────────────── find_recurring_subscriptions ─────


def test_find_recurring_finds_sportlife_as_stale():
    res = execute_tool("find_recurring_subscriptions", {"stale_after_days": 60})
    subs = {s["merchant"]: s for s in res["subscriptions"]}
    assert "Sportlife" in subs, "Sportlife should appear as a recurring subscription"
    assert subs["Sportlife"]["is_stale"] is True
    assert subs["Sportlife"]["days_since_last"] >= 60


def test_find_recurring_lists_active_subs():
    res = execute_tool("find_recurring_subscriptions", {})
    merchants = {s["merchant"] for s in res["subscriptions"]}
    # Common subs should appear (Netflix, Spotify, etc.)
    assert len(merchants) >= 3


# ───────────────────────────────────────── compare_periods ──────────────────


def test_compare_periods_first_vs_last_quarter():
    res = execute_tool("compare_periods", {
        "period_a_start": "2025-01-01",
        "period_a_end": "2025-03-31",
        "period_b_start": "2025-09-01",
        "period_b_end": "2025-11-30",
        "group_by": "category",
    })
    assert res["period_a"]["total_spending"] > 0
    assert res["period_b"]["total_spending"] > 0
    # diff has correct sign relationship
    expected_diff = res["period_b"]["total_spending"] - res["period_a"]["total_spending"]
    assert abs(res["total_diff"] - expected_diff) < 0.01
    assert len(res["deltas_by_group"]) > 0


def test_compare_periods_no_group():
    res = execute_tool("compare_periods", {
        "period_a_start": "2025-06-01",
        "period_a_end": "2025-06-30",
        "period_b_start": "2025-07-01",
        "period_b_end": "2025-07-31",
        "group_by": "none",
    })
    assert res["deltas_by_group"] == []
    assert res["total_pct_change"] is not None


# ───────────────────────────────────────── detect_pattern ───────────────────


def test_late_night_delivery_share_matches_readme():
    res = execute_tool("detect_pattern", {
        "pattern": "late_night_share",
        "category": "delivery",
    })
    # README says ~50% of delivery is after 21:00
    assert 35 <= res["late_share_pct"] <= 65


def test_weekend_vs_weekday_avg_higher():
    res = execute_tool("detect_pattern", {"pattern": "weekend_vs_weekday"})
    assert res["weekend"]["avg_per_txn"] > res["weekday"]["avg_per_txn"]
    # ~50% higher per README
    assert res["weekend_avg_to_weekday_ratio"] >= 1.2


def test_time_of_day_distribution_sums_to_100():
    res = execute_tool("detect_pattern", {"pattern": "time_of_day_distribution"})
    total_share = sum(b["share_pct"] for b in res["buckets"])
    assert 99.5 <= total_share <= 100.5


# ───────────────────────────────────────── get_recent_transactions ──────────


def test_get_recent_transactions_credit_card():
    # Dataset has exactly 2 credit_card txns (Booking.com + AliExpress fraud cases)
    res = execute_tool("get_recent_transactions", {"account": "credit_card", "n": 10})
    assert res["count"] == 2
    merchants = {tx["merchant"] for tx in res["transactions"]}
    assert merchants == {"Booking.com", "AliExpress"}
    for tx in res["transactions"]:
        assert tx["account"] == "credit_card"


def test_get_recent_transactions_main_debit_default():
    res = execute_tool("get_recent_transactions", {"n": 5})
    assert res["count"] == 5
    dates = [tx["date"] for tx in res["transactions"]]
    assert dates == sorted(dates, reverse=True)
