"""Sanity tests for the SQLite data layer.

These tests assert that the patterns documented in homework/README.md
actually exist in the generated dataset.  If they fail, the golden set
evaluation will also fail.
"""

from __future__ import annotations

import pytest

from src.db import db_summary, get_db, reset_db


@pytest.fixture(autouse=True)
def _clean_db():
    reset_db()
    yield
    reset_db()


def test_dataset_shape():
    summary = db_summary()
    assert summary["total_rows"] == 842
    assert summary["date_range"] == ["2024-12-01", "2025-11-30"]
    assert "salary" in summary["categories"]
    assert set(summary["accounts"]) == {"credit_card", "main_debit"}


def test_salary_is_positive_and_other_categories_are_negative():
    conn = get_db()
    pos_count = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE category='salary' AND amount <= 0"
    ).fetchone()[0]
    neg_count = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE category != 'salary' AND amount > 0"
    ).fetchone()[0]
    assert pos_count == 0, "salary must always be positive"
    assert neg_count == 0, "non-salary categories must always be negative"


def test_forgotten_sportlife_subscription():
    """Sportlife should appear in the data but have its last txn months ago."""
    conn = get_db()
    rows = conn.execute(
        "SELECT date_only FROM transactions WHERE merchant='Sportlife' ORDER BY date_only DESC"
    ).fetchall()
    assert len(rows) >= 3, "Sportlife should appear at least a few months"
    last_seen = rows[0]["date_only"]
    # dataset ends 2025-11-30; last Sportlife should be at least ~3 months earlier
    assert last_seen < "2025-09-01", f"Sportlife last_seen={last_seen} not stale enough"


def test_late_night_delivery_pattern():
    """About half of delivery txns should be after 21:00."""
    conn = get_db()
    total = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE category='delivery'"
    ).fetchone()[0]
    late = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE category='delivery' AND hour >= 21"
    ).fetchone()[0]
    share = late / total
    assert 0.35 <= share <= 0.65, f"late-night delivery share={share:.2%} out of expected range"


def test_weekend_spike():
    """Average expense per weekend transaction should be noticeably higher than weekdays."""
    conn = get_db()
    weekday_avg = conn.execute(
        "SELECT AVG(amount) FROM transactions WHERE is_weekend=0 AND category != 'salary'"
    ).fetchone()[0]
    weekend_avg = conn.execute(
        "SELECT AVG(amount) FROM transactions WHERE is_weekend=1 AND category != 'salary'"
    ).fetchone()[0]
    # amounts are negative, so "higher spending" means more negative
    assert weekend_avg < weekday_avg, "weekend transactions should be larger on average"


def test_fraud_test_cases_present():
    """Booking.com and AliExpress on credit_card — used to test fraud escalation."""
    conn = get_db()
    booking = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE merchant='Booking.com' AND account='credit_card'"
    ).fetchone()[0]
    aliexpress = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE merchant='AliExpress' AND account='credit_card'"
    ).fetchone()[0]
    assert booking >= 1
    assert aliexpress >= 1


def test_coffee_monthly_spend_in_range():
    """Coffee ritual: $80-95/month per README. Average across full months."""
    conn = get_db()
    row = conn.execute(
        "SELECT AVG(monthly) FROM ("
        "  SELECT year, month, SUM(-amount) AS monthly "
        "  FROM transactions WHERE category='coffee' "
        "  GROUP BY year, month"
        ")"
    ).fetchone()
    avg_monthly = row[0]
    assert 60 <= avg_monthly <= 120, f"avg coffee/month=${avg_monthly:.2f} outside expected range"
