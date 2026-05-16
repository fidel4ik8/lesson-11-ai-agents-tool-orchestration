"""SQLite-backed data layer for the Personal Finance Coach.

Loads the CSV dataset once per process into an in-memory SQLite database
and exposes a read-only connection through ``get_db()``.

The schema enriches each transaction with derived columns (``hour``,
``weekday``, ``is_weekend``, ``year``, ``month``) so that tools can answer
time-of-day / weekday questions without parsing the ISO timestamp.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "starter" / "data" / "transactions.csv"

_DB: sqlite3.Connection | None = None


SCHEMA = """
CREATE TABLE transactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT    NOT NULL,
    date_only   TEXT    NOT NULL,
    hour        INTEGER NOT NULL,
    weekday     INTEGER NOT NULL,
    is_weekend  INTEGER NOT NULL,
    year        INTEGER NOT NULL,
    month       INTEGER NOT NULL,
    merchant    TEXT    NOT NULL,
    amount      REAL    NOT NULL,
    currency    TEXT    NOT NULL,
    category    TEXT    NOT NULL,
    account     TEXT    NOT NULL,
    recurring   INTEGER NOT NULL
);
CREATE INDEX idx_tx_date     ON transactions (date_only);
CREATE INDEX idx_tx_category ON transactions (category);
CREATE INDEX idx_tx_merchant ON transactions (merchant);
CREATE INDEX idx_tx_account  ON transactions (account);
"""


def _load_dataframe() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    ts = pd.to_datetime(df["date"], format="ISO8601")
    df["date_only"] = ts.dt.strftime("%Y-%m-%d")
    df["hour"] = ts.dt.hour.astype(int)
    df["weekday"] = ts.dt.weekday.astype(int)  # Monday=0
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["year"] = ts.dt.year.astype(int)
    df["month"] = ts.dt.month.astype(int)
    df["recurring"] = df["recurring"].astype(bool).astype(int)
    df["amount"] = df["amount"].astype(float)
    return df


def _build_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)

    df = _load_dataframe()
    cols = [
        "date", "date_only", "hour", "weekday", "is_weekend",
        "year", "month", "merchant", "amount", "currency",
        "category", "account", "recurring",
    ]
    rows = list(df[cols].itertuples(index=False, name=None))
    placeholders = ",".join(["?"] * len(cols))
    conn.executemany(
        f"INSERT INTO transactions ({','.join(cols)}) VALUES ({placeholders})",
        rows,
    )
    conn.commit()
    return conn


def get_db() -> sqlite3.Connection:
    """Return the singleton SQLite connection, building it on first call."""
    global _DB
    if _DB is None:
        _DB = _build_db()
    return _DB


def reset_db() -> None:
    """Tear down the cached connection (used in tests)."""
    global _DB
    if _DB is not None:
        _DB.close()
        _DB = None


def db_summary() -> dict:
    """Quick sanity-check summary used by CLI / tests."""
    conn = get_db()
    cur = conn.cursor()

    total = cur.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    first_date, last_date = cur.execute(
        "SELECT MIN(date_only), MAX(date_only) FROM transactions"
    ).fetchone()
    categories = [
        row[0] for row in cur.execute(
            "SELECT DISTINCT category FROM transactions ORDER BY category"
        ).fetchall()
    ]
    accounts = [
        row[0] for row in cur.execute(
            "SELECT DISTINCT account FROM transactions ORDER BY account"
        ).fetchall()
    ]
    salary_sum, expense_sum = cur.execute(
        "SELECT "
        "SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END), "
        "SUM(CASE WHEN amount < 0 THEN amount ELSE 0 END) "
        "FROM transactions"
    ).fetchone()
    return {
        "total_rows": total,
        "date_range": [first_date, last_date],
        "categories": categories,
        "accounts": accounts,
        "total_income": round(salary_sum or 0.0, 2),
        "total_expense": round(expense_sum or 0.0, 2),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(db_summary(), indent=2, ensure_ascii=False))
