"""
Генератор синтетичних транзакцій для homework Personal Finance Coach.

Генерує ~12 місяців історії з реалістичними патернами:
- Salary раз на місяць (25-го числа)
- Регулярні підписки (Netflix, Spotify, Apple One, Sportlife) — Sportlife
  навмисно "забута" 4 місяці тому (forgotten subscription pattern)
- Weekend spending spike (~30% більше у Sat/Sun)
- Coffee — щодня в будні зранку
- Groceries — 2-3 рази на тиждень
- Restaurants/delivery — 60% замовлень після 21:00 (impulse pattern)
- Utilities — раз на місяць
- 1-2 fraud-like великі транзакції в Booking.com/AliExpress

Запуск: python data/generate.py
Вивід: data/transactions.csv
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

random.seed(42)

OUT_PATH = Path(__file__).parent / "transactions.csv"
START_DATE = date(2024, 12, 1)
END_DATE = date(2025, 11, 30)
CURRENCY = "USD"
ACCOUNT_DEBIT = "main_debit"
ACCOUNT_CREDIT = "credit_card"


@dataclass
class Merchant:
    name: str
    category: str
    amount_range: tuple[float, float]
    weekday_bias: list[int] | None = None  # 0=Mon..6=Sun
    hour_bias: list[int] | None = None
    account: str = ACCOUNT_DEBIT
    recurring: bool = False


COFFEE = [
    Merchant("Aroma Kava", "coffee", (2.5, 4.5), weekday_bias=[0, 1, 2, 3, 4], hour_bias=[7, 8, 9]),
    Merchant("Lviv Croissants", "coffee", (3.0, 6.0), weekday_bias=[0, 1, 2, 3, 4], hour_bias=[8, 9, 10]),
    Merchant("Blue Bottle", "coffee", (4.0, 6.5), weekday_bias=[0, 1, 2, 3, 4], hour_bias=[8, 9]),
]

GROCERIES = [
    Merchant("ATB", "groceries", (15, 80)),
    Merchant("Silpo", "groceries", (20, 120)),
    Merchant("Novus", "groceries", (25, 100)),
    Merchant("Auchan", "groceries", (40, 180)),
]

RESTAURANTS = [
    Merchant("Puzata Hata", "restaurants", (8, 18)),
    Merchant("Kanapa", "restaurants", (25, 70)),
    Merchant("Kyivska Perepichka", "restaurants", (3, 6)),
]

DELIVERY = [
    Merchant("Glovo", "delivery", (12, 35), hour_bias=[12, 13, 19, 20, 21, 22, 23]),
    Merchant("Bolt Food", "delivery", (10, 30), hour_bias=[12, 13, 20, 21, 22, 23]),
    Merchant("Uber Eats", "delivery", (15, 40), hour_bias=[19, 20, 21, 22]),
]

TRANSPORT = [
    Merchant("Bolt", "transport", (3, 12), weekday_bias=[0, 1, 2, 3, 4]),
    Merchant("Uklon", "transport", (3, 10), weekday_bias=[0, 1, 2, 3, 4]),
    Merchant("KyivPass", "transport", (8, 8)),
]

ENTERTAINMENT = [
    Merchant("Cinema City", "entertainment", (8, 15)),
    Merchant("Steam", "entertainment", (10, 60)),
    Merchant("Bookstore", "entertainment", (5, 25)),
]

SUBSCRIPTIONS_ACTIVE = [
    Merchant("Netflix", "subscriptions", (12, 12), recurring=True),
    Merchant("Spotify", "subscriptions", (5, 5), recurring=True),
    Merchant("Apple One", "subscriptions", (15, 15), recurring=True),
    Merchant("iCloud Storage", "subscriptions", (3, 3), recurring=True),
]

SUBSCRIPTIONS_FORGOTTEN = [
    Merchant("Sportlife", "subscriptions", (15, 15), recurring=True),
]

UTILITIES = [
    Merchant("KyivEnergo", "utilities", (40, 90), recurring=True),
    Merchant("Vodokanal", "utilities", (15, 30), recurring=True),
    Merchant("Internet KyivStar", "utilities", (12, 12), recurring=True),
    Merchant("Mobile Lifecell", "utilities", (8, 8), recurring=True),
]

SHOPPING = [
    Merchant("Rozetka", "shopping", (20, 250)),
    Merchant("Comfy", "shopping", (40, 400)),
    Merchant("Zara", "shopping", (30, 180)),
    Merchant("H&M", "shopping", (25, 120)),
]

HEALTH = [
    Merchant("ANC Pharmacy", "health", (5, 50)),
    Merchant("Med Clinic", "health", (40, 200)),
]

SUSPICIOUS_FOREIGN = [
    Merchant("Booking.com", "travel", (300, 900), account=ACCOUNT_CREDIT),
    Merchant("AliExpress", "shopping", (150, 400), account=ACCOUNT_CREDIT),
]


def random_amount(low: float, high: float) -> float:
    if low == high:
        return low
    val = random.uniform(low, high)
    if low > 5:
        val = round(val * 2) / 2
    return round(val, 2)


def pick_hour(merchant: Merchant) -> int:
    if merchant.hour_bias:
        return random.choice(merchant.hour_bias)
    return random.randint(8, 22)


def is_weekend(d: date) -> bool:
    return d.weekday() >= 5


def transactions_for_day(d: date) -> list[dict]:
    txs: list[dict] = []
    weekend = is_weekend(d)

    if not weekend and random.random() < 0.85:
        m = random.choice(COFFEE)
        txs.append(make_tx(d, m, sign=-1))

    if random.random() < (0.45 if not weekend else 0.65):
        m = random.choice(GROCERIES)
        txs.append(make_tx(d, m, sign=-1))

    if random.random() < (0.20 if not weekend else 0.45):
        if random.random() < 0.6:
            m = random.choice(DELIVERY)
        else:
            m = random.choice(RESTAURANTS)
        txs.append(make_tx(d, m, sign=-1))

    if not weekend and random.random() < 0.55:
        m = random.choice(TRANSPORT)
        txs.append(make_tx(d, m, sign=-1))

    if random.random() < (0.05 if not weekend else 0.18):
        m = random.choice(ENTERTAINMENT)
        txs.append(make_tx(d, m, sign=-1))

    if random.random() < 0.10:
        m = random.choice(SHOPPING)
        txs.append(make_tx(d, m, sign=-1))

    if random.random() < 0.04:
        m = random.choice(HEALTH)
        txs.append(make_tx(d, m, sign=-1))

    return txs


def make_tx(d: date, merchant: Merchant, sign: int = -1, amount: float | None = None) -> dict:
    if amount is None:
        amount = random_amount(*merchant.amount_range)
    hour = pick_hour(merchant)
    minute = random.randint(0, 59)
    ts = datetime.combine(d, datetime.min.time()).replace(hour=hour, minute=minute)
    return {
        "date": ts.isoformat(timespec="minutes"),
        "merchant": merchant.name,
        "amount": round(sign * amount, 2),
        "currency": CURRENCY,
        "category": merchant.category,
        "account": merchant.account,
        "recurring": merchant.recurring,
    }


def add_monthly_recurring(txs: list[dict], start: date, end: date) -> None:
    forgotten_cutoff = end - timedelta(days=120)

    cur = date(start.year, start.month, 1)
    while cur <= end:
        salary_day = date(cur.year, cur.month, min(25, last_day_of_month(cur)))
        if start <= salary_day <= end:
            txs.append({
                "date": datetime.combine(salary_day, datetime.min.time()).replace(hour=10).isoformat(timespec="minutes"),
                "merchant": "ACME Corp Salary",
                "amount": 2400.00,
                "currency": CURRENCY,
                "category": "salary",
                "account": ACCOUNT_DEBIT,
                "recurring": True,
            })

        for m in SUBSCRIPTIONS_ACTIVE:
            day = date(cur.year, cur.month, random.randint(1, 28))
            if start <= day <= end:
                txs.append(make_tx(day, m, sign=-1))

        for m in SUBSCRIPTIONS_FORGOTTEN:
            day = date(cur.year, cur.month, random.randint(1, 28))
            if start <= day <= end and day <= forgotten_cutoff:
                txs.append(make_tx(day, m, sign=-1))

        for m in UTILITIES:
            day = date(cur.year, cur.month, random.randint(5, 15))
            if start <= day <= end:
                txs.append(make_tx(day, m, sign=-1))

        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)


def add_credit_card_payments(txs: list[dict], start: date, end: date) -> None:
    """Кредитна картка: minimum payment 4 з 6 місяців, full 2 з 6."""
    cur = date(start.year, start.month, 1)
    month_idx = 0
    while cur <= end:
        pay_day = date(cur.year, cur.month, min(14, last_day_of_month(cur)))
        if start <= pay_day <= end:
            is_minimum = month_idx % 3 != 2
            amount = 50.00 if is_minimum else random.uniform(380, 520)
            txs.append({
                "date": datetime.combine(pay_day, datetime.min.time()).replace(hour=14).isoformat(timespec="minutes"),
                "merchant": "Credit Card Payment",
                "amount": round(-amount, 2),
                "currency": CURRENCY,
                "category": "credit_payment",
                "account": ACCOUNT_DEBIT,
                "recurring": False,
            })
        month_idx += 1
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)


def add_suspicious(txs: list[dict], start: date, end: date) -> None:
    for _ in range(2):
        days = random.randint(0, (end - start).days)
        d = start + timedelta(days=days)
        m = random.choice(SUSPICIOUS_FOREIGN)
        txs.append(make_tx(d, m, sign=-1))


def last_day_of_month(d: date) -> int:
    if d.month == 12:
        return 31
    next_month = date(d.year, d.month + 1, 1)
    return (next_month - timedelta(days=1)).day


def main() -> None:
    txs: list[dict] = []

    cur = START_DATE
    while cur <= END_DATE:
        txs.extend(transactions_for_day(cur))
        cur += timedelta(days=1)

    add_monthly_recurring(txs, START_DATE, END_DATE)
    add_credit_card_payments(txs, START_DATE, END_DATE)
    add_suspicious(txs, START_DATE, END_DATE)

    txs.sort(key=lambda t: t["date"])

    fields = ["date", "merchant", "amount", "currency", "category", "account", "recurring"]
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for tx in txs:
            writer.writerow(tx)

    print(f"Generated {len(txs)} transactions → {OUT_PATH}")
    by_cat: dict[str, int] = {}
    for tx in txs:
        by_cat[tx["category"]] = by_cat.get(tx["category"], 0) + 1
    print("By category:")
    for cat, count in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
