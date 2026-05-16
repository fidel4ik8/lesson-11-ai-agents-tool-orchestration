# Personal Finance Crew — Quickstart

Multi-agent vs single-agent Personal Finance Coach. Постановка задачі — у [README.md](README.md).

## Setup

```bash
cd lesson-11-ai-agents-tool-orchestration/homework
uv sync                          # створить .venv і поставить залежності
cp .env.example .env             # заповнити OPENAI_API_KEY, LANGFUSE_*
```

## Запуск

```bash
# UI
uv run streamlit run app.py

# Golden set evaluation
uv run python evals/run_experiments.py

# Tests
uv run pytest
```

## Архітектура

- `src/db.py` — CSV → SQLite, query layer
- `src/tools.py` — 6 tools (query_transactions, aggregate_spending, find_recurring_subscriptions, detect_pattern, compare_periods, get_recent_transactions)
- `src/baseline.py` — single-agent (один LLM + усі tools)
- `src/crew/` — multi-agent на LangGraph (Router, DataAnalyst, Advisor, Safety, OutOfScope)
- `src/api.py` — спільний `answer(query, history, arch)` контракт
- `evals/` — golden set + custom evaluators (groundedness, tool_selection, judge LLM)
- `app.py` — Streamlit demo
- `REPORT.md` — фінальний звіт з метриками
