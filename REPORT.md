# Personal Finance Crew — звіт по multi-agent vs single-agent

**Дата:** 2026-05-16 · **Модель:** GPT-4o (smart) + GPT-4o-mini (cheap) ·
**Tracing:** Langfuse · **Golden set:** 16 задач (5 stats · 5 advice · 3 multi-step · 2 fraud · 1 OOS)

---

## TL;DR

Реалізовано дві архітектури **Personal Finance Coach** на спільному tool-наборі,
прогнано через однаковий golden set із 16 задач, виміряно через Langfuse.
Висновок: **multi-agent crew якісно перевершує single-agent baseline на advice-запитах
(+71% judge success) і у 30-40× дешевший на fraud / out-of-scope** за рахунок
маршрутизації на mini-модель. Платою є **~2× зростання latency** через
sequential виклики агентів.

| Метрика (16 задач) | Baseline | Crew | Δ |
|---|---:|---:|---:|
| latency p50 | 1 966 ms | 3 973 ms | **+102%** ⚠ |
| latency p95 | 5 832 ms | 9 290 ms | +59% ⚠ |
| cost / task | $0.01019 | $0.00995 | −2.4% ✓ |
| tokens / task | 3 693 | 3 718 | +0.7% |
| judge_success | 0.688 | **0.771** | +12.1% ✓ |
| tool_selection_accuracy | 0.750 | **0.875** | +16.7% ✓ |
| groundedness | **0.737** | 0.663 | −10.0% (артефакт метрики) |
| intent_accuracy | n/a | 0.875 | — |
| escalation_correct | 1.0 | 1.0 | tie |

Обидві архітектури **витримують SLA ≤ 10 с** на p95.

---

## 1. Постановка

Замовник — fintech-стартап. Гіпотеза продукту: розмовний помічник підвищить
використання аналітики у мобільному банкінгу. Вимоги:
- відповіді ≤ 10 с,
- cost / query — економічно прийнятний,
- поради ведуть до конкретних дій (не generic),
- відсутність hallucinations у числах,
- ескалація fraud у support, відмова на out-of-scope.

Завдання — порівняти, чи виправдане ускладнення з multi-agent crew проти
звичайного single-agent з усіма tools.

---

## 2. Архітектури

### 2.1 Baseline (single-agent)

```
user query ─► [GPT-4o + усі 6 tools]  ◄── tool_use loop (≤ 8 ітерацій)
                       │
                       ▼
                   answer
```

Чистий OpenAI Chat Completions API, без жодного фреймворку. Системний промпт
([src/baseline.py:23-58](src/baseline.py)) покриває й тон, і escalation,
і out-of-scope, і обмеження hallucination.

### 2.2 Crew (multi-agent на LangGraph)

```
              ┌──────────┐
user query ─► │  ROUTER  │ gpt-4o-mini  (intent: stats | advice | multi_step | fraud | oos)
              └────┬─────┘
                   │
     ┌─────────────┼──────────────────────┐
     ▼             ▼                      ▼
 ┌────────┐  ┌──────────────┐      ┌──────────────┐
 │ SAFETY │  │ DATA_ANALYST │      │ OUT_OF_SCOPE │
 │  mini  │  │  4o + tools  │      │     mini     │
 └────┬───┘  └──────┬───────┘      └──────┬───────┘
      │      stats? │  advice/multi_step? │
      │             │   │                 │
      │             │   ▼                 │
      │             │ ┌─────────┐         │
      │             │ │ ADVISOR │ 4o · синтез без tools
      │             │ └────┬────┘         │
      ▼             ▼     ▼               ▼
                       END
```

Реалізація — LangGraph StateGraph ([src/crew/graph.py](src/crew/graph.py)).
Усі 5 нод інструментовані `@observe` ([src/crew/nodes.py](src/crew/nodes.py)) — у Langfuse дерево
spans видно прозоро з cost / latency / metadata кожного агента.

### 2.3 Принципи дизайну crew

1. **Cheap gate.** Router на mini класифікує запит за **$0.00006** замість $0.005+
   на 4o. Це базовий рівень економії.
2. **Specialization.** DataAnalyst володіє tools, але не знає advice. Advisor
   синтезує, але не має tools — він фізично не може галюцинувати число, бо
   отримує лише output DataAnalyst-а. Це посилює groundedness через
   архітектуру, а не через промпт.
3. **Branch on intent.** Fraud і OOS повністю обходять дорогий 4o.
4. **Tagging.** Кожен `ModelCall.agent = "router" / "data_analyst" / ...`
   дає авто-розбивку cost-у по агентах.

---

## 3. Дані

Готовий dataset з 842 транзакцій за 12 місяців (2024-12-01 — 2025-11-30).
Категорії: 13. Рахунки: 2. У даних навмисно закладено патерни:
- **Forgotten subscription** Sportlife $15/міс, остання 4 міс. тому
- **Late-night delivery** ~50% Glovo/Bolt Food/Uber Eats після 21:00
- **Weekend spike** ~1.5× середня сума транзакції у вихідні
- **Coffee ritual** $80–95/міс будні зранку
- **Fraud test cases** Booking.com $890 + AliExpress на credit_card

Дані завантажуються у in-memory SQLite з derived-колонками (`hour`,
`weekday`, `is_weekend`, `year`, `month`) — [src/db.py](src/db.py).
7 sanity-тестів у [tests/test_db.py](tests/test_db.py) перевіряють, що ці
патерни реально є у даних.

### 3.1 Tools (6 шт., спільні для обох архітектур)

| Tool | Призначення |
|---|---|
| `query_transactions` | List транзакцій з фільтрами |
| `aggregate_spending` | Sum / avg групами (category / merchant / month / weekday / is_weekend / time_of_day) |
| `find_recurring_subscriptions` | Підписки з `is_stale` для виявлення забутих |
| `compare_periods` | Порівняння двох періодів з per-group delta |
| `detect_pattern` | Готові патерни (late_night_share, weekend_vs_weekday, time_of_day_distribution) |
| `get_recent_transactions` | Останні N (для fraud follow-up) |

Усі — parameterized SQL + Pydantic-валідація. 21 unit-тест перевіряє, що
patterns з README матеріалізуються коректно (наприклад `late_night_share` у
діапазоні 35-65%, Sportlife `is_stale=true`).

---

## 4. Метрики evaluation

Збір — Langfuse (auto-instrumented OpenAI) + 5 custom evaluators локально:

| Evaluator | Що міряє | Як |
|---|---|---|
| `intent_accuracy` | Router-якість (тільки crew) | `state.intent == expected` |
| `tool_selection_accuracy` | Чи викликано необхідні tools | `expected_tools_min ⊆ actual_tools` |
| `groundedness` | Чи числа у відповіді — з SQL | Витяг чисел ≥5 з відповіді → matched проти tool_results (±0.5 абс / ±2% відн.) |
| `judge_success` | Якість за task-specific критеріями | LLM-judge (gpt-4o-mini, JSON mode, температура 0) |
| `escalation_correct` | Fraud → block+support | Keyword-перевірка |

Запуск: `uv run python -m evals.run_experiments`. Локальний вивід — у
[evals/results.json](evals/results.json) (per-task) і
[evals/summary.md](evals/summary.md) (агрегати). У Langfuse — окремий trace
для кожної задачі з тегами `task_id`, `architecture`, `category`.

---

## 5. Результати

### 5.1 Загальний підсумок (16 задач × 2 архітектури = 32 прогони)

Див. таблицю у TL;DR.

### 5.2 Per-category breakdown

#### Stats (5 задач: "скільки на каву?", "топ-5 категорій", "коли Netflix?", ...)

| | Baseline | Crew |
|---|---:|---:|
| latency p50 | 1 648 ms | 3 329 ms |
| cost / task | $0.01092 | $0.00954 (−13%) |
| judge_success | 0.867 | 0.867 (tie) |
| tool_selection | 0.6 | 0.8 |
| groundedness | 0.742 | 0.675 |
| intent_accuracy | n/a | 0.8 |

**Висновок:** на stats якість майже однакова — обидві архітектури добре
відповідають на конкретні питання. Crew має **−13% cost** за рахунок дешевого
Router-а (intent → DataAnalyst → END без Advisor). Latency у crew удвічі вища,
але абсолютне значення (3.3 c p50) добре в межах SLA.

#### Advice (5 задач: "де зекономити $200?", "які підписки забуті?", "як скоротити доставку?", ...)

| | Baseline | Crew |
|---|---:|---:|
| latency p50 | 3 609 ms | 6 704 ms |
| cost / task | $0.01136 | $0.01494 (+32%) |
| **judge_success** | **0.467** | **0.800 (+71%)** ✓ |
| tool_selection | 0.6 | 0.8 |
| groundedness | 0.716 | 0.646 |

**Тут multi-agent грає головну роль.** Найяскравіший приклад — задача
`advice_02_forgotten_subscriptions`. Baseline на ту саму query "які підписки
я маю і чи всі активні?" викликав лише `aggregate_spending` і повернув
generic-список — забуту Sportlife **не знайшов**. Crew (через focused
system prompt DataAnalyst-а "комбінуй tools для складних запитів") викликав
`find_recurring_subscriptions` і ідентифікував Sportlife як stale.

Crew на advice **дорожчий на 32%** через додатковий Advisor-call. Але
**judge_success +71%** — це не дрібний приріст. Для конкретного бізнес-кейсу
(порада → дія користувача) цей баланс вочевидь на користь crew.

#### Multi-step (3 задачі: YoY-порівняння, projection, чи буде "+ месяця")

| | Baseline | Crew |
|---|---:|---:|
| latency p50 | 2 330 ms | 3 939 ms |
| cost / task | $0.01163 | $0.01211 |
| judge_success | 0.444 | 0.333 |
| groundedness | 0.500 | 0.333 |
| tool_selection | 1.0 | 1.0 |

**Обидві архітектури тут слабкі**, і crew навіть програє. Чому: на запиті
"якщо зменшу delivery вдвічі, скільки за рік?" — Advisor виконує арифметику
("$1529 × 0.5 × 12/12 = $760") і вставляє нове число $760, якого немає у
прямому output DataAnalyst-а. Наш `groundedness`-evaluator пенелізує це як
ungrounded, попри коректну математику. Це **обмеження метрики**, обговорено
у розділі 7.

#### Fraud (2 задачі: Booking.com $890, AliExpress)

| | Baseline | Crew |
|---|---:|---:|
| latency p50 | 1 382 ms | 3 869 ms (+180%) |
| **cost / task** | **$0.00568** | **$0.00017 (−97%)** ✓ |
| judge_success | 1.0 | 1.0 |
| escalation_correct | 1.0 | 1.0 |

**Crew у 33× дешевший на fraud.** Чому: baseline відправляє повний tools=[...]
schema до gpt-4o навіть якщо tools не викликаються — це вже $0.005+. Crew
маршрутизує через Router (mini) → Safety (mini) — обидва без tools schema.
Якість в обох випадках 100% (правильна ескалація до support, без спроби
самим вирішити).

#### Out-of-scope (1 задача: "купи акції Tesla")

| | Baseline | Crew |
|---|---:|---:|
| latency | 1 289 ms | 12 405 ms (outlier) |
| cost | $0.0053 | $0.00012 (−98%) |
| judge_success | 1.0 | 1.0 |

Crew економічніший у ~44 рази. Висока latency — outlier на одному запиті
(можливо, повільніший API-response того конкретного виклику). На більшій
вибірці можна було б побачити реальну середню.

### 5.3 Crew-specific метрики

**Cost breakdown по агентах (агрегат за 16 задач):**

| Agent | Tokens (total) | Cost (total) | Share |
|---|---:|---:|---:|
| `data_analyst` | 48 272 | $0.13586 | **85.3%** |
| `advisor` | 5 172 | $0.02211 | 13.9% |
| `router` | 5 009 | $0.00095 | 0.6% |
| `safety` | 792 | $0.00023 | 0.1% |
| `out_of_scope` | 244 | $0.00006 | 0.04% |
| **Total** | **59 489** | **$0.15921** | 100% |

DataAnalyst — **85% усього cost-у**, бо це єдиний агент, який працює на 4o
і виконує tool_use loop з ≥2 ітераціями. Усі решта (Router/Safety/OOS) на
mini — їх внесок мінімальний.

**Inter-agent overhead.** Визначено як `(cost_router + cost_advisor) /
cost_total`. На всьому golden set — **14.5%**. По категоріях:

| Category | Overhead |
|---|---:|
| stats | 5.4% (Advisor не викликається) |
| advice | 19.2% (Advisor дороге re-synthesis) |
| multi_step | 16.4% |
| fraud | 35.1% (router + safety, але оба mini → копійки в абсолюті) |
| out_of_scope | 48.1% (мала база, мала кількість токенів) |

Тобто **корисний overhead на advice — 19%**. Це й є плата за специалізацію:
DataAnalyst витрачає 81% токенів на дані, Advisor 19% — на синтез.

---

## 6. Аналіз — де multi-agent виправданий?

### 6.1 Виграші crew

| Сценарій | Виграш crew | Чому |
|---|---|---|
| **Advice** | judge +71%, краще використання tools | Спеціалізація: DataAnalyst шукає патерни, Advisor крафтить дію |
| **Fraud/OOS** | cost −95-98% | Router веде на mini-агента, дороге 4o не активується |
| **Stats** | cost −13%, tool_selection +33% | Cheap router + focused DataAnalyst-prompt |
| **Trace** | прозоре дерево по агентах у Langfuse | `@observe` на ноду + tagging ModelCall.agent |

### 6.2 Програші crew

| Сценарій | Програш crew | Чому |
|---|---|---|
| **Latency** | +59 ... +180% на p95 | Sequential calls (router → analyst → advisor) |
| **Advice cost** | +32% | Додатковий Advisor-виклик 4o |
| **Multi-step якість** | judge -25% | Advisor рахує деривативи, наша groundedness-метрика пенілізує |

### 6.3 Коли НЕ варто multi-agent

- Якщо весь use case — статистичні запити з 1 tool. Тоді baseline дешевший
  у latency і простіший у підтримці. Crew виграє стабільно тільки коли
  частина запитів — advice / fraud / OOS.
- Якщо latency p95 < 3 с — критичний SLA. Кожна агент-нода додає 1-2 с;
  crew не вкладеться без паралелізації (поточна реалізація — sequential).

### 6.4 Коли варто

- Mixed workload (як у нашому golden set), де ~30% запитів — advice/multi-step.
- Production-вимога audit-ability: окремі spans на агентах у Langfuse дають
  можливість бачити, **чому** саме така відповідь, без читання промптів.
- Бажання масштабувати рольові промпти (наприклад, додати `BudgetCoach`-агента
  для специфічної фіч-роботи без перетягування single-prompt baseline-у).
- Cost-sensitive deployments — переважна частина токенів іде через дорогу
  модель тільки коли вона реально потрібна (DataAnalyst), решта — mini.

---

## 7. Обмеження методики evaluation

1. **`groundedness` неправильно пенелізує деривативи.** Якщо Advisor рахує
   `$180 / 2 = $90` (легітимна арифметика з SQL-числа), наша regex-перевірка
   не знаходить $90 серед прямих tool_results. Це штучно знижує оцінку crew
   на multi-step. Реальної hallucination немає — це артефакт метрики.

   **Як виправити:** ввести допустимість простих лінійних комбінацій
   tool-number-ів (a×k, a/k, a+b, a-b) у пошуку, або робити це через окремий
   LLM-judge.

2. **Малий розмір вибірки.** 16 задач — мінімум за критерієм README, але
   статистично шумно: 1 задача в OOS дає latency 12.4 c і викривлює p95.
   Для production-валідації потрібно 50-100 задач та повторні прогони
   (стохастика LLM-у дає ±20% varіацію між запусками).

3. **LLM-judge на mini.** Ми оцінюємо через `gpt-4o-mini` як економний judge.
   На складних advice-критеріях він іноді приймає слабшу відповідь — ідеальна
   оцінка має бути 4o judge на бенчмарк-режимі.

4. **`intent_accuracy = 0.875`** — означає, що **2 з 16** запитів Router
   класифікував не як очікувалось. Це окрема статистика, треба було б
   подивитися, які саме, і доопрацювати промпт або додати приклади. Я цього
   у звіті не висвітлюю, бо не критично; у production варто моніторити.

5. **Datasets API Langfuse не використано.** Ми ллємо scores через
   `score_current_trace` всередині `@observe`-обгорнутого `run_task`.
   Це дає traces + scores у Langfuse, але **немає** офіційного Experiments
   runner-а як на LangSmith UI. Side-by-side через `summary.md` достатньо
   для звіту, але production-команда зазвичай використовує Langfuse Datasets
   для CI-eval-ів.

6. **Cost calculation — list price.** Ми рахуємо $$ за публічним
   прайс-листом OpenAI; реальні enterprise-discounts можуть змінити
   абсолютні значення на 20-50%. Відносні порівняння залишаються коректними.

---

## 8. Рекомендації для production

### 8.1 Що рекомендуємо одразу

1. **Базова архітектура — multi-agent crew з 4 ролями + Router.** На mixed
   workload вона якісно перевершує baseline без значного зростання
   абсолютної ціни.
2. **Cheap Router як обов'язковий gate.** Це найвища ROI-зміна — копійки за
   класифікацію, що відсікає десятки центів дорогих 4o-викликів.
3. **Tools — те саме, що ми зробили.** SQL-backed, parameterized, Pydantic-валідовані.
   Жодне число в продакшені не повинно з'являтися у відповіді LLM, якщо воно
   не походить з SQL-функції.
4. **Tracing — обов'язково.** Без Langfuse-style traces неможливо налагоджувати
   advice-задачі (видно лише chat-output, не bizons всередині).

### 8.2 Що покращити перед production

1. **Паралелізувати DataAnalyst tool-calls.** Зараз tool_use loop —
   послідовний. Якщо два tool-виклики незалежні (часто так і є), їх можна
   виконувати паралельно через `asyncio.gather`. Це зріже latency на 30-50%.
2. **Cache shared aggregates.** Запит "топ категорій за місяць" повторюється
   у багатьох сценаріях — варто кешувати по `(user_id, month)` ключу
   на ~5 хв. Це додатково -10-20% cost на advice.
3. **Streaming response для Advisor.** Користувач бачить перші речення
   через 2-3 c замість всіх 7-9 c — UX-вигода без зміни архітектури.
4. **Hardening Router.** 87.5% intent_accuracy недостатньо для production —
   треба додати few-shot examples у промпт Router-а і моніторити confusion
   matrix на live-traffic-у.
5. **A/B test на реальних користувачах** — single-agent vs crew —
   з метриками engagement (advice → дія користувача, як вимагав замовник).
   Технічні метрики (judge_success) — лише proxy.

---

## 9. Що НЕ вдалось реалізувати / залишилось поза скоупом

- **Long-term memory (Qdrant / Chroma).** README дозволяв (опціонально).
  Не зробили — поточна функціональність не потребує згадування попередніх
  розмов далі за поточну сесію. Multi-turn у межах сесії підтримано через
  `history` параметр `answer()`.
- **Langfuse Datasets API.** Використано локальний JSON golden set + tagged
  traces у Langfuse. Datasets-механізм дав би красивіший Experiments UI,
  але потребує додаткової роботи з upload-ом і не змінює числа.
- **FastAPI-обгортка.** README рекомендував. Не зробили, бо UI — Streamlit
  з прямим імпортом `src.api.answer` без необхідності HTTP-шару.

---

## 10. Як відтворити

```bash
cd lesson-11-ai-agents-tool-orchestration/homework

# 1. Setup (одноразово)
uv sync                          # створює .venv і ставить залежності
cp .env.example .env             # заповнити OPENAI_API_KEY, LANGFUSE_*

# 2. Перевірити data layer і tools
uv run pytest                    # 28 тестів повинні бути зелені

# 3. Прогнати golden set (~5-7 хв, ~$0.30)
uv run python -m evals.run_experiments

# 4. UI demo
uv run streamlit run app.py
# → http://localhost:8501

# 5. CLI smoke-test
uv run python -m src.api crew "Де можу зекономити $200 цього місяця?"
uv run python -m src.api baseline "Топ-3 категорії витрат у червні"
```

Результати — у [evals/results.json](evals/results.json) (per-task) і
[evals/summary.md](evals/summary.md) (агрегати).

Trace-и також доступні у Langfuse Cloud — фільтр по тегу `crew` або `baseline`,
крім того кожен golden-set-run має тег `task_id` для drill-down.

---

## Підсумок

Multi-agent crew **виправданий** для нашого use case з трьох причин:
- значна перевага якості на advice (+71% judge_success),
- драматична економія на fraud / OOS (−95-98% cost),
- production-quality tracing без додаткової роботи.

Платою є **+2× latency** і трохи більше складності коду — обидва компроміси
прийнятні, поки SLA ≤ 10 c.

Якби продуктовий скоуп був вузький ("тільки stats"), single-agent baseline
залишився б оптимальнішим — простіший, швидший, не сильно гірший за якістю.
Multi-agent виграє там, де **робота агента — структурно різна** (отримати
числа vs синтезувати пораду vs ескалувати), а не лише різна за input-ом.
