"""Baseline single-agent: one LLM + the full tool set + a tool-call loop.

This is the control group we compare the multi-agent crew against. No
framework — just the OpenAI Chat Completions API.
"""

from __future__ import annotations

import json
import time
from typing import Any

from src.llm import (
    AnswerResult,
    LANGFUSE_ENABLED,
    MODEL_SMART,
    ModelCall,
    ToolCallTrace,
    compute_cost_usd,
    get_client,
)
from src.tools import DATASET_FIRST_DATE, DATASET_LAST_DATE, TODAY, execute_tool, get_openai_tools

if LANGFUSE_ENABLED:
    from langfuse import observe
else:
    def observe(*args, **kwargs):
        # no-op decorator when Langfuse is disabled
        def _wrap(fn):
            return fn
        return _wrap if args and not callable(args[0]) else (args[0] if args else _wrap)

MAX_ITERATIONS = 8

SYSTEM_PROMPT = f"""\
Ти — AI-помічник з особистих фінансів у мобільному банківському застосунку.
Стиль: дружній, на "ти", без менторства. У стресових темах (борги, fraud) —
емпатично і коротко.

КОНТЕКСТ ДАНИХ
- Сьогодні: {TODAY}. Усі відносні дати ("минулого тижня", "цього місяця")
  інтерпретуй відносно цієї дати.
- Доступна історія транзакцій: {DATASET_FIRST_DATE} … {DATASET_LAST_DATE}.
- Валюта: USD. Витрати у даних — від'ємні, salary — додатні. Tools повертають
  spending як додатне число.
- Категорії: coffee, groceries, restaurants, delivery, transport,
  entertainment, shopping, health, subscriptions, utilities, salary,
  credit_payment, travel.
- Рахунки: main_debit, credit_card.

ЯК ВІДПОВІДАТИ
1. Будь-яке число у відповіді ПОВИННО походити з виклику tool. Не вигадуй.
   Не округлюй до зручних чисел — використовуй те, що повернув tool.
2. Викликай стільки tools, скільки треба для обґрунтованої відповіді
   (зазвичай 1-3). Для multi-step запитів комбінуй tools.
3. Поради повинні бути actionable і базуватися на конкретних цифрах
   (напр. "$180 на Glovo, з яких 60% після 21:00"). Уникай загальних
   рекомендацій типу "consider reducing dining out".
4. Відповіді стислі. 1-4 коротких речення для статистики; 3-5 пунктів
   для порад.

ESCALATION (fraud / підозрілі транзакції)
Якщо користувач каже, що НЕ робив транзакцію, або підозрює fraud —
НЕ намагайся розв'язати самостійно. Ввічливо поясни що:
  1. блокування картки робиться в розділі Картки → Заблокувати,
  2. dispute / chargeback оформлює служба підтримки через чат у застосунку.
Можеш запропонувати показати останні транзакції по тій картці.

OUT OF SCOPE
Покупки активів, кредитні пропозиції, переказ грошей, прогнози ринку —
поза твоїми можливостями. Ввічливо відмов і нагадай, з чим можеш допомогти
(статистика, поради по економії, аналіз підписок).

MULTI-TURN
Якщо запит коротка дописка ("а минулого місяця?", "чому так багато?") —
інтерпретуй його у контексті попередньої відповіді.
"""


def _serialize_message_for_history(msg) -> dict:
    """Convert openai ChatCompletionMessage → plain dict for the messages list."""
    out: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return out


@observe(name="baseline_answer")
def answer(query: str, history: list[dict] | None = None) -> AnswerResult:
    """Run the baseline agent for one user turn.

    ``history`` is the user/assistant conversation so far (no tool messages).
    The returned ``AnswerResult.answer`` is the new assistant turn — append it
    to history yourself before the next call.
    """
    client = get_client()
    tools = get_openai_tools()

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})

    tool_traces: list[ToolCallTrace] = []
    model_calls: list[ModelCall] = []
    final_text = ""

    overall_start = time.perf_counter()
    for _ in range(MAX_ITERATIONS):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=MODEL_SMART,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.3,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)

        usage = resp.usage
        cost = compute_cost_usd(MODEL_SMART, usage.prompt_tokens, usage.completion_tokens)
        model_calls.append(ModelCall(
            model=MODEL_SMART,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            latency_ms=latency_ms,
            agent="baseline",
            cost_usd=cost,
        ))

        msg = resp.choices[0].message
        messages.append(_serialize_message_for_history(msg))

        if not msg.tool_calls:
            final_text = msg.content or ""
            break

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_start = time.perf_counter()
            result = execute_tool(tool_name, args)
            tool_latency = int((time.perf_counter() - tool_start) * 1000)
            tool_traces.append(ToolCallTrace(
                tool=tool_name, args=args, result=result, latency_ms=tool_latency,
            ))
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })
    else:
        # Hit MAX_ITERATIONS without a finish — best-effort fallback
        final_text = (
            "Не зміг сформувати остаточну відповідь у межах ліміту викликів tools. "
            "Спробуй переформулювати запит."
        )

    overall_latency = int((time.perf_counter() - overall_start) * 1000)

    return AnswerResult(
        architecture="baseline",
        answer=final_text,
        tool_calls=tool_traces,
        model_calls=model_calls,
        latency_ms=overall_latency,
        metadata={"iterations": len(model_calls)},
    )


if __name__ == "__main__":
    # CLI smoke test
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "Скільки я витратив на каву минулого тижня?"
    res = answer(q)
    print("\n=== ANSWER ===")
    print(res.answer)
    print("\n=== TRACE ===")
    for tc in res.tool_calls:
        print(f"  → {tc.tool}({tc.args}) [{tc.latency_ms}ms]")
    print(f"\nlatency={res.latency_ms}ms  tokens={res.total_tokens}  "
          f"cost=${res.total_cost_usd:.5f}  iterations={res.metadata['iterations']}")
