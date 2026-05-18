"""
Скрипт оценки качества системы для дипломной работы.
Запуск: python run_evaluation.py
"""

import asyncio
import json
import time
import httpx
from pathlib import Path

ACCESSORIES_URL = "http://localhost:8000"
CORE_URL = "http://localhost:8001"
CAR_IMAGE = Path(__file__).parent / "mitsubishi-attrage-2020-s.jpg"

# ─── Тестовые запросы для оценки семантического поиска ───────────────────────
# (запрос, ожидаемое ключевое слово в name/category найденной детали)
SEARCH_QUERIES = [
    ("spoiler",                   "spoiler"),
    ("rear wing carbon spoiler",  "spoiler"),
    ("gt wing",                   "spoiler"),
    ("антикрыло спойлер",         "spoiler"),   # русский → английский каталог
    ("wheels rims sport",         "wheel"),
    ("sport rims",                "wheel"),
    ("volk racing",               "volk"),
    ("диски колёса",              "wheel"),     # русский → английский каталог
    ("headlights audi",           "headlight"),
    ("фары ауди",                 "headlight"), # русский → английский каталог
    ("тонировка стёкла",          None),        # нет в каталоге
    ("межпланетный корабль",      None),        # явно нерелевантно
]

# ─── Тестовые сценарии для пайплайна агента ──────────────────────────────────
PIPELINE_SCENARIOS = [
    {
        "prompt": "поставь карбоновый спойлер на крышу",
        "expected_tool": "segment_object",
    },
    {
        "prompt": "смени диски на спортивные",
        "expected_tool": "segment_object",
    },
    {
        "prompt": "установи передний губ-спойлер",
        "expected_tool": "segment_object",
    },
]


async def test_semantic_search(client: httpx.AsyncClient) -> dict:
    """Precision@1 и Precision@3 для семантического поиска."""
    print("\n── Семантический поиск ─────────────────────────────────────────")
    results = []

    for query, expected_keyword in SEARCH_QUERIES:
        t0 = time.perf_counter()
        r = await client.post(
            f"{ACCESSORIES_URL}/api/parts/search",
            json={"query": query, "limit": 3},
        )
        elapsed = (time.perf_counter() - t0) * 1000

        if r.status_code != 200:
            print(f"  [{query:30s}]  HTTP {r.status_code}")
            results.append({"query": query, "p1": 0, "p3": 0, "latency_ms": elapsed, "top": []})
            continue

        body = r.json()
        items = body.get("items", body) if isinstance(body, dict) else body
        names = [it.get("name", "").lower() for it in items]
        cats  = [it.get("category", "").lower() for it in items]
        scores = [it.get("score", 0) for it in items]

        if expected_keyword is None:
            # Ожидаем пустой результат — правильно если ничего не нашлось
            p1 = 1 if len(items) == 0 else 0
            p3 = p1
            label = "OK (пусто)" if p1 else f"ЛИШНЕЕ: {names[:3]}"
        else:
            kw = expected_keyword.lower()
            # проверяем и в name, и в category
            def hit(i): return kw in names[i] or kw in cats[i]
            p1 = 1 if names and hit(0) else 0
            p3 = 1 if any(hit(i) for i in range(min(3, len(names)))) else 0
            label = f"P@1={'✓' if p1 else '✗'}  P@3={'✓' if p3 else '✗'}  top={names[:3]}  scores={[round(s,3) for s in scores[:3]]}"

        print(f"  [{query:35s}]  {elapsed:5.0f}ms  {label}")
        results.append({"query": query, "p1": p1, "p3": p3, "latency_ms": elapsed, "top": names[:3]})

    relevant = [r for r in results if SEARCH_QUERIES[results.index(r)][1] is not None]
    p1_avg = sum(r["p1"] for r in relevant) / len(relevant) if relevant else 0
    p3_avg = sum(r["p3"] for r in relevant) / len(relevant) if relevant else 0
    avg_lat = sum(r["latency_ms"] for r in results) / len(results)

    print(f"\n  Precision@1 = {p1_avg:.2f}  |  Precision@3 = {p3_avg:.2f}  |  Avg latency = {avg_lat:.0f}ms")
    return {"precision_at_1": p1_avg, "precision_at_3": p3_avg, "avg_latency_ms": avg_lat, "details": results}


async def test_pipeline_timing(client: httpx.AsyncClient) -> dict:
    """Замер времени каждого шага пайплайна."""
    print("\n── Тайминг пайплайна ───────────────────────────────────────────")

    scenario = PIPELINE_SCENARIOS[0]
    image_bytes = CAR_IMAGE.read_bytes()

    # Создать сессию
    t_start = time.perf_counter()
    r = await client.post(
        f"{CORE_URL}/api/agent/sessions",
        files={"images": ("car.jpg", image_bytes, "image/jpeg")},
        data={"user_prompt": scenario["prompt"]},
        timeout=30,
    )
    t_create = (time.perf_counter() - t_start) * 1000

    if r.status_code not in (200, 201, 202):
        print(f"  Создание сессии: HTTP {r.status_code} — {r.text[:200]}")
        return {}

    session_id = r.json().get("session_id") or r.json().get("id")
    print(f"  Сессия создана: {session_id}  ({t_create:.0f}ms)")

    # Поллинг до завершения с замером времени шагов
    step_times = {}
    t_poll_start = time.perf_counter()
    deadline = t_poll_start + 300  # 5 минут максимум

    prev_event_count = 0
    step_start_times = {}

    while time.perf_counter() < deadline:
        await asyncio.sleep(1)
        r = await client.get(f"{CORE_URL}/api/agent/sessions/{session_id}", timeout=10)
        if r.status_code != 200:
            continue

        data = r.json()
        status = data.get("status")
        events = data.get("events") or []

        # Фиксируем время появления новых событий
        for i, ev in enumerate(events[prev_event_count:], prev_event_count):
            ev_type = ev.get("type", "")
            tool = ev.get("tool", "")
            t_now = time.perf_counter() - t_poll_start

            if ev_type == "tool_call":
                step_start_times[tool] = t_now
                print(f"  → tool_call: {tool}  (t={t_now:.1f}s)")
            elif ev_type == "tool_result" and tool in step_start_times:
                elapsed = t_now - step_start_times[tool]
                step_times[tool] = elapsed
                print(f"  ← tool_result: {tool}  ({elapsed:.1f}s)")

        prev_event_count = len(events)

        if status in ("complete", "failed"):
            total = time.perf_counter() - t_poll_start
            print(f"\n  Статус: {status}  |  Всего: {total:.1f}s")
            result_urls = data.get("result_image_urls") or []
            print(f"  Результатов: {len(result_urls)}")
            return {
                "status": status,
                "total_s": round(total, 1),
                "step_times": {k: round(v, 1) for k, v in step_times.items()},
                "result_count": len(result_urls),
                "session_id": session_id,
            }

    print("  Таймаут ожидания!")
    return {"status": "timeout"}


async def test_parts_catalog(client: httpx.AsyncClient) -> dict:
    """Сколько деталей в каталоге, сколько готово."""
    print("\n── Каталог деталей ─────────────────────────────────────────────")
    r = await client.get(f"{ACCESSORIES_URL}/api/parts/", timeout=10)
    if r.status_code != 200:
        print(f"  HTTP {r.status_code}")
        return {}

    body = r.json()
    parts = body.get("items", body) if isinstance(body, dict) else body
    total = len(parts)
    ready = sum(1 for p in parts if p.get("status") == "ready")
    pending = sum(1 for p in parts if p.get("status") == "pending")
    failed = sum(1 for p in parts if p.get("status") == "failed")

    print(f"  Всего деталей: {total}  |  ready: {ready}  |  pending: {pending}  |  failed: {failed}")

    categories = {}
    for p in parts:
        cat = p.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    print(f"  Категории: {categories}")

    return {"total": total, "ready": ready, "pending": pending, "failed": failed, "categories": categories}


async def main():
    print("=" * 60)
    print("  Оценка системы AI-визуализации")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=60) as client:
        # Проверка доступности
        try:
            await client.get(f"{ACCESSORIES_URL}/api/parts/", timeout=5)
        except Exception as e:
            print(f"\n❌ accessories_processing недоступен на {ACCESSORIES_URL}: {e}")
            return
        try:
            await client.get(f"{CORE_URL}/api/agent/sessions/test", timeout=5)
        except httpx.HTTPStatusError:
            pass  # 404 — это нормально, сервис доступен
        except Exception as e:
            print(f"\n❌ core_service недоступен на {CORE_URL}: {e}")
            return

        catalog = await test_parts_catalog(client)
        search = await test_semantic_search(client)
        timing = await test_pipeline_timing(client)

    # Сохранить результаты
    results = {
        "catalog": catalog,
        "search": search,
        "pipeline": timing,
    }
    out = Path(__file__).parent / "evaluation_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n✓ Результаты сохранены в {out}")


if __name__ == "__main__":
    asyncio.run(main())
