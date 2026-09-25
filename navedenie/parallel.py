"""Параллельный счёт боёв для генетических циклов (школа, королева, ринг).

Каждый бой — чистая функция (сценарий + геномы → метрики исхода), а их в
поколении десятки. Раньше они крутились подряд в единственном процессе стенда:
тяжёлое поколение школы намертво занимало GIL, и «Пуск» интерактивного прогона
ждал своей очереди минуты. Пул процессов решает обе беды: бои летят на всех
ядрах, а процесс стенда остаётся свободным для запусков и стрима.

Детерминизм не тронут: map сохраняет порядок задач, бой не читает глобальное
состояние. Под pytest и при MUHOLET_WORKERS=0/1 — последовательный откат
(тесты не плодят дочерние процессы, численный результат тот же)."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import Executor, ProcessPoolExecutor
from typing import Any

_executor: Executor | None = None
_pool_broken = False


def worker_count() -> int:
    env = os.environ.get("MUHOLET_WORKERS", "").strip()
    if env:
        try:
            return max(0, int(env))
        except ValueError:
            pass
    return max(2, min(8, (os.cpu_count() or 2) - 1))


def _executor_or_none() -> Executor | None:
    """Ленивый общий пул; None — считать последовательно."""
    global _executor, _pool_broken
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    if worker_count() < 2 or _pool_broken:
        return None
    if _executor is None:
        try:
            _executor = ProcessPoolExecutor(max_workers=worker_count())
        except Exception:
            _pool_broken = True
            return None
    return _executor


def _star(args):
    fn, job = args
    return fn(*job)


def parallel_map(fn: Callable[..., Any], jobs: Iterable[tuple]) -> list[Any]:
    """fn(*job) по каждому кортежу задач; порядок совпадает с порядком jobs."""
    jobs = list(jobs)
    ex = _executor_or_none() if len(jobs) > 1 else None
    if ex is None:
        return [fn(*job) for job in jobs]
    return list(ex.map(_star, ((fn, job) for job in jobs), chunksize=1))
