"""Потоковый прогон «Пуск без паузы»: collect-коллбэк on_frame и WS-стрим
/api/ws/run (кадры уходят по мере счёта, метрики — финальным 'done').

Проверки: on_frame виден на каждом шаге и его keep-критерий совпадает с
отбором в frames (без коллбэка collect бит-в-бит прежний); 'done' стрима и
POST /api/run отдают ОДИН ответ (_run_answer); форма — общий словарь.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from navedenie.engine import collect, frame_to_dict
from navedenie.app import ScenarioIn, _run_answer, _sc, run_once
from navedenie.sim import Scenario

REPO = Path(__file__).resolve().parents[1]

FAST = dict(dt=0.05, t_max=12, aspect="head-on")  # короткий честный прогон


def _sc_fast() -> Scenario:
    return Scenario(**FAST)


def test_on_frame_sees_every_step_and_agrees_with_keep() -> None:
    """on_frame(i, fr, keep): вызов строго по одному на шаг генератора
    (включая терминальный hit-кадр перед break), keep == правда тогда и только
    тогда, когда кадр попал в result.frames."""
    sc = _sc_fast()
    seen: list[tuple[int, float, bool]] = []
    res = collect(
        sc, stride=max(1, int(0.04 / sc.dt)),
        on_frame=lambda i, fr, keep: seen.append((i, fr.t, keep)),
    )
    assert seen, "коллбэк не вызывался"
    assert [i for i, _, _ in seen] == list(range(len(seen))), "пропущены шаги"
    kept_t = {round(fr.t, 6) for fr in res.frames}
    assert {round(t, 6) for _, t, k in seen if k} == kept_t
    # каждый шаг либо keep, либо нет, но число вызовов = число шагов прогона
    assert len(seen) >= len(kept_t)


def test_collect_without_callback_bitwise_same() -> None:
    """Регрессия: collect без on_frame — метрики и кадры совпадают с версией
    с коллбэком (он ничего не меняет в счёте)."""
    sc = _sc_fast()
    a = collect(sc, stride=4)
    b = collect(sc, stride=4, on_frame=lambda i, fr, keep: None)
    assert a.hit == b.hit and a.reason == b.reason
    assert a.miss_m == b.miss_m and a.n_int == b.n_int
    assert len(a.frames) == len(b.frames)
    assert np.allclose(a.frames[-1].missile, b.frames[-1].missile)


def test_run_answer_shared_by_post_and_ws_done() -> None:
    """'done' стрима — тот же словарь метрик, что ответ POST (общий
    _run_answer, frames в done пусты): сверка на одном seed-прогоне."""
    body = ScenarioIn(**FAST)
    post = run_once(body)
    res = collect(_sc(body), stride=max(1, int(0.04 / _sc(body).dt)))
    answer = _run_answer(res, [])
    assert answer["frames"] == []
    for k in ("hit", "miss_m", "duel", "duel_result", "t_survived", "n_peak", "reason"):
        assert answer[k] == post[k], k


def test_ws_stream_wired() -> None:
    """Проводка маркерами: ws_run считает через collect+on_frame в потоке-
    экзекьюторе и шлёт 'frame'/'done'/'error'; фронт стартует игру по первым
    кадрам (12) с фолбэком на прежний POST."""
    src = (REPO / "navedenie" / "app.py").read_text(encoding="utf-8")
    for marker in ('on_frame=on_frame', 'run_in_executor(None, compute)', 'call_soon_threadsafe', '"type": "done"'):
        assert marker in src, f"app.py: нет маркера {marker!r}"
    app = (REPO / "web" / "src" / "App.tsx").read_text(encoding="utf-8")
    for marker in ("/api/ws/run", "frames.length >= 12", "done: () => finished", "streamRun("):
        assert marker in app, f"App.tsx: нет маркера {marker!r}"
    assert "'api/ws/run'" not in app  # адрес собирается из location, не хардкодом
    assert "await fetch('/api/run'" in app  # фолбэк на прежний путь жив


def test_final_voice_follows_metrics_not_summary_prefix() -> None:
    """Финальная реплика мухи — по метрике m.hit, а не по началу текста сводки:
    в дуэли сводка начинается с «Ракета взяла», и текстовый тест давал «промах»
    при взятии цели. Проводка маркерами: ленивый hit в playFrames, m.hit в обеих
    нестримовых точках вызова, hitFinal в стриме."""
    app = (REPO / "web" / "src" / "App.tsx").read_text(encoding="utf-8")
    assert "hit?: boolean | null | (() => boolean | null)" in app  # параметр playFrames
    assert "got ?? sum.startsWith('Перехват')" in app  # голос: метрика, текст — лишь фолбэк
    assert "hitFinal = m.hit" in app  # стрим-путь: честный hit из 'done'
    # POST и офлайн-фолбэк несут m.hit (виток 16: после hit добавился wendy-план,
    # поэтому POST-вызов разбит построчно и «undefined, m.hit, …)» в нём не встречается;
    # виток 21: рядом поехал m.miss — та же цифра CPA нужна финальной подписи сцены)
    assert "undefined, m.hit, m.miss)" in app  # офлайн-фолбэк: одной строкой
    assert "undefined,\n          m.hit," in app  # POST: та же пара аргументов
    for f in ("web/src/App.tsx", "web/src/shell/FlightWorkspace.tsx"):
        src = (REPO / f).read_text(encoding="utf-8")
        assert "startsWith('Ракета')" in src, f  # статус-окраска вердикта дуэли
