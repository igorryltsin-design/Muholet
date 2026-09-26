"""Финальная подпись сцены: в конце промаха — вердикт и CPA, а не живая дистанция.

На жалобу «какая-то глупость: там должно быть „промах, минимальное расстояние
цель—ракета“» разведка показала пару дырок в одном условии. Жёлтая подпись над
серединой отрезка «ракета—цель» — это расстояние В ДАННЫЙ МОМЕНТ; она гасилась
только по терминальному событию кадра (`промах`/`miss_pass`), а дуэль «цель ушла»
по истечении боевой жизни никакого события не даёт (см. первый тест). Итог: после
боя подпись доживала до следующего пуска и показывала последний кадр сближения —
на экране модели сливаются, и честное «0 м» выглядит насмешкой над телеметрией,
которая в тот же момент даёт 0,37 км и CPA 319 м.

Лечение: сцена знает итог прогона (вердикт + наименьшее сближение) и в конце
промаха прячет живую дистанцию, показывая «промах · наименьшее N м». Проверяем и
причину (события нет), и что прежняя залипшая подпись не вернулась, и что прогон,
снятый «Стопом», не оставляет в сцене чужого вердикта.
"""

from __future__ import annotations

from pathlib import Path

from navedenie.engine import collect
from navedenie.sim import Scenario

REPO = Path(__file__).resolve().parents[1]
SCENE = (REPO / "web" / "src" / "EngagementView.tsx").read_text(encoding="utf-8")
APP = (REPO / "web" / "src" / "App.tsx").read_text(encoding="utf-8")
FLIGHT = (REPO / "web" / "src" / "shell" / "FlightWorkspace.tsx").read_text(encoding="utf-8")
DUEL = (REPO / "web" / "src" / "shell" / "DuelWorkspace.tsx").read_text(encoding="utf-8")
HELP = (REPO / "web" / "src" / "HelpView.tsx").read_text(encoding="utf-8")

# «мухиные» скорости и слабая ракета: цель уходит, БЧ не звонит
ESCAPE = dict(
    mode="pn", law="pn", v_m=300, v_t=290, n_max=4, pn_n=4, dt=0.02,
    t_max=40, kill_radius_m=45, range_m=6000, n_target=8.0,
)


def test_fuse_expired_duel_ends_without_a_terminal_event() -> None:
    """Причина правки: бой «цель ушла» кончается по боевой жизни молча — в последнем
    кадре события нет, и по одному лишь кадру сцена промах не узнает."""
    res = collect(Scenario(**{**ESCAPE, "duel": True, "evader_law": "away", "fuse_life_s": 6.0}))
    assert res.reason == "fuse_expired" and not res.hit
    assert res.frames, "нужен хотя бы один кадр"
    assert res.frames[-1].event is None
    # минимум по дискретным кадрам КРУПНЕЕ непрерывного CPA (между кадрами пара
    # сходится теснее) — значит цифру для подписи надо брать из метрик, как вердикт
    assert res.frames[-1].miss > res.cpa_m


def test_scene_paints_cpa_verdict_at_the_end_of_a_miss() -> None:
    """Финал промаха: надпись «промах · наименьшее N м», источник — CPA прогона,
    а не расстояние последнего кадра."""
    for marker in (
        "промах · наименьшее ",
        "const endedNoKill = pb",
        "outcomeRef.current !== null && !outcomeRef.current.hit",
        "missLabel.spr.visible = missedEnd",
        "bestHit = best.hit",
    ):
        assert marker in SCENE, f"EngagementView.tsx: нет маркера {marker!r}"
    # CPA живого прогона — из метрик, та же цифра, что в вердикте; кадр только как запасной
    assert "typeof missM === 'function' ? missM() : missM" in APP
    assert "m.hit, m.miss" in APP


def test_sticky_live_distance_cannot_outlive_the_run() -> None:
    """Живая дистанция обязана гаснуть в финале промаха: прежнее условие показа
    (только терминальное событие) возвращает залипшее «0 м»."""
    assert "const showDist = !hit && !missedEnd && sepRaw < 3" in SCENE
    # прежняя короткая подпись и прежний узкий критерий конца боя
    assert "промах ${Math.round(fr.miss)} м" not in SCENE
    assert "missLabel.spr.visible = missedEnd && fr !== null" not in SCENE
    # синтетический кадр обучения — не проигрыш: вердикт рисует только живой прогон
    assert "!idleRef.current" in SCENE


def test_outcome_is_wired_to_both_flying_scenes() -> None:
    """Итог прогона доходит и до «Пуска», и до дуэльного космоса; в «Рое» его нет —
    там бой показывает веер траекторий (ветка `pb`)."""
    assert "outcome={runOutcome}" in APP
    assert "outcome: { hit: boolean; missM: number } | null" in FLIGHT
    assert "outcome={outcome}" in FLIGHT
    assert "outcome: { hit: boolean; missM: number } | null" in DUEL
    assert "outcome={outcome}" in DUEL


def test_new_run_clears_the_previous_verdict() -> None:
    """Вердикт прошлого боя не переживает новый пуск: итог сбрасывается на каждом
    старте — в «Пуске», в дуэльном цикле, в рое и в обучении. Пишется же он только
    вместе со сводкой, то есть после досчитанного прогона, а не на его середине."""
    assert APP.count("setRunOutcome(null)") >= 4, "итог чистится не на всех стартах прогона"
    assert "setDone(sum)\n      // сцене нужен именно итог" in APP


def test_help_says_which_number_is_which() -> None:
    """Справка различает две цифры: наименьшее сближение за бой и расстояние
    в данный момент — иначе «0 м» снова выглядит поломкой физики."""
    for marker in ("промах · наименьшее", "расстояние <b>в данный момент</b>", "из вердикта прогона"):
        assert marker in HELP, f"HelpView.tsx: нет маркера {marker!r}"
