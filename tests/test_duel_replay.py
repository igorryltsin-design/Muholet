"""Показ учебной стрельбы: стенд отдаёт кадры боя поколения (`replay`).

Смысл витка: «Дуэль» стала вкладкой про то, как учатся две мухи, — значит учебный
бой должен быть виден, а не только посчитан. Проверяем, что `replay` (а) появляется
только по флагу, (б) не трогает отбор бит-в-бит, (в) несёт траектории обеих сторон и
честный момент начала манёвра, (г) остаётся маленьким (≤300 точек), и (д) что фронтовые
литералы новой вкладки зеркалятся тестами, как остальной интерфейс.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from navedenie.app import EvaderGenIn, QueenGenIn
from navedenie.evader_train import (
    REPLAY_MAX_PTS,
    init_evader_population,
    replay_school,
    train_generation,
)
from navedenie.redqueen import init_brain_population, queen_generation
from navedenie.sim import Scenario
from navedenie.swarm import fly_from_json

FAST = dict(dt=0.02, t_max=10, aspect="head-on")
REPO = Path(__file__).resolve().parents[1]


def _base() -> Scenario:
    # перегрузка цели nonzero: иначе уклонист неманёвренный по определению
    return Scenario(**FAST, n_target=8.0, duel=True, fuse_life_s=30.0)


def test_school_replay_only_on_flag_and_off_selection() -> None:
    """`replay=True` добавляет поле и не двигает отбор: следующее поколение,
    фитнеси и статистика — бит-в-бит те же, что без флага."""
    pop = init_evader_population(4, seed=7)
    with_flag = train_generation(_base(), pop, gen=1, seed=7, replay=True)
    without = train_generation(_base(), pop, gen=1, seed=7, replay=False)
    assert "replay" in with_flag and "replay" not in without
    assert with_flag["next_population"] == without["next_population"]
    assert with_flag["stats"] == without["stats"]
    assert with_flag["results"] == without["results"]


def test_school_replay_carries_both_trajectories() -> None:
    r = train_generation(_base(), init_evader_population(4, seed=7), gen=0, seed=7, replay=True)["replay"]
    assert r["traj_m"] and r["traj_t"]
    assert len(r["traj_m"]) == len(r["traj_t"])  # пары точек не разъезжаются
    assert len(r["traj_m"]) <= REPLAY_MAX_PTS
    assert all(len(p) == 3 for p in r["traj_m"])
    assert r["hit"] in (True, False) and r["n_target_g"] == 8.0
    assert r["scenario"]["aspect"] in ("head-on", "beam", "tail-chase")
    assert "школа · поколение 0" in r["label"]


def test_queen_replay_is_one_fight_of_champions() -> None:
    """У королевы показ — бой чемпионов той же геометрии поколения; обе фитнеси
    того же боя, что берёт отбор (медианы круга могут отличаться, сама пара — нет)."""
    out = queen_generation(
        _base(), init_brain_population(2, seed=7), init_brain_population(2, seed=8), gen=1, seed=7, replay=True
    )
    plain = queen_generation(
        _base(), init_brain_population(2, seed=7), init_brain_population(2, seed=8), gen=1, seed=7
    )
    assert "replay" not in plain
    r = out["replay"]
    assert "королева · поколение 1" in r["label"]
    assert len(r["traj_m"]) == len(r["traj_t"]) and 0 < len(r["traj_m"]) <= REPLAY_MAX_PTS
    assert {"missile_fitness", "evader_fitness"} <= set(r)
    assert out["champions"] == plain["champions"]


def test_turn_field_honest_about_unmaneuverable_target() -> None:
    """При n_target = 0 цель неманёвренна по определению — поля манёвра нет; если
    манёвр найден, его время и дальность лежат внутри боя."""
    still = replay_school(_base(), init_evader_population(2, seed=7)[0], 0)
    zero = replay_school(replace(_base(), n_target=0.0), init_evader_population(2, seed=7)[0], 0)
    assert zero["turn"] is None and zero["n_target_g"] == 0.0
    if still["turn"] is not None:
        assert 0.0 <= still["turn"]["t_s"] <= still["t_end"]
        assert still["turn"]["range_m"] > 0.0


def test_endpoints_take_the_flag() -> None:
    """Флаг живёт в моделях входа: без тела ответ остаётся прежним — старые
    потребители (и тесты формы) ничего не замечают."""
    assert EvaderGenIn().replay is False
    assert QueenGenIn().replay is False
    src = (REPO / "navedenie" / "app.py").read_text(encoding="utf-8")
    assert "replay=bool(body.replay)" in src


def test_background_campaign_exposes_replay() -> None:
    """Фоновая кампания обязана иметь видимый аналог: статус отдаёт последний
    показ поколения, а `_shape` (из него пишется хроника) — только числа, чтобы
    траектории не оседали в data/queen_chronicle.json."""
    from navedenie import queen_train

    assert "replay" not in queen_train._shape(None)
    assert queen_train.status()["replay"] is None
    src = (REPO / "navedenie" / "queen_train.py").read_text(encoding="utf-8")
    assert "replay=True" in src  # цикл поколения воюет с кадрами


def test_tab_has_two_layers_and_one_learning_cycle() -> None:
    """Перегруженность убрана прятанием, а не вычёркиванием: у вкладки два слоя
    («Учёба»/«Дуэли»), цикл поколений один с диспетчером режимов, второстепенное
    сложено в `<details>`. Литералы — те же, что держат остальные тесты фронта."""
    ui = (REPO / "web" / "src" / "shell" / "DuelWorkspace.tsx").read_text(encoding="utf-8")
    for marker in (
        "Учёба",
        "Дуэли",
        "Две мухи учатся",
        "кто учится",
        "className=\"ws-fold\"",
        "muholet-duel",  # слой и тумблер переживают перезагрузку
    ):
        assert marker in ui, f"DuelWorkspace.tsx: нет маркера {marker!r}"
    # три кнопки запуска стали одной: карточки режимов сами ничего не начинают
    for gone in ("onClick={runSchool}", "onClick={runQueen}", "onClick={() => void srvStart()}"):
        assert gone not in ui, f"DuelWorkspace.tsx: цикл {gone!r} завёлся второй раз"
    for marker in ("const startLearning = () => {", "const stopLearning = () => {", "onClick={startLearning}", "onClick={stopLearning}"):
        assert marker in ui, f"DuelWorkspace.tsx: нет диспетчера цикла {marker!r}"


def test_maneuver_is_explained_not_hidden() -> None:
    """Честный ответ на «почему они летят друг на друга»: геометрия боя видна
    полоской, неманёвренная цель подписана прямо, а момент поворота цели считается
    из показанного боя, а не берётся из воздуха."""
    ui = (REPO / "web" / "src" / "shell" / "DuelWorkspace.tsx").read_text(encoding="utf-8")
    for marker in (
        "неманёвренный по определению",
        "манёвр начался: t = ",
        "с, до цели ",
        "GEO_PRESETS",
        "ASPECT_LABEL[sc.aspect]",
        "showLesson(d.replay)",
        "if (busyRef.current) return",  # учебный бой не перекрывает живой прогон
    ):
        assert marker in ui, f"DuelWorkspace.tsx: нет маркера {marker!r}"
    app = (REPO / "web" / "src" / "App.tsx").read_text(encoding="utf-8")
    for marker in ("onShowReplay={showDuelReplay}", "const showDuelReplay = (r: DuelReplay)", "caption: r.label"):
        assert marker in app, f"App.tsx: нет проводки показа {marker!r}"
