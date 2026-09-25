"""Самообучение на сервере: фоновый цикл Красной королевы (`navedenie/queen_train.py`).

Проверки: снимок имеет форму до всякой задачи, фоновый цикл совпадает с ручным
поколение в поколение (тот же seed → бит-в-бит те же чемпионы), на стенде живёт
только одна гонка (busy → 409), по готовности чемпионы уходят в мозг и на полку
ринга, остановка досчитывает текущее поколение и сворачивается, проводка маркерами.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from navedenie import brain_store, circuit, queen_train
from navedenie.app import (
    QueenTrainIn,
    ScenarioIn,
    queen_train_endpoint,
    queen_train_status_endpoint,
)
from navedenie.circuit import FEAT_DIM
from navedenie.queen_ring import RING_FILE
from navedenie.redqueen import init_brain_population, queen_generation
from navedenie.sim import Scenario
from navedenie.swarm import fly_from_json

REPO = Path(__file__).resolve().parents[1]
FAST = dict(dt=0.02, t_max=10, aspect="head-on")

SHAPE = {
    "running",
    "generations",
    "generations_done",
    "pop",
    "seed",
    "scenario",
    "inherit",
    "auto_ring",
    "save_duel",
    "apply",
    "log",
    "champions",
    "saved",
    "error",
    "stop_requested",
    "started_at",
    "finished_at",
    "seconds",
}


def _base() -> Scenario:
    return Scenario(**FAST)


def _digest(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()


@pytest.fixture(autouse=True)
def _one_job_at_a_time():
    """Фоновый поток переживает тест — гасим его и до, и после, иначе следующая
    проверка упрётся в busy, а запись весов придёт уже в другой песочнице."""
    queen_train.stop()
    queen_train.wait(180)
    yield
    queen_train.stop()
    queen_train.wait(180)
    queen_train._JOB = None
    queen_train._THREAD = None


@pytest.fixture
def _restore_brains():
    prev_stub = brain_store._circuits.get("stub")
    prev_evader = brain_store._evader
    yield
    if prev_stub is None:
        brain_store._circuits.pop("stub", None)
    else:
        brain_store._circuits["stub"] = prev_stub
    brain_store._evader = prev_evader


def test_status_has_shape_before_any_job() -> None:
    """Форма ответа постоянна и до первой задачи: фронту не надо различать
    «войны ещё не было» и «война кончилась»."""
    st = queen_train.status()
    assert set(st) == SHAPE
    assert st["running"] is False and st["generations_done"] == 0 and st["champions"] is None
    assert st["saved"] == {"ring": None, "weights": None}


def test_background_loop_matches_manual_loop() -> None:
    """Фон крутит ТОТ ЖЕ цикл, что фронт руками: те же seed/поколения → фитнеси
    поколение в поколение и чемпионы бит-в-бит. Иначе «война в фоне» была бы
    другим экспериментом, а не тем же."""
    mp = init_brain_population(2, seed=7)
    ep = init_brain_population(2, seed=8)
    manual = []
    for g in range(2):
        out = queen_generation(_base(), mp, ep, gen=g, seed=7)
        mp = [fly_from_json(x) for x in out["missile_population"]]
        ep = [fly_from_json(x) for x in out["evader_population"]]
        manual.append(out)

    queen_train.start(_base(), generations=2, pop=2, seed=7, save_duel=False, apply=False)
    done = queen_train.wait(300)
    assert done["error"] is None
    assert done["generations_done"] == 2 and done["running"] is False
    assert [r["missile_best"] for r in done["log"]] == [m["stats"]["missile_best"] for m in manual]
    assert [r["evader_best"] for r in done["log"]] == [m["stats"]["evader_best"] for m in manual]
    assert _digest(done["champions"]) == _digest(manual[-1]["champions"])
    # флаги выключены — ни весов, ни полок не появилось
    assert done["saved"] == {"ring": None, "weights": None}


def test_only_one_queen_fights() -> None:
    """Вторая задача не подменяет первую: busy и в модуле, и 409 в эндпоинте."""
    queen_train.start(_base(), generations=30, pop=2, seed=7, save_duel=False, apply=False)
    with pytest.raises(queen_train.QueenTrainBusy):
        queen_train.start(_base(), generations=2, pop=2, seed=9, save_duel=False, apply=False)
    with pytest.raises(HTTPException) as ei:
        queen_train_endpoint(QueenTrainIn(scenario=ScenarioIn(**FAST), generations=2, pop=2, seed=9))
    assert ei.value.status_code == 409
    assert queen_train.stop()["stop_requested"] is True


def test_stop_finishes_current_generation() -> None:
    """Остановка не рвёт поколение на середине: оно досчитывается, лог остаётся
    целым, задача помечается завершённой."""
    queen_train.start(_base(), generations=30, pop=2, seed=7, save_duel=False, apply=False)
    queen_train.stop()
    done = queen_train.wait(300)
    assert done["running"] is False and done["finished_at"] is not None
    assert 1 <= done["generations_done"] < 30
    assert [r["gen"] for r in done["log"]] == list(range(done["generations_done"]))


def test_finished_queen_seats_champions_and_shelves_duel(_restore_brains) -> None:
    """По готовности чемпионы сами садятся за штурвалы (обе ветки весов) и
    встают на полку ринга — проснуться может только стенд, а не пользователь."""
    snap = queen_train.start(_base(), generations=1, pop=2, seed=7, save_duel=True, apply=True)
    assert snap["running"] is True
    done = queen_train.wait(300)
    assert done["error"] is None
    champ_e = np.asarray(done["champions"]["evader"]["w"], dtype=float)
    champ_m = np.asarray(done["champions"]["missile"]["w"], dtype=float)
    assert champ_e.size == champ_m.size == 2 * FEAT_DIM

    live = brain_store.get_circuit("stub")
    assert np.allclose(np.asarray(live.W_dn).reshape(-1), champ_m, atol=1e-12)
    assert live.trained
    ev = brain_store._evader
    assert ev is not None and ev.trained
    assert np.allclose(np.asarray(ev.W_dn).reshape(-1), champ_e, atol=1e-12)
    # путь читаем ВО ВРЕМЯ: фикстура-песочница патчит DATA_DIR на tmp_path
    assert (brain_store.DATA_DIR / brain_store.EVADER_WEIGHTS_FILE).exists()

    ring = done["saved"]["ring"]
    assert ring and ring["id"] == 1 and "Королева" in ring["label"]
    shelf_path = circuit.DATA_DIR / RING_FILE
    shelf = json.loads(shelf_path.read_text(encoding="utf-8"))
    assert [d["label"] for d in shelf] == [ring["label"]]
    # веса с полки — те же чемпионы (дуэт лёг целиком, а не одной стороной)
    assert np.allclose(np.asarray(shelf[0]["evader"]["w"], dtype=float), champ_e, atol=1e-12)


def test_train_endpoints_roundtrip(_restore_brains) -> None:
    """POST стартует и отдаёт снимок, GET показывает готовность: тот же цикл
    доступен стенду без вкладки, а тело — вложенное (как /api/duel)."""
    started = queen_train_endpoint(
        QueenTrainIn(scenario=ScenarioIn(**FAST), generations=1, pop=2, seed=7, save_duel=False, apply=False)
    )
    assert started["running"] is True and started["generations"] == 1
    deadline = queen_train.wait(300)
    assert deadline["running"] is False and deadline["generations_done"] == 1
    st = queen_train_status_endpoint()
    assert st["generations_done"] == 1 and st["champions"] is not None
    assert st["saved"] == {"ring": None, "weights": None}


def test_train_body_validated() -> None:
    """Границы параметров — на входе, а не clamp'ом внутри потока: тело с
    generations=0 молча поднимало бы войну с дефолтной записью в живые веса."""
    from pydantic import ValidationError

    for bad in ({"generations": 0}, {"generations": 61}, {"pop": 1}, {"seed": 10**12}):
        with pytest.raises(ValidationError):
            QueenTrainIn(**bad)
    ok = QueenTrainIn(generations=60, pop=12, seed=-3)
    assert (ok.generations, ok.pop, ok.seed) == (60, 12, -3)


def test_train_wired() -> None:
    """Проводка маркерами: у модуля есть эндпоинты запуска/статуса/остановки,
    фронтовый космос «Дуэль» умеет воевать в фоне и опрашивать стенд, а
    применение чемпионов — общие ворота для руки и для фона."""
    src = (REPO / "navedenie" / "app.py").read_text(encoding="utf-8")
    for marker in (
        "/api/queen/train",
        "/api/queen/train/stop",
        "except QueenTrainBusy as exc",
        "status_code=409",
        "from navedenie.redqueen import apply_champions",
    ):
        assert marker in src, f"app.py: нет маркера {marker!r}"
    ui = (REPO / "web" / "src" / "shell" / "DuelWorkspace.tsx").read_text(encoding="utf-8")
    for marker in (
        "Самообучение на сервере",
        "Воевать в фоне",
        "fetch('/api/queen/train')",
        "fetch('/api/queen/train/stop', { method: 'POST' })",
        "generations_done",
        "window.setTimeout(() => void srvPoll(), 2000)",
        "void refreshRing()",
    ):
        assert marker in ui, f"DuelWorkspace: нет маркера {marker!r}"
    rq = (REPO / "navedenie" / "redqueen.py").read_text(encoding="utf-8")
    assert "def apply_champions(" in rq and "def _dn_weights(" in rq


def _shelf(n: int) -> list[int]:
    """N дуэтов на полку из evader-популяций — свои мозги, без фоновой войны."""
    from navedenie.evader_train import init_evader_population
    from navedenie.queen_ring import save_duel

    ids = []
    for i in range(n):
        g = init_evader_population(2, seed=60 + i)[i % 2]
        blob = {"kind": "bio", "w": [float(x) for x in np.asarray(g.w).reshape(-1)], "gain": float(g.gain)}
        ids.append(save_duel(f"Проба {i}", blob, blob)["id"])
    return ids


def test_auto_ring_battles_the_shelf_after_the_campaign(_restore_brains) -> None:
    """Сезон замыкается: досчитанная кампания сама сводит полку с новичком,
    форма обновляется без ручного клика — следующий inherit растёт от свежих рангов."""
    shelf_ids = _shelf(2)
    queen_train.start(_base(), generations=1, pop=2, seed=7, save_duel=True, apply=False, auto_ring=True)
    done = queen_train.wait(300)
    assert done["error"] is None and done["auto_ring"] is True
    fresh = done["saved"]["ring"]["id"]
    rb = done["saved"].get("ring_battle")
    assert rb is not None, "auto_ring обязан оставить свод в снимке"
    assert {d["id"] for d in rb["duels"]} == set(shelf_ids) | {fresh}
    assert sorted(s["rank"] for s in rb["standings"]) == [1, 2, 3]
    # свод пишет форму полки — и делает это как виток 9, с движением
    raw = json.loads((circuit.DATA_DIR / "queen_ring_form.json").read_text(encoding="utf-8"))
    assert {int(r["id"]) for r in raw["rows"]} == set(shelf_ids) | {fresh}
    assert all("move" in r for r in raw["rows"])


def test_auto_ring_skips_stopped_campaign(_restore_brains) -> None:
    """Прерванная кампания — не итог сезона: свод не играется, полка сведена не была."""
    _shelf(2)
    queen_train.start(_base(), generations=40, pop=2, seed=7, save_duel=True, apply=False, auto_ring=True)
    queen_train.stop()
    done = queen_train.wait(600)
    assert done["stop_requested"] is True and done["running"] is False
    assert "ring_battle" not in done["saved"]


def test_auto_ring_caps_subset_with_the_newcomer(_restore_brains) -> None:
    """Шестеро на полке — сводится не марафон: не больше четырёх дуэтов,
    и свежий чемпион обязан быть в составе, даже если форма его ещё не знает."""
    _shelf(6)
    queen_train.start(_base(), generations=1, pop=2, seed=7, save_duel=True, apply=False, auto_ring=True)
    done = queen_train.wait(600)
    rb = done["saved"]["ring_battle"]
    assert 2 <= len(rb["duels"]) <= 4
    assert done["saved"]["ring"]["id"] in {d["id"] for d in rb["duels"]}


def test_auto_ring_wired() -> None:
    """Проводка маркерами: поток зовёт свод после финиша и только без остановки,
    тело и эндпоинт прокидывают флаг, фронт его показывает."""
    tr = (REPO / "navedenie" / "queen_train.py").read_text(encoding="utf-8")
    for marker in (
        "def _season_battle(",
        'if job.get("auto_ring") and not job.get("stop_requested"):',
        "auto_ring: bool = False,",
    ):
        assert marker in tr, f"queen_train: нет маркера {marker!r}"
    app = (REPO / "navedenie" / "app.py").read_text(encoding="utf-8")
    assert "auto_ring=body.auto_ring," in app
    ui = (REPO / "web" / "src" / "shell" / "DuelWorkspace.tsx").read_text(encoding="utf-8")
    for marker in ("auto_ring: srvRing,", "свести ринг после войны", "сезон сведён"):
        assert marker in ui, f"DuelWorkspace: нет маркера {marker!r}"
