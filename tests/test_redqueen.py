"""«Красная королева»: поколение совместной эволюции мозг-ракеты против мозг-цели.

Проверки: детерминизм (тот же seed → бит-в-бит тот же снимок поколения),
зеркальность фитнесей по исходу боя (шкалы штрафов 1000 не пересекаются),
форма ответа и эндпоинтов, применение чемпионов в обе ветки весов, маркеры UI.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from navedenie import brain_store
from navedenie.app import (
    EvaderSaveIn,
    QueenApplyIn,
    QueenGenIn,
    ScenarioIn,
    queen_apply_endpoint,
    queen_gen_endpoint,
)
from navedenie.circuit import FEAT_DIM, FlyCircuit
from navedenie.redqueen import _missile_fitness, init_brain_population, queen_generation
from navedenie.sim import Scenario
from navedenie.evader_train import _fitness

REPO = Path(__file__).resolve().parents[1]

FAST = dict(dt=0.02, t_max=10, aspect="head-on")


def _base() -> Scenario:
    return Scenario(**FAST)


def _digest(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def test_queen_generation_is_deterministic() -> None:
    """Один и тот же seed/генерация — дословно один и тот же снимок поколения:
    фронт крутит цикл поколений, кривые обязаны сходиться."""
    mp = init_brain_population(2, seed=7)
    ep = init_brain_population(2, seed=8)
    a = queen_generation(_base(), mp, ep, gen=0, seed=7)
    b = queen_generation(_base(), mp, ep, gen=0, seed=7)
    assert _digest(a) == _digest(b)


def test_generation_shape_and_populations_survive() -> None:
    """Популяции возвращаются того же размера, чемпионы — валидные геномы
    (2·FEAT_DIM весов), фитнеси и доли экзамена конечны и в своих границах."""
    mp = init_brain_population(2, seed=7)
    ep = init_brain_population(2, seed=8)
    out = queen_generation(_base(), mp, ep, gen=0, seed=7)
    assert len(out["missile_population"]) == 2 and len(out["evader_population"]) == 2
    for side in ("missile", "evader"):
        w = np.asarray(out["champions"][side]["w"], dtype=float)
        assert w.size == 2 * FEAT_DIM
        assert np.all(np.abs(w) <= 4.0 + 1e-9)
    st = out["stats"]
    assert np.isfinite(st["missile_best"]) and np.isfinite(st["evader_best"])
    assert 0.0 <= st["p_hit_ring"] <= 1.0
    assert 0.0 <= out["exam"]["p_hit"] <= 1.0
    assert out["exam"]["p_hit"] * 3 == round(out["exam"]["p_hit"] * 3)  # три фикс. геометрии


def test_fitness_scales_are_mirror_separated() -> None:
    """Зеркальность шкал (без боёв, на синтетическом исходе): взятие — фитнес
    ракеты в секундах (<500) и фитнес цели в штрафной зоне (>500); промах —
    ровно наоборот. Иначе отбор одной стороны ломает другую."""
    res_hit = type("R", (), {"hit": True, "t_hit": 4.0, "t_end": 4.0, "n_int": 20.0, "cpa_m": 10.0})()
    res_miss = type("R", (), {"hit": False, "t_hit": None, "t_end": 10.0, "n_int": 30.0, "cpa_m": 900.0})()
    assert _missile_fitness(res_hit, 30.0) < 500.0 < _fitness(res_hit, 30.0)
    assert _fitness(res_miss, 30.0) < 500.0 < _missile_fitness(res_miss, 30.0)


def test_queen_endpoints_roundtrip() -> None:
    """/api/queen/gen пустыми популяциями стартует сам; /api/queen/apply сажает
    чемпионов в обе ветки: цель — weights_evader.npz, ракета — живой stub.
    Глобальный контейнер цепей возвращаем на место: test_* рядом не должны
    наследовать тренированные веса."""
    prev_stub = brain_store._circuits.get("stub")
    prev_evader = brain_store._evader
    try:
        r = queen_gen_endpoint(QueenGenIn(scenario=ScenarioIn(**FAST), pop=2, gen=0))
        champ_m = r["champions"]["missile"]
        champ_e = r["champions"]["evader"]
        out = queen_apply_endpoint(
            QueenApplyIn(
                missile=EvaderSaveIn(w=champ_m["w"], gain=champ_m.get("gain") or 1.0),
                evader=EvaderSaveIn(w=champ_e["w"], gain=champ_e.get("gain") or 1.0),
            )
        )
        assert out["ok"] and out["evader"]["trained"] and out["missile"]["trained"]
        saved = brain_store.DATA_DIR / brain_store.EVADER_WEIGHTS_FILE
        assert saved.exists()  # путь читаем ВО ВРЕМЯ вызова: фикстура патчит DATA_DIR
        c = FlyCircuit(kind="stub")
        assert c.load(saved)
        assert np.allclose(np.asarray(c.W_dn).reshape(-1), np.asarray(champ_e["w"], dtype=float), atol=1e-12)
        live = brain_store.get_circuit("stub")
        assert np.allclose(np.asarray(live.W_dn).reshape(-1), np.asarray(champ_m["w"], dtype=float), atol=1e-12)
    finally:
        if prev_stub is None:
            brain_store._circuits.pop("stub", None)
        else:
            brain_store._circuits["stub"] = prev_stub
        brain_store._evader = prev_evader


def test_queen_apply_rejects_bad_shapes() -> None:
    for side in ("missile", "evader"):
        with pytest.raises(HTTPException) as ei:
            queen_apply_endpoint(QueenApplyIn(**{side: {"w": [0.1] * 5, "gain": 1.0}}))
        assert ei.value.status_code == 422


def test_queen_wired() -> None:
    """Проводка маркерами: фронт дёргает /api/queen/gen и умеет применять
    чемпионов; модуль использует боевой прогон с обеими схемами."""
    ui = (REPO / "web" / "src" / "shell" / "DuelWorkspace.tsx").read_text(encoding="utf-8")
    for marker in ("/api/queen/gen", "/api/queen/apply", "runQueen", "Применить чемпионов", "missile_population"):
        assert marker in ui, f"DuelWorkspace: нет маркера {marker!r}"
    src = (REPO / "navedenie" / "redqueen.py").read_text(encoding="utf-8")
    for marker in ('mode="bio"', 'evader_law="brain"', "evader_circuit=", "circuit="):
        assert marker in src, f"redqueen.py: нет маркера {marker!r}"


def test_duel_champions_one_click_and_repeat() -> None:
    """Виток 3: «Дуэль чемпионов» сажает обоих мозгов и тут же выпускает живой
    бой одним патчем сценария (не ждёт, пока setState дойдёт), а «повторить бой»
    в оверлее повторят детерминированный прогон тем же путём. Маркеры:
    onDuelNow проброшен из App в DuelWorkspace, duelChampions зовёт applyQueens
    и onDuelNow с дуэльной четвёркой полей, run принимает patch и летит по flown."""
    ui = (REPO / "web" / "src" / "shell" / "DuelWorkspace.tsx").read_text(encoding="utf-8")
    for marker in (
        "onDuelNow",
        "onDuelNow({",
        "if (!(await applyQueens())) return",
        "mode: 'bio'",
        "duel: true",
        "evader_law: 'brain'",
        "повторить бой",
        "onClick={() => onDuelNow({})}",
    ):
        assert marker in ui, f"DuelWorkspace: нет маркера {marker!r}"
    app = (REPO / "web" / "src" / "App.tsx").read_text(encoding="utf-8")
    for marker in (
        "const run = async (patch?: Partial<Scenario>) => {",
        "const s = patch ? ({ ...sc, ...patch } as Scenario) : sc",
        "onDuelNow={(patch) => void run(patch)}",
        "flown: Scenario = sc",
        "streamRun({ ...s, brain: s.brain }, he, s)",
    ):
        assert marker in app, f"App.tsx: нет маркера {marker!r}"
