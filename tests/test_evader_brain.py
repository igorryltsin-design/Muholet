"""Фаза 3 дуэли: мозг-уклонист (navedenie/evader_brain.py + evader_train.py).

Цель с обучаемой схемой: свой сенсор (сетчатка, задержка, шум — как у ракеты),
тот же контур FlyCircuit-stub, веса в data/weights_evader.npz. Проверяем:
честность команды (⊥ V_ц, клип n_target·g, ноль при потере и при n_target=0),
врезку в движок (кадр с evader-снимком, нет дуэли — нет снимка), детерминизм,
«школу уклониста» (фитнес из исхода боя, элита без изменений, ValueError на
чужом законе), эндпоинты /api/evader/gen и /api/evader/save и проводку фронта.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from navedenie.app import (
    EvaderGenIn,
    EvaderSaveIn,
    ScenarioIn,
    evader_gen_endpoint,
    evader_save_endpoint,
)
from navedenie import brain_store
from navedenie.brain_store import evader_status
from navedenie.circuit import FEAT_DIM, FlyCircuit
from navedenie.engine import collect
from navedenie.evader_brain import EvaderBrainSensor
from navedenie.evader_train import (
    battle,
    genome_circuit,
    init_evader_population,
    train_generation,
)
from navedenie.sim import G, Body, Scenario

REPO = Path(__file__).resolve().parents[1]

FAIR = dict(
    v_m=320, v_t=260, n_max=10, n_target=8, pn_n=4, dt=0.05,
    t_max=30, kill_radius_m=45, range_m=6000, aspect="head-on", off_axis_m=400,
    fuse_life_s=30,
)


def _brain_sc(**kw) -> Scenario:
    return Scenario(**{**FAIR, "duel": True, "evader_law": "brain", "mode": "pn", "law": "pn", **kw})


# ── сенсор и команда ─────────────────────────────────────────────────────────

def test_brain_command_is_orthogonal_clipped_and_honest() -> None:
    """Кнопка ⊥ скорости цели, клип n_target·g; без перегрузки и без захвата —
    ноль. Сенсор строит команду из запаздывающего изображения, как настоящая
    муха, а не из истинной геометрии."""
    sc = _brain_sc()
    circ = FlyCircuit(kind="stub", tau_s=sc.tau_s, gain=1.0)
    sens = EvaderBrainSensor(sc, circ, seed=3)
    t = Body(p=np.zeros(3), v=np.array([260.0, 0.0, 0.0]))
    m = Body(p=np.array([5000.0, 800.0, 200.0]), v=np.array([-320.0, 0.0, 0.0]))
    step = 0.0
    nonzero = 0
    for _ in range(80):
        a = sens.accel(t, m, step, sc.dt)
        assert abs(float(np.dot(a, t.v))) < 1e-6 * (np.linalg.norm(a) * np.linalg.norm(t.v) + 1.0)
        assert float(np.linalg.norm(a)) <= 8.0 * G + 1e-9
        if np.linalg.norm(a) > 1e-9:
            nonzero += 1
        step += sc.dt
        t.p = t.p + t.v * sc.dt
        m.p = m.p + m.v * sc.dt
    assert nonzero > 0, "на налетающей ракете мозг обязан давать команду"
    # нечем уходить
    z = EvaderBrainSensor(_brain_sc(n_target=0.0), FlyCircuit(kind="stub"), seed=3)
    assert np.allclose(z.accel(t, m, 0.0, sc.dt), 0.0)


def test_engine_wires_brain_and_snapshot() -> None:
    """duel+brain: кадры несут evader-снимок (слои, DN, act_b64; без картинки
    сетчатки); без мозга в цели — None; без дуэли — None."""
    g = init_evader_population(2, seed=5)[0]
    r = collect(_brain_sc(), stride=20, evader_circuit=genome_circuit(_brain_sc(), g))
    assert r.frames, "прогон не дал кадров"
    fr = next(f for f in r.frames if f.evader is not None)
    snap = fr.evader
    assert snap["kind"] == "evader" and "photo" not in snap
    for k in ("layers", "dn", "act_b64", "regions", "n_neurons", "weights", "lock"):
        assert k in snap, k
    assert r.duel and r.duel_result in ("missile", "evader")
    # реактивный закон — никакого evader-снимка
    r2 = collect(Scenario(**{**FAIR, "duel": True, "evader_law": "away"}), stride=20)
    assert all(f.evader is None for f in r2.frames)


def test_brain_run_is_deterministic() -> None:
    """Тот же геном, тот же seed — бит-в-бит тот же бой (сенсор на своём rng)."""
    g = init_evader_population(1, seed=11)[0]
    sc = _brain_sc(dt=0.05)
    a = collect(sc, stride=100_000, evader_circuit=genome_circuit(sc, g))
    b = collect(sc, stride=100_000, evader_circuit=genome_circuit(sc, g))
    assert a.t_survived == b.t_survived and a.hit == b.hit and a.n_int == b.n_int


# ── школа уклониста ──────────────────────────────────────────────────────────

def test_school_battle_fitness_is_honest() -> None:
    """Фитнес считается из исхода: сбитая цель — штраф 1000+ (с градиентом по
    времени гибели), выжившая — отрицательный («чем дольше и изматывающие,
    тем меньше»)."""
    sc = _brain_sc(dt=0.05)
    g = init_evader_population(1, seed=2)[0]
    b = battle(sc, g)
    assert np.isfinite(b["fitness"])
    if b["hit_by_missile"]:
        assert b["fitness"] >= 1000.0
    else:
        assert b["fitness"] < 1000.0


def test_school_generation_shape_and_elite() -> None:
    """Поколение: результаты по числу учеников, следующее поколение то же по
    размеру, элита (лучший фитнес) переходит без изменений; чужой закон
    ракеты — ValueError."""
    sc = Scenario(**{**FAIR, "dt": 0.05})
    pop = init_evader_population(3, seed=4)
    out = train_generation(sc, pop, laws=["pn"], gen=0, seed=4)
    assert len(out["results"]) == 3 and len(out["next_population"]) == 3
    best = min(range(3), key=lambda i: out["results"][i]["fitness"])
    assert np.allclose(
        np.asarray(out["next_population"][0]["w"]), np.asarray(pop[best].to_json()["w"])
    ), "элитарность потеряна"
    assert {"best", "avg", "worst", "survive_rate", "diversity"} <= set(out["stats"])
    with pytest.raises(ValueError):
        train_generation(sc, pop, laws=["kitesurfing"], gen=0, seed=4)


def test_evader_weights_roundtrip_and_status() -> None:
    """endpoint save: кривые веса — 422, верные — файл weights_evader.npz +
    evader_status(trained). Путь данных не коммитится, как и остальные веса.
    DATA_DIR берётся из brain_store во время вызова: фикстура conftest
    sandbox-ит его на tmp_path, и прод-каталог трогать нельзя."""
    with pytest.raises(HTTPException):
        evader_save_endpoint(EvaderSaveIn(w=[0.0] * 3))
    body = EvaderSaveIn(w=[float(i) * 0.01 for i in range(2 * FEAT_DIM)], gain=1.2)
    out = evader_save_endpoint(body)
    assert out["ok"] and out["evader"]["trained"] is True
    saved = brain_store.DATA_DIR / brain_store.EVADER_WEIGHTS_FILE
    assert saved.exists()
    assert evader_status()["trained"] is True
    # загрузка мигрирует/принимает сняток той же схемы
    c = FlyCircuit(kind="stub")
    assert c.load(saved)
    assert np.allclose(c.W_dn, np.asarray(body.w).reshape(2, FEAT_DIM))
    assert c.trained and abs(c.gain - 1.2) < 1e-9


def test_api_evader_gen_endpoint() -> None:
    """Тело — вложенный сценарий (как /api/duel); пустая популяция — авто-старт;
    чужой закон — 422."""
    scin = ScenarioIn(**{**FAIR})
    out = evader_gen_endpoint(EvaderGenIn(scenario=scin, laws=["pn"], population=[], seed=9))
    assert len(out["results"]) >= 2
    assert len(out["next_population"]) == len(out["results"])
    assert out["laws"] == ["pn"]
    with pytest.raises(HTTPException):
        evader_gen_endpoint(EvaderGenIn(scenario=scin, laws=["kitesurfing"]))


def test_duel_matrix_and_engine_accept_brain_column() -> None:
    """brain — законный столбец «Ринга» (EVADER_COLUMNS вырос до пяти)."""
    from navedenie.duel import EVADER_COLUMNS, duel_matrix

    assert "brain" in EVADER_COLUMNS
    out = duel_matrix(Scenario(**FAIR), ["pn"], ["brain"], repeats=1, stride=100_000)
    assert len(out["cells"]) == 1 and out["cells"][0]["t_survived"] > 0


def test_phase3_wired_in_frontend() -> None:
    """Проводка фронта маркерами: brain в типах и зеркале меток, школа и
    применение чемпиона в космосе, evader-снимок в типе кадра."""
    types = (REPO / "web" / "src" / "types.ts").read_text(encoding="utf-8")
    assert "'away' | 'negpn' | 'cpa_max' | 'brain'" in types
    assert "EvaderSnap" in types and "evader?: EvaderSnap | null" in types
    ws = (REPO / "web" / "src" / "shell" / "DuelWorkspace.tsx").read_text(encoding="utf-8")
    for marker in ("Школа уклониста", "/api/evader/gen", "/api/evader/save", "Применить чемпиона", "frame?.evader"):
        assert marker in ws, f"DuelWorkspace.tsx: нет маркера {marker!r}"
