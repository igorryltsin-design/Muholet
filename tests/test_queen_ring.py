"""«Ринг чемпионов»: самобой сохранённых дуэтов (модуль queen_ring).

Проверки: полка (валидация весов, id, потолок), форма кругового сводного боя,
инвариант «взятие атакующего = не-взятие обороняющегося» и детерминизм сводки.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from navedenie import circuit
from navedenie.app import (
    EvaderSaveIn,
    QueenRingBattleIn,
    QueenRingDeleteIn,
    QueenRingSaveIn,
    ScenarioIn,
    queen_ring_battle_endpoint,
    queen_ring_delete_endpoint,
    queen_ring_list_endpoint,
    queen_ring_save_endpoint,
)
from navedenie.circuit import FEAT_DIM
from navedenie.evader_train import init_evader_population
from navedenie.queen_ring import RING_MAX, delete_duel, list_duels, ring_battle, save_duel
from navedenie.sim import Scenario

REPO = Path(__file__).resolve().parents[1]
FAST = dict(dt=0.02, t_max=10, aspect="head-on")


def _duet_blob(i: int) -> dict:
    """Два разных «выученных» дуэта из врождённого рефлекса школы + свой шум."""
    g = init_evader_population(2, seed=20 + i)[i % 2]
    return {"kind": "bio", "w": [float(x) for x in np.asarray(g.w).reshape(-1)], "gain": float(g.gain)}


def _duel(i: int, label: str) -> dict:
    return save_duel(label, _duet_blob(i), _duet_blob(i + 3))


def _digest(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def test_shelf_validates_and_ids_increment() -> None:
    d1 = _duel(0, "Первый")
    d2 = _duel(1, "Второй")
    assert [d1["id"], d2["id"]] == [1, 2]
    assert [d["label"] for d in list_duels()] == ["Первый", "Второй"]
    assert delete_duel(1) and not delete_duel(1)
    assert [d["id"] for d in list_duels()] == [2]
    with pytest.raises(ValueError):
        save_duel("пустой", {"kind": "bio", "w": [], "gain": 1.0}, _duet_blob(1))
    with pytest.raises(ValueError):
        save_duel("кривой", {"kind": "bio", "w": [0.5] * (2 * FEAT_DIM - 1), "gain": 1.0}, _duet_blob(1))
    with pytest.raises(ValueError):  # pn-рой на мозговом ринге не воюет
        save_duel("pn", {"kind": "pn", "w": [], "gain": 1.0, "pn_n": 4.0}, _duet_blob(1))


def test_shelf_is_capped() -> None:
    for i in range(RING_MAX + 3):
        _duel(i % 2, f"Д{i}")
    shelf = list_duels()
    assert len(shelf) == RING_MAX
    assert shelf[0]["label"] == "Д3"  # oldest fell off, ids keep growing
    assert shelf[-1]["id"] == RING_MAX + 3


def test_ring_shape_and_hit_save_invariant() -> None:
    a = _duel(0, "А")
    b = _duel(1, "Б")
    out = ring_battle(Scenario(**FAST), [a, b])
    assert len(out["cells"]) == 2  # упорядоченные пары: А→Б и Б→А
    for c in out["cells"]:
        assert len(c["battles"]) == 3  # три фиксированные геометрии экзамена
        assert {b["aspect"] for b in c["battles"]} == {"head-on", "beam", "tail-chase"}
        assert 0.0 <= c["p_hit"] <= 1.0
        assert np.isfinite(c["t_survived_median"]) and c["cpa_m_median"] >= 0.0
    st = {s["id"]: s for s in out["standings"]}
    # каждый бой даёт ровно одно «взятие ИЛИ отражение» на пару дуэтов-участников
    total_battles = len(out["cells"]) * 3
    assert sum(s["attack_hits"] + s["defense_saves"] for s in st.values()) == total_battles
    assert [s["rank"] for s in out["standings"]] == [1, 2]
    assert out["standings"][0]["score"] >= out["standings"][1]["score"]


def test_ring_is_deterministic_and_demands_a_field() -> None:
    a = _duel(0, "А")
    b = _duel(1, "Б")
    sc = Scenario(**FAST)
    assert _digest(ring_battle(sc, [a, b])) == _digest(ring_battle(sc, [a, b]))
    with pytest.raises(ValueError):
        ring_battle(sc, [a])


def test_ring_file_isolated_per_data_dir() -> None:
    """Полка живёт в песочнице фикстуры (tmp_path), а не в реальном data/:
    путь читается во время вызова."""
    _duel(0, "А")
    assert (circuit.DATA_DIR / "queen_ring.json").exists()


def test_ring_endpoints_roundtrip() -> None:
    """Полка → сводный бой → удаление — через эндпоинты; тело боя вложенное,
    как у /api/duel; ids фильтруют дуэты, кривые веса — 422."""
    for i in range(2):
        blob = _duet_blob(i)
        r = queen_ring_save_endpoint(
            QueenRingSaveIn(
                label=f"Э{i}",
                missile=EvaderSaveIn(w=blob["w"], gain=blob["gain"]),
                evader=EvaderSaveIn(w=_duet_blob(i + 5)["w"], gain=1.0),
            )
        )
        assert r["ok"] and r["duel"]["id"] == i + 1
    assert len(queen_ring_list_endpoint()["duels"]) == 2
    out = queen_ring_battle_endpoint(QueenRingBattleIn(scenario=ScenarioIn(**FAST)))
    assert len(out["cells"]) == 2 and len(out["standings"]) == 2
    bad = _duet_blob(0)
    bad["w"] = bad["w"][:-2]
    with pytest.raises(HTTPException) as ei:
        queen_ring_save_endpoint(
            QueenRingSaveIn(label="хромой", missile=EvaderSaveIn(w=bad["w"], gain=1.0), evader=EvaderSaveIn(w=_duet_blob(1)["w"], gain=1.0))
        )
    assert ei.value.status_code == 422
    assert queen_ring_delete_endpoint(QueenRingDeleteIn(id=1))["ok"]
    with pytest.raises(HTTPException) as ei:
        queen_ring_battle_endpoint(QueenRingBattleIn(scenario=ScenarioIn(**FAST)))
    assert ei.value.status_code == 422  # один дуэт на ринге воюет не против кого


def test_ring_wired() -> None:
    """Фронт дёргает все четыре маршрута полки и рисует ранги."""
    ui = (REPO / "web" / "src" / "shell" / "DuelWorkspace.tsx").read_text(encoding="utf-8")
    for marker in (
        "/api/queen/ring",
        "/api/queen/ring/save",
        "/api/queen/ring/delete",
        "/api/queen/ring/battle",
        "Ринг чемпионов",
        "Свести на ринге",
        "На полку ринга",
        "attack_hits",
        "defense_saves",
    ):
        assert marker in ui, f"DuelWorkspace: нет маркера {marker!r}"
