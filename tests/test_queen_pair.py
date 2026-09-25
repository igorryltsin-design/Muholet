"""Сборная пара с полки ринга (`queen_ring.apply_pair`, /api/queen/ring/pair).

Полка была только таблицей рангов. Здесь: ракету берём у одного дуэта, уклониста
— у другого, сажаем в живые мозг-контейнеры теми же воротами, что кнопка
«Применить чемпионов», и бой можно смотреть в сцене. Проверки честности
сборки (чужие стороны не пересекаются), собственной пары, чужого id и проводки.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from navedenie import brain_store
from navedenie.app import QueenPairIn, queen_ring_pair_endpoint
from navedenie.circuit import FEAT_DIM
from navedenie.evader_train import init_evader_population
from navedenie.queen_ring import apply_pair, save_duel

REPO = Path(__file__).resolve().parents[1]


def _blob(i: int) -> dict:
    """«Выученный» геном: врождённый рефлекс школы + свой шум — два разных бойца."""
    g = init_evader_population(2, seed=30 + i)[i % 2]
    return {"kind": "bio", "w": [float(x) for x in np.asarray(g.w).reshape(-1)], "gain": float(g.gain)}


def _dn(w: list[float]) -> np.ndarray:
    return np.clip(np.asarray(w, dtype=np.float64).reshape(2, FEAT_DIM), -4.0, 4.0)


@pytest.fixture
def _restore_brains():
    """Применение пары пишет в процессный реестр мозгов — вернуть как было."""
    prev_stub = brain_store._circuits.get("stub")
    prev_evader = brain_store._evader
    yield
    if prev_stub is None:
        brain_store._circuits.pop("stub", None)
    else:
        brain_store._circuits["stub"] = prev_stub
    brain_store._evader = prev_evader


def test_pair_mixes_sides_from_two_duels(_restore_brains) -> None:
    a = save_duel("Атака", _blob(0), _blob(1))
    b = save_duel("Оборона", _blob(2), _blob(3))

    out = apply_pair(a["id"], b["id"])

    assert out["attacker"] == a["id"] and out["defender"] == b["id"]
    assert out["ok"] is True
    # живой прогон берёт усиление ракеты из сценария, поэтому стенд обязан
    # вернуть фактическое — иначе фронт выведет в бой чемпиона с чужими руками
    assert out["gains"] == {"evader": pytest.approx(float(np.clip(b["evader"]["gain"], 0.2, 3.0))),
                            "missile": pytest.approx(float(np.clip(a["missile"]["gain"], 0.2, 3.0)))}
    m = brain_store._circuits["stub"]  # не get_circuit(): он передёргивает gain из аргументов
    e = brain_store._evader
    # ракета — сторона атакующего, цель — сторона обороняющегося; чужие не пересекаются
    assert np.array_equal(m.W_dn, _dn(a["missile"]["w"]))
    assert np.array_equal(e.W_dn, _dn(b["evader"]["w"]))
    assert not np.array_equal(m.W_dn, _dn(b["missile"]["w"]))
    assert not np.array_equal(e.W_dn, _dn(a["evader"]["w"]))
    assert m.trained and e.trained
    assert m.gain == pytest.approx(out["gains"]["missile"])
    assert e.gain == pytest.approx(out["gains"]["evader"])


def test_pair_of_one_duel_is_its_own_battle(_restore_brains) -> None:
    """Дуэт против себя — честная пара: его же ракета против его же уклониста."""
    a = save_duel("Сам", _blob(4), _blob(5))
    apply_pair(a["id"], a["id"])
    assert np.array_equal(brain_store.get_circuit("stub").W_dn, _dn(a["missile"]["w"]))
    assert np.array_equal(brain_store._evader.W_dn, _dn(a["evader"]["w"]))


def test_pair_rejects_ids_off_the_shelf() -> None:
    a = save_duel("Единственный", _blob(6), _blob(7))
    for bad in ((a["id"], 99), (99, a["id"])):
        with pytest.raises(ValueError) as exc:
            apply_pair(*bad)
        assert "на полке ринга нет дуэта 99" in str(exc.value)
    with pytest.raises(ValueError):
        apply_pair(98, 99)


def test_pair_endpoint_roundtrip_and_422(_restore_brains) -> None:
    a = save_duel("А", _blob(8), _blob(9))
    b = save_duel("Б", _blob(10), _blob(11))

    got = queen_ring_pair_endpoint(QueenPairIn(attacker=a["id"], defender=b["id"]))
    assert got["ok"] is True and got["attacker"] == a["id"] and got["defender"] == b["id"]
    assert got["missile"]["kind"] == "stub" and got["evader"]["weights_file"] == "weights_evader.npz"

    with pytest.raises(HTTPException) as exc:
        queen_ring_pair_endpoint(QueenPairIn(attacker=a["id"], defender=b["id"] + 1000))
    assert exc.value.status_code == 422
    assert "нет дуэта" in exc.value.detail


def test_pair_wired() -> None:
    """Проводка маркерами: маршрут у стенда есть, а космос «Дуэль» берёт стороны
    с полки и летит тем же патчем, что «Дуэль чемпионов»."""
    src = (REPO / "navedenie" / "app.py").read_text(encoding="utf-8")
    for marker in (
        "@app.post(\"/api/queen/ring/pair\")",
        "from navedenie.queen_ring import apply_pair",
        "class QueenPairIn(BaseModel):",
    ):
        assert marker in src, f"app.py: нет маркера {marker!r}"
    rq = (REPO / "navedenie" / "queen_ring.py").read_text(encoding="utf-8")
    assert "def apply_pair(" in rq and "from navedenie.redqueen import EXAM_GEOMETRY, apply_champions, brain_battle, init_brain_population" in rq
    ui = (REPO / "web" / "src" / "shell" / "DuelWorkspace.tsx").read_text(encoding="utf-8")
    for marker in (
        "Свести вживую",
        "fetch('/api/queen/ring/pair'",
        "const duelPair = async (att?: number, def?: number) => {",
        "const a = att ?? attId",
        "const attId = ringDuels.some((d) => d.id === pairAtt)",
        "const lastDuel = formCells.find((c) => c.attacker === attId && c.defender === defId)",
        "прошлый круг: ${ringName(c.attacker)} брал ${ringName(c.defender)} в",
        "отбивался от ${ringName(c.attacker)} всухую",
    ):
        assert marker in ui, f"DuelWorkspace: нет маркера {marker!r}"
    # общий патч с «Дуэлью чемпионов» — иначе setState не дойдёт до сценария
    assert ui.count("evader_law: 'brain',") >= 2
