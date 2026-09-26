"""«Хроника войн» (`navedenie/queen_chronicle.py`): память о кампаниях самообучения.

Проверки: запись берётся из снимка фоновой задачи той же формы, что отдаёт стенд;
пустой кампании в хронике не место; id — преемственный максимум, а не len+1
(после потолка длина стоит на месте); доигранная фоновая война заходит в хронику
сама, без ручного вызова; эндпоинты списка/вычёркивания и проводка фронта.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from navedenie import circuit, queen_chronicle, queen_train
from navedenie.app import (
    QueenChronicleDeleteIn,
    ScenarioIn,
    queen_chronicle_delete_endpoint,
    queen_chronicle_endpoint,
)
from navedenie.queen_chronicle import CHRONICLE_FILE, CHRON_MAX
from navedenie.sim import Scenario

REPO = Path(__file__).resolve().parents[1]
FAST = dict(dt=0.02, t_max=10, aspect="head-on")

FIELDS = {
    "id",
    "finished_at",
    "label",
    "generations",
    "planned",
    "pop",
    "seed",
    "seconds",
    "scenario",
    "inherited",
    "curve",
    "p_hit_first",
    "p_hit_last",
    "ring_id",
    "shelved",
    "applied",
    "stopped",
    "error",
}


def _snap(**over) -> dict:
    """Снимок фоновой задачи — ровно той формы, какой его отдаёт GET /api/queen/train."""
    snap = {
        "running": False,
        "generations": 2,
        "generations_done": 2,
        "pop": 4,
        "seed": 7,
        "scenario": {"aspect": "head-on", "range_m": 6000, "v_t": 260, "off_axis_m": 400},
        "inherit": [],
        "save_duel": True,
        "apply": True,
        "log": [
            {"gen": 0, "p_hit_ring": 0.25, "exam_p_hit": 0.33, "missile_best": 1004.0, "evader_best": -8.0, "seconds": 2.0},
            {"gen": 1, "p_hit_ring": 0.75, "exam_p_hit": 0.66, "missile_best": 7.4, "evader_best": -11.0, "seconds": 2.1},
        ],
        "champions": None,
        "saved": {"ring": {"id": 3, "label": "Королева ×2 (сервер, seed 7)"}, "weights": {"ok": True}},
        "error": None,
        "stop_requested": False,
        "started_at": 100.0,
        "finished_at": 105.0,
        "seconds": 5.0,
    }
    snap.update(over)
    return snap


@pytest.fixture(autouse=True)
def _one_job_at_a_time():
    """Фоновая война не должна переживать тест: неостановленный поток допишет
    хронику уже в другой песочнице (или в чужом каталоге)."""
    queen_train.stop()
    queen_train.wait(180)
    yield
    queen_train.stop()
    queen_train.wait(180)
    queen_train._JOB = None
    queen_train._THREAD = None


def test_record_shape_and_order() -> None:
    entry = queen_chronicle.record(_snap())
    assert set(entry) == FIELDS
    assert entry["id"] == 1 and entry["label"] == "Королева ×2 (сервер, seed 7)"
    assert entry["generations"] == 2 and entry["planned"] == 2
    assert [r["gen"] for r in entry["curve"]] == [0, 1]
    assert entry["curve"][0].keys() == {"gen", "p_hit_ring", "exam_p_hit", "missile_best", "evader_best"}
    assert (entry["p_hit_first"], entry["p_hit_last"]) == (0.25, 0.75)
    assert entry["ring_id"] == 3 and entry["shelved"] is True and entry["applied"] is True

    second = queen_chronicle.record(_snap(saved={"ring": None, "weights": None}))
    assert second["id"] == 2 and second["shelved"] is False and second["applied"] is False
    assert second["label"] == "Кампания 2"  # без дуэта на полке кампания получает номер
    assert [c["id"] for c in queen_chronicle.list_campaigns()] == [1, 2]  # порядок записи


def test_record_skips_empty_campaign() -> None:
    """Без досчитанного поколения воевать нечем: прерванная на нуле война — не событие."""
    assert queen_chronicle.record(_snap(generations_done=0)) is None
    assert queen_chronicle.record(_snap(log=[])) is None
    assert queen_chronicle.list_campaigns() == []
    assert not (circuit.DATA_DIR / CHRONICLE_FILE).exists()


def test_chronicle_cap_keeps_inherited_ids() -> None:
    """Потолок без дублей id: после заполнения длина стоит на месте, и len+1
    повторял бы уже занятые номера."""
    for _ in range(CHRON_MAX + 5):
        queen_chronicle.record(_snap())
    entries = queen_chronicle.list_campaigns()
    assert len(entries) == CHRON_MAX
    ids = [e["id"] for e in entries]
    assert len(set(ids)) == CHRON_MAX
    assert queen_chronicle.record(_snap())["id"] == max(ids) + 1


def test_file_is_bare_list_like_ring() -> None:
    queen_chronicle.record(_snap())
    raw = json.loads((circuit.DATA_DIR / CHRONICLE_FILE).read_text(encoding="utf-8"))
    assert isinstance(raw, list) and raw[0]["id"] == 1


def test_background_campaign_writes_itself() -> None:
    """Фон доиграл — хроника пополнилась сама, без ручного record(): именно для
    этого запись и стоит в завершении потока."""
    queen_train.start(
        Scenario(**FAST),
        generations=1,
        pop=2,
        seed=5,
        save_duel=False,
        apply=False,
    )
    done = queen_train.wait(300)
    assert done["running"] is False and done["generations_done"] == 1
    entries = queen_chronicle.list_campaigns()
    assert len(entries) == 1
    e = entries[0]
    assert e["generations"] == 1 and e["seed"] == 5
    assert len(e["curve"]) == 1 and e["p_hit_last"] == e["curve"][0]["p_hit_ring"]
    assert e["shelved"] is False and e["applied"] is False and e["error"] is None


def test_chronicle_endpoints_roundtrip() -> None:
    from fastapi import HTTPException

    empty = queen_chronicle_endpoint()
    assert empty == {"ok": True, "campaigns": [], "total": 0}

    queen_chronicle.record(_snap())
    got = queen_chronicle_endpoint()
    assert got["total"] == 1 and got["campaigns"][0]["id"] == 1

    after = queen_chronicle_delete_endpoint(QueenChronicleDeleteIn(id=1))
    assert after["ok"] is True and after["campaigns"] == []

    with pytest.raises(HTTPException) as exc:
        queen_chronicle_delete_endpoint(QueenChronicleDeleteIn(id=1))
    assert exc.value.status_code == 404


def test_chronicle_wired() -> None:
    """Проводка маркерами: фон зовёт record в завершении, у стенда есть пара
    эндпоинтов, а космос «Дуэль» рисует хронику и освежает её по готовности."""
    train = (REPO / "navedenie" / "queen_train.py").read_text(encoding="utf-8")
    assert "from navedenie.queen_chronicle import record" in train
    assert "record(_shape(job))" in train
    src = (REPO / "navedenie" / "app.py").read_text(encoding="utf-8")
    for marker in (
        "/api/queen/chronicle",
        "/api/queen/chronicle/delete",
        "from navedenie.queen_chronicle import list_campaigns",
        "from navedenie.queen_chronicle import delete_campaign, list_campaigns",
    ):
        assert marker in src, f"app.py: нет маркера {marker!r}"
    ui = (REPO / "web" / "src" / "shell" / "DuelWorkspace.tsx").read_text(encoding="utf-8")
    for marker in (
        "Хроника войн",
        "fetch('/api/queen/chronicle')",
        "fetch('/api/queen/chronicle/delete', {",
        "void refreshChron()",
        "spark(c.curve.map((r) => r.p_hit_ring))",
    ):
        assert marker in ui, f"DuelWorkspace: нет маркера {marker!r}"
