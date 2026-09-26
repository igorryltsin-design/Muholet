"""Форма полки (`queen_ring.save_form`/`form`/`best_duels`, /api/queen/ring/form).

Полка без свода — очередь сохранения, и наследие доставало бы «кого последним
поставили». Сводный бой теперь запоминает расстановку сил, а наследие берёт
родителей по рангу. Проверки: снимок пишется боем и живёт в голом объекте,
удалённые дуэты из формы выпадают, нерасставленный новичок идёт в хвост,
базисы «форма»/«свежесть»/«пусто», границы эндпоинта и проводка маркерами.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from navedenie import circuit
from navedenie.app import queen_ring_form_endpoint
from navedenie.evader_train import init_evader_population
from navedenie.redqueen import EXAM_GEOMETRY
from navedenie.queen_ring import (
    FORM_FILE,
    SEASONS_FILE,
    best_duels,
    delete_duel,
    form,
    heritage,
    list_duels,
    ring_battle,
    save_duel,
)
from navedenie.sim import Scenario

REPO = Path(__file__).resolve().parents[1]
FAST = dict(dt=0.02, t_max=10, aspect="head-on")


def _duet_blob(i: int) -> dict:
    g = init_evader_population(2, seed=40 + i)[i % 2]
    return {"kind": "bio", "w": [float(x) for x in np.asarray(g.w).reshape(-1)], "gain": float(g.gain)}


def _duel(i: int, label: str) -> dict:
    return save_duel(label, _duet_blob(i), _duet_blob(i + 4))


def _fight(n: int = 2) -> dict:
    for i in range(n):
        _duel(i, f"Д{i}")
    return ring_battle(Scenario(**FAST), list_duels())


def test_battle_leaves_the_form_on_disk() -> None:
    out = _fight(2)
    raw = json.loads((circuit.DATA_DIR / FORM_FILE).read_text(encoding="utf-8"))
    # голый объект — как полка ринга: без обёрток, читается тем же загрузчиком
    assert isinstance(raw, dict) and set(raw) == {"computed_at", "scope", "rows", "cells"}
    assert set(raw["scope"]) == {"dt", "geometries"}
    assert [r["id"] for r in raw["rows"]] == [s["id"] for s in out["standings"]]
    assert [r["rank"] for r in raw["rows"]] == [1, 2]
    for r, s in zip(raw["rows"], out["standings"]):
        assert set(r) == {"id", "label", "rank", "score", "p_attack", "p_defense", "move"}
        assert r["score"] == s["score"] and r["label"] == s["label"]
        # первый свод сверять не с чем: движение появляется только со второго боя
        assert r["move"] is None


def test_form_drops_duels_off_the_shelf() -> None:
    got = _fight(2)
    champ = got["standings"][0]["id"]
    assert delete_duel(champ)

    f = form()
    assert f["basis"] == "форма"  # один расставленный боец ещё остался
    assert champ not in f["ids"]
    assert [r["id"] for r in f["rows"]] == [s["id"] for s in got["standings"][1:]]


def test_best_duels_ranks_above_freshness() -> None:
    """Кто сильнее, а не кто раньше встал: нерасставленный новичок — самый свежий
    id на полке, и всё равно идёт после участников свода."""
    got = _fight(2)
    ranked = [s["id"] for s in got["standings"]]
    fresh = _duel(9, "Новичок")["id"]
    assert fresh > max(ranked)

    best = best_duels(2)
    assert best["basis"] == "форма" and best["ids"] == ranked  # новичка в паре родителей нет
    assert best["rows"][0]["rank"] == 1 and len(best["rows"]) == 2
    # а вся форма новичка помнит — в хвосте, по свежести
    assert form()["ids"] == ranked + [fresh]


def test_form_falls_back_to_freshness_then_empty() -> None:
    _duel(0, "А")
    _duel(1, "Б")
    f = form()  # сводного боя ещё не было: наследовать не по чему, кроме свежести
    assert f["basis"] == "свежесть" and f["rows"] == [] and f["ids"] == [2, 1]

    delete_duel(1)
    delete_duel(2)
    g = form()
    assert g["basis"] == "пусто" and g["ids"] == [] and best_duels(2)["ids"] == []


def test_broken_form_file_is_tolerated() -> None:
    _fight(2)
    (circuit.DATA_DIR / FORM_FILE).write_text("{битый json", encoding="utf-8")
    assert form()["basis"] == "свежесть"  # полка цела — наследие живёт, снимок переживём


def test_form_endpoint_roundtrip_and_clamps() -> None:
    got = _fight(3)
    ranked = [s["id"] for s in got["standings"]]

    r = queen_ring_form_endpoint(limit=2)
    assert r["ok"] is True and r["basis"] == "форма" and r["ids"] == ranked[:2]
    assert len(r["rows"]) == 2 and r["rows"][0]["rank"] == 1

    # нижняя и верхняя границы: мест для прививки не больше четырёх
    assert len(queen_ring_form_endpoint(limit=0)["ids"]) == 1
    assert len(queen_ring_form_endpoint(limit=99)["ids"]) <= 4


def test_heritage_accepts_the_form_as_parents() -> None:
    """Связка витков 7 и 8: родителей следующей кампании даёт форма, а не порядок
    сохранения — и прививка берёт ровно их."""
    got = _fight(2)
    parents = best_duels(2)["ids"]
    heirs = heritage(parents, pop=4, seed=11, sigma=0.0)
    assert [p["id"] for p in heirs["from"]] == parents
    shelf = {d["id"]: d for d in list_duels()}
    want = np.asarray(shelf[parents[0]]["missile"]["w"], dtype=np.float64)
    assert np.array_equal(np.asarray(heirs["missile"][0]["w"], dtype=np.float64), want)


def test_form_wired() -> None:
    """Проводка маркерами: бой пишет форму, стенд её отдаёт, фронт наследует по
    рангу — и честно говорит, когда ранга ещё нет."""
    rq = (REPO / "navedenie" / "queen_ring.py").read_text(encoding="utf-8")
    assert "save_form(standings, sc_base, cells)" in rq.split("def ring_battle", 1)[1]
    for marker in ("def save_form(", "def form() -> dict:", "def best_duels(limit: int = 2)",
                   "prev[cid] - rank if cid in prev else None",
                   'SEASONS_FILE = "queen_ring_seasons.json"', "history[-SEASONS_MAX:]"):
        assert marker in rq, f"queen_ring: нет маркера {marker!r}"

    app = (REPO / "navedenie" / "app.py").read_text(encoding="utf-8")
    for marker in (
        '@app.get("/api/queen/ring/form")',
        "def queen_ring_form_endpoint(limit: int = 2)",
        "from navedenie.queen_ring import best_duels",
    ):
        assert marker in app, f"app.py: нет маркера {marker!r}"

    ui = (REPO / "web" / "src" / "shell" / "DuelWorkspace.tsx").read_text(encoding="utf-8")
    for marker in (
        "fetch('/api/queen/ring/form?limit=2')",
        "const heirIds = (ringForm?.ids.length ? ringForm.ids :",
        "const heirByRank = ringForm?.basis === 'форма'",
        "await refreshForm()",
        "сводного боя ещё не было, война начнётся с самых свежих дуэтов: ",
        "const ringMoveTag = (id: number)",
        "(поднялся на ${r.move})",
        "{!ringCells && formCells.length > 0 && (",
        "атака ↓ / оборона →",
        "матрица «кто кого бьёт» — память последнего круга",
        "правления: {reigns.map((s, i) => (",
        "{i === reigns.length - 1 ? ' ✓' : ''}",
        "медальный зачёт: {medalTop.map((m, i) => (",
        "s.order.slice(0, 3).forEach((id, i) => {",
        "m.pts += 3 - i",
        "void duelPair(a, d)",
        "клик по числу — свести эту пару вживую",
    ):
        assert marker in ui, f"DuelWorkspace: нет маркера {marker!r}"


def test_form_remembers_movement_between_battles() -> None:
    """Движение формы: второй свод сверяет расстановку с прежним снимком.
    Поднялся — плюс, сел — минус, новичка в прошлой форме не было — null."""
    got = _fight(2)
    ranked = [s["id"] for s in got["standings"]]
    path = circuit.DATA_DIR / FORM_FILE
    snap = json.loads(path.read_text(encoding="utf-8"))
    for r in snap["rows"]:  # сфабруем прежнюю расстановку: чемпионы поменялись местами
        r["rank"] = 3 - r["rank"]
    path.write_text(json.dumps(snap, ensure_ascii=False), encoding="utf-8")

    ring_battle(Scenario(**FAST), list_duels())  # тот же исход — тот же порядок
    rows = {r["id"]: r for r in form()["rows"]}
    assert rows[ranked[0]]["move"] == 1 and rows[ranked[1]]["move"] == -1

    # свежий дуэт встал на полку и свёл с ним: у новичка прежнего ранга нет
    fresh = _duel(9, "Новичок")["id"]
    ring_battle(Scenario(**FAST), list_duels())
    rows = {r["id"]: r for r in form()["rows"]}
    assert rows[fresh]["move"] is None
    assert all(isinstance(rows[i]["move"], int) for i in ranked)


def test_form_keeps_the_prediction_matrix() -> None:
    """Матрица «кто кого бьёт» доживает до следующего боя: свод сжимает пары до
    (атакующий, обороняющийся, доля взятий), а ✕ с полки уносит всю строку и весь
    столбец погибшего дуэта."""
    got = _fight(3)
    f = form()
    assert [(c["attacker"], c["defender"]) for c in f["cells"]] == [
        (c["attacker"], c["defender"]) for c in got["cells"]
    ]
    for c, live in zip(f["cells"], got["cells"]):
        assert set(c) == {"attacker", "defender", "p_hit"}
        assert c["p_hit"] == pytest.approx(live["p_hit"])
    assert best_duels(2)["cells"] == f["cells"]  # эндпоинт формы отдаёт матрицу как есть

    loser = got["standings"][-1]["id"]
    delete_duel(loser)
    kept = form()["cells"]
    assert kept and all(loser not in (c["attacker"], c["defender"]) for c in kept)


def test_reign_strip_grows_with_battles() -> None:
    """Лента правлений: каждый свод оставляет в истории чемпиона и порядок рангов.
    Повторный бой — новый сезон; битый файл переносится, а потолок — 24 записи."""
    got = _fight(2)
    f = form()
    assert len(f["seasons"]) == 1
    s = f["seasons"][0]
    assert set(s) == {"at", "champ_id", "champ_label", "order"}
    assert s["champ_id"] == got["standings"][0]["id"]
    assert s["champ_label"] == got["standings"][0]["label"]
    assert s["order"] == [x["id"] for x in got["standings"]]
    assert best_duels(2)["seasons"] == f["seasons"]  # эндпоинт отдаёт ленту как есть

    ring_battle(Scenario(**FAST), list_duels())  # повторный свод — новый сезон, не правка старого
    assert len(form()["seasons"]) == 2

    # сфабрикованная длинная история: бой дописывает сезон и срезает потолок хвостом
    stale = [{"at": 0.0, "champ_id": 1, "champ_label": "П", "order": [1]} for _ in range(30)]
    (circuit.DATA_DIR / SEASONS_FILE).write_text(json.dumps(stale, ensure_ascii=False), encoding="utf-8")
    champ = form()["seasons"][0]  # до боя история читается вся — form() не редактор
    assert champ["champ_label"] == "П"
    ring_battle(Scenario(**FAST), list_duels())
    seasons = form()["seasons"]
    assert len(seasons) == 24 and seasons[-1]["champ_label"] != "П"
    assert all(x["champ_label"] == "П" for x in seasons[:-1])

    (circuit.DATA_DIR / SEASONS_FILE).write_text("{битый json", encoding="utf-8")
    assert form()["seasons"] == []  # нет ленты — не значит падение ринга


def test_form_snapshot_keeps_what_earned_the_ranks() -> None:
    """Ранги считаются не на одном курсе, а на трёх геометриях свода и на шаге
    базового сценария — форма обязана помнить именно это, иначе сводка на грубом
    dt выглядела бы силой на точном."""
    _duel(0, "А")
    _duel(1, "Б")
    sc = replace(Scenario(**FAST), dt=0.01)
    out = ring_battle(sc, list_duels())
    scope = form()["scope"]
    assert scope["geometries"] == [g["aspect"] for g in EXAM_GEOMETRY]
    assert set(scope["geometries"]) == {b["aspect"] for b in out["cells"][0]["battles"]}
    assert scope["dt"] == pytest.approx(0.01)
