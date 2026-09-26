"""Наследие полки: кампания королевы стартует с выученных мозгов.

Без наследия каждая фоновая война начинается с врождённого рефлекса и шума —
сила прошлых кампаний обнуляется. `queen_ring.heritage` прививает в стартовые
популяции выученные стороны с полки ринга, `queen_train` берёт их, если задан
`inherit`, и хроника помнит, с кем война начиналась. Здесь: прививка не трогает
остальные места старта, родители представлены дословно (нулевой шум) и по кругу
(несколько родителей), шум держит веса в границах, детерминизм, чужой id,
настоящее влияние наследия на первое поколение и проводка.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from navedenie import queen_train
from navedenie.app import QueenTrainIn
from navedenie.circuit import FEAT_DIM
from navedenie.evader_train import init_evader_population
from navedenie.queen_ring import heritage, save_duel
from navedenie.redqueen import init_brain_population
from navedenie.sim import Scenario

REPO = Path(__file__).resolve().parents[1]
FAST = dict(dt=0.02, t_max=10, aspect="head-on")
W_LIMIT = 4.0


def _blob(i: int) -> dict:
    """«Выученный» геном: врождённый рефлекс школы + свой шум — разные родители."""
    g = init_evader_population(2, seed=50 + i)[i % 2]
    return {"kind": "bio", "w": [float(x) for x in np.asarray(g.w).reshape(-1)], "gain": float(g.gain)}


def _flat(side: list[dict], i: int) -> np.ndarray:
    return np.asarray(side[i]["w"], dtype=np.float64).reshape(-1)


def _parent(entry: dict, key: str) -> np.ndarray:
    return np.asarray(entry[key]["w"], dtype=np.float64).reshape(-1)


def _w(g: dict) -> np.ndarray:
    return np.asarray(g["w"], dtype=np.float64).reshape(-1)


@pytest.fixture(autouse=True)
def _one_job_at_a_time():
    """Фоновый поток переживает тест — гасим его и до, и после."""
    queen_train.stop()
    queen_train.wait(180)
    yield
    queen_train.stop()
    queen_train.wait(180)
    queen_train._JOB = None
    queen_train._THREAD = None


def test_heritage_grafts_parents_without_touching_the_rest() -> None:
    """Нулевой шум даёт дословных клонов родителей в первых местах, а хвост
    популяции остаётся ровно тем стартом, что был бы без наследия."""
    a = save_duel("Родитель", _blob(0), _blob(1))
    pop, seed = 6, 7
    got = heritage([a["id"]], pop, seed, sigma=0.0)

    assert len(got["missile"]) == pop and len(got["evader"]) == pop
    assert np.array_equal(_flat(got["missile"], 0), _parent(a, "missile"))
    assert np.array_equal(_flat(got["evader"], 0), _parent(a, "evader"))
    # усиление наследуется вместе с весами: клон — это и веса, и рука
    assert got["missile"][0]["gain"] == pytest.approx(float(a["missile"]["gain"]))
    assert got["evader"][0]["gain"] == pytest.approx(float(a["evader"]["gain"]))
    for side, off in (("missile", 0), ("evader", 1)):
        plain = init_brain_population(pop, seed + off)
        for j in range(pop // 2, pop):
            assert np.array_equal(_flat(got[side], j), plain[j].w.reshape(-1)), f"{side}[{j}]"
        assert [g["gain"] for g in got[side][pop // 2:]] == [f.gain for f in plain[pop // 2:]]
    assert got["from"] == [{"id": a["id"], "label": "Родитель"}]


def test_heritage_cycles_several_parents() -> None:
    """Несколько родителей — слоты идут по кругу: 0 и 2 от первого, 1 от второго."""
    a = save_duel("Первый", _blob(2), _blob(3))
    b = save_duel("Второй", _blob(4), _blob(5))
    got = heritage([a["id"], b["id"], a["id"]], 6, 11, sigma=0.0)

    assert np.array_equal(_flat(got["missile"], 0), _parent(a, "missile"))
    assert np.array_equal(_flat(got["missile"], 1), _parent(b, "missile"))
    assert np.array_equal(_flat(got["missile"], 2), _parent(a, "missile"))
    # повторный id не плодит слотов: наследники перечислены один раз
    assert [f["id"] for f in got["from"]] == [a["id"], b["id"]]


def test_heritage_noise_stays_inside_the_weights_and_changes_clones() -> None:
    """С шумом наследник — уже не клон, но остаётся в границах эндпоинта применения."""
    a = save_duel("Родитель", _blob(6), _blob(7))
    got = heritage([a["id"]], 4, 3, sigma=0.5)

    child = _flat(got["missile"], 0)
    assert not np.array_equal(child, _parent(a, "missile"))
    assert np.all(np.abs(child) <= W_LIMIT + 1e-12)
    assert all(0.2 <= g["gain"] <= 3.0 for g in got["missile"] + got["evader"])
    assert child.size == 2 * FEAT_DIM


def test_heritage_is_deterministic_and_empty_means_plain_start() -> None:
    a = save_duel("Родитель", _blob(8), _blob(9))
    assert heritage([a["id"]], 4, 5, sigma=0.2) == heritage([a["id"]], 4, 5, sigma=0.2)

    plain = heritage([], 4, 5)
    assert [g["w"] for g in plain["missile"]] == [list(np.asarray(f.w).reshape(-1)) for f in init_brain_population(4, 5)]
    assert [g["w"] for g in plain["evader"]] == [list(np.asarray(f.w).reshape(-1)) for f in init_brain_population(4, 6)]
    assert plain["from"] == []


def test_heritage_rejects_ids_off_the_shelf() -> None:
    a = save_duel("Единственный", _blob(10), _blob(11))
    with pytest.raises(ValueError) as exc:
        heritage([a["id"], 99], 4, 1)
    assert "на полке ринга нет дуэта 99" in str(exc.value)


def test_background_campaign_fights_differently_with_heritage() -> None:
    """Наследие — не украшение снимка: при том же seed первое поколение считается
    из других популяций, иначе привитые мозги ни на что не повлияли."""
    a = save_duel("Родитель", _blob(12), _blob(13))
    sc = Scenario(**FAST)

    def run(*, inherit: list[int]) -> dict:
        queen_train.start(sc, generations=1, pop=2, seed=7, save_duel=False, apply=False, inherit=inherit)
        done = queen_train.wait(600)
        assert done["error"] is None
        assert done["generations_done"] == 1
        return done

    heir = run(inherit=[a["id"]])
    plain = run(inherit=[])

    assert heir["inherit"] == [a["id"]]
    assert plain["inherit"] == []
    assert heir["log"][0] != plain["log"][0]


def test_unknown_heir_refuses_before_the_thread_starts() -> None:
    """Чужой id — отказ до подъёма потока: иначе война молча сгорела бы в error."""
    with pytest.raises(ValueError) as exc:
        queen_train.start(Scenario(**FAST), generations=1, pop=2, seed=1, inherit=[77])
    assert "на полке ринга нет дуэта 77" in str(exc.value)
    assert queen_train.status()["running"] is False


def test_train_body_bounds_inherit() -> None:
    from navedenie.app import ScenarioIn

    QueenTrainIn(scenario=ScenarioIn(**FAST), inherit=[1, 2, 3, 4])  # верхняя граница ещё проходит
    with pytest.raises(ValidationError):
        QueenTrainIn(scenario=ScenarioIn(**FAST), inherit=[1, 2, 3, 4, 5])
    with pytest.raises(ValidationError):
        QueenTrainIn(scenario=ScenarioIn(**FAST), inherit=[0])


def test_heritage_wired() -> None:
    """Проводка маркерами: фон берёт наследие, стенд его пропускает, фронт — тумблер,
    хроника помнит, с кем война начиналась."""
    qt = (REPO / "navedenie" / "queen_train.py").read_text(encoding="utf-8")
    for marker in (
        'if job.get("inherit"):',
        'heirs = heritage(job["inherit"], job["pop"], job["seed"])',
        '"inherit": heirs,',
        'inherit: list[int] | None = None,',
    ):
        assert marker in qt, f"queen_train: нет маркера {marker!r}"
    api = (REPO / "navedenie" / "app.py").read_text(encoding="utf-8")
    assert "inherit: list[int] = []" in api and "inherit=body.inherit," in api
    assert 'raise HTTPException(status_code=422, detail=str(exc))' in api
    chr_ = (REPO / "navedenie" / "queen_chronicle.py").read_text(encoding="utf-8")
    assert '"inherited": [int(i) for i in (snap.get("inherit") or [])],' in chr_
    ui = (REPO / "web" / "src" / "shell" / "DuelWorkspace.tsx").read_text(encoding="utf-8")
    for marker in (
        "наследие полки",
        "inherit: srvHeirs ? heirIds : []",
        # родителей берёт форма полки (ранг), а не очередь сохранения
        "const heirIds = (ringForm?.ids.length ? ringForm.ids :",
        "srv.inherit?.length",
    ):
        assert marker in ui, f"DuelWorkspace: нет маркера {marker!r}"
