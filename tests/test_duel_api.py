"""Фаза 2 дуэли: «Ринг» (navedenie/duel.py + /api/duel) и проводка фронта.

Сверки: EVADER_RU (TypeScript labels.ts) — дословное зеркало Python EVADER_LABEL
(прецедент LAW_RU из test_glossary); матрица считает все пары и честно ловит
ValueError на неизвестной стороне; /api/duel по умолчанию — четыре закона против
базовой линии и уклонистов; космос «Дуэль» провязан в навигации и App.
"""

from __future__ import annotations

import re
from pathlib import Path

from navedenie.app import DuelIn, ScenarioIn, duel_endpoint
from navedenie.duel import BRAIN_KINDS, EVADER_COLUMNS, duel_matrix
from navedenie.evader import EVADERS
from navedenie.glossary import EVADER_LABEL
from navedenie.pn import LAWS
from navedenie.sim import Scenario

REPO = Path(__file__).resolve().parents[1]

# честный бой одного класса: встречный курс — неманёвренная цель сбита,
# уклонист выигрывает время и живёт до fuse
FAIR = dict(
    v_m=320, v_t=260, n_max=10, n_target=8, pn_n=4, dt=0.05,
    t_max=30, kill_radius_m=45, range_m=6000, aspect="head-on", off_axis_m=400,
    fuse_life_s=30,
)


def test_ts_evader_labels_match_python() -> None:
    src = (REPO / "web" / "src" / "shell" / "labels.ts").read_text(encoding="utf-8")
    block = re.search(r"EVADER_RU\s*:\s*Record<string,\s*string>\s*=\s*\{(.*?)\}", src, re.S)
    assert block, "не найден EVADER_RU в labels.ts"
    ts = dict(re.findall(r"(\w+)\s*:\s*'([^']*)'", block.group(1)))
    assert set(ts) == set(EVADER_LABEL), "наборы законов уклонения в TS и Python разошлись"
    for lid, ru in EVADER_LABEL.items():
        assert ts[lid] == ru, f"{lid}: TS {ts[lid]!r} ≠ Python {ru!r}"
    assert set(EVADER_LABEL) == set(EVADERS)
    assert EVADER_COLUMNS == ("straight",) + EVADERS


def test_duel_matrix_covers_all_pairs_and_verdicts() -> None:
    out = duel_matrix(Scenario(**FAIR), ["pn", "bio:stub"], ["straight", "away"], repeats=1, stride=100_000)
    assert len(out["cells"]) == 4
    assert {c["win"] for c in out["cells"]} <= {"missile", "evader"}
    for c in out["cells"]:
        for k in ("t_survived", "cpa_m", "n_int", "n_peak", "hit_rate"):
            assert c[k] is not None, (c["row"], c["col"], k)
    # неманёвренная цель на этих условиях — базовая линия: ракета берёт
    base = next(c for c in out["cells"] if c["col"] == "straight")
    assert base["win"] == "missile" and base["hit_rate"] == 1.0, base
    # уклонист оттягивает время: живёт дольше, чем сбитая базовая цель
    away = next(c for c in out["cells"] if c["col"] == "away" and c["row"] == "pn")
    assert away["t_survived"] > base["t_survived"], (away, base)


def test_duel_matrix_rejects_unknown_sides() -> None:
    sc = Scenario(**FAIR)
    for bad_msl in ("kitesurfing", "bio:golub"):
        try:
            duel_matrix(sc, [bad_msl], ["straight"])
        except ValueError:
            continue
        raise AssertionError(f"пропустил сторону ракеты {bad_msl!r}")
    try:
        duel_matrix(sc, ["pn"], ["kitesurfing"])
    except ValueError:
        pass
    else:
        raise AssertionError("пропустил неизвестный закон уклонения")
    assert "pn" in LAWS and set(BRAIN_KINDS) == {"stub", "full", "connectome"}


def test_api_duel_endpoint_defaults_and_nested_body() -> None:
    """Тело «Ринга» — вложенный сценарий (как /api/brain/compare): defaults дают
    базовые законы против базовой линии и всех уклонистов; dt полаге не считается."""
    body = DuelIn(scenario=ScenarioIn(**FAIR, duel=True, evader_law="away"))
    out = duel_endpoint(body)
    assert set(out["missiles"]) == {"pn", "tpn", "apn", "pn_gsn"}
    assert out["evaders"] == list(EVADER_COLUMNS)
    assert len(out["cells"]) == 4 * len(EVADER_COLUMNS)
    assert all(c["fuse_expired"] in (True, False) for c in out["cells"])
    big = duel_endpoint(DuelIn(scenario=ScenarioIn(**{**FAIR, "dt": 0.002}), missiles=["pn"], evaders=["away"]))
    assert big["cells"][0]["t_survived"] > 0  # бой идёт на matrix_dt, а не на точном dt проигрывания
    assert DuelIn().matrix_dt == 0.02 and "matrix_dt: float = 0.02" in (REPO / "navedenie" / "app.py").read_text(encoding="utf-8")


def test_duel_space_wired_in_frontend() -> None:
    top = (REPO / "web" / "src" / "shell" / "TopBar.tsx").read_text(encoding="utf-8")
    assert "'duel'" in top and "['duel', 'Дуэль'" in top, "дуэль не в навигации пространств"
    app = (REPO / "web" / "src" / "App.tsx").read_text(encoding="utf-8")
    for marker in (
        "import { DuelWorkspace }",
        "ws === 'duel'",
        "runDuelMatrix",
        "exportDuelCsv",
        "setDuelVerdict(duelInfo)",
        "saved === 'duel'",
        "duel_result?: 'missile' | 'evader' | null",
    ):
        assert marker in app, f"App.tsx: не найден маркер {marker!r}"
    ws = (REPO / "web" / "src" / "shell" / "DuelWorkspace.tsx").read_text(encoding="utf-8")
    assert "/api/duel" not in ws  # запрос живёт в App (общий fetch-слой)
    for marker in ("EVADER_RU[id]", "straight", "fuse_life_s", "missileIdOf", "bio:stub"):
        assert marker in ws, f"DuelWorkspace.tsx: не найден маркер {marker!r}"
    for marker in ("duel: boolean", "evader_law: EvaderLaw", "fuse_life_s: number"):
        assert marker in (REPO / "web" / "src" / "types.ts").read_text(encoding="utf-8"), "types.ts: нет полей дуэли"
