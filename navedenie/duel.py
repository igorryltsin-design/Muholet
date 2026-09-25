"""Матрица дуэлей «Ринг»: закон/мозг ракеты × закон уклонения цели.

Каждая ячейка — честный прогон на ОДИНАКОВЫХ условиях (тот же сценарий, та же
дальность, тот же dt): вердикт (кто взял), время жизни цели до fuse или
перехвата, промах R_min, усилие ракеты n_int (∫n·dt, g·с — «накрутка» из
запроса пользователя) и пик перегрузки. Строки — сторона ракеты: базовые id
законов наведения (pn/tpn/…) либо био-мозг (`bio:<kind>`); столбцы — сторона
цели: законы-уклонисты из evader.EVADERS либо «straight» (неманёвренная цель —
нижняя оценка, без неё непонятно, чего стоит уклонение).

При repeats>1 прогоны с шумом сенсора различаются seed'ом (Scenario.seed+i):
сводим по медиане и доле перехватов hit_rate.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from statistics import median

from navedenie.engine import collect
from navedenie.evader import EVADERS
from navedenie.pn import LAWS
from navedenie.sim import Scenario

BRAIN_KINDS = ("stub", "full", "connectome")
# «straight» — управляемая цель по нотам: базовая линия матрицы
EVADER_COLUMNS = ("straight",) + EVADERS


def _cell(sc: Scenario, msl: str, ev: str, repeats: int, stride: int) -> dict:
    rows: list[dict] = []
    for i in range(max(1, repeats)):
        s = replace(sc, seed=int(sc.seed) + i) if repeats > 1 else sc
        if msl.startswith("bio:"):
            s = replace(s, mode="bio", brain=msl.split(":", 1)[1])
        else:
            s = replace(s, mode="pn", law=msl)
        if ev == "straight":
            s = replace(s, duel=False)
        else:
            s = replace(s, duel=True, evader_law=ev)
        res = collect(s, stride=stride)
        rows.append(
            {
                "win": "missile" if res.hit else "evader",
                "hit_rate": 1.0 if res.hit else 0.0,
                "t_survived": res.t_survived if res.t_survived is not None else res.t_end,
                "cpa_m": res.cpa_m,
                "n_int": res.n_int,
                "n_peak": res.n_peak,
                "fuse_expired": bool(res.fuse_expired),
            }
        )
    out = {
        "win": rows[0]["win"],
        "hit_rate": float(median(r["hit_rate"] for r in rows)) if repeats > 1 else rows[0]["hit_rate"],
        "t_survived": float(median(r["t_survived"] for r in rows)),
        "cpa_m": float(median(r["cpa_m"] for r in rows)),
        "n_int": float(median(r["n_int"] for r in rows)),
        "n_peak": float(median(r["n_peak"] for r in rows)),
        "fuse_expired": rows[0]["fuse_expired"],
    }
    if repeats > 1:
        out["win"] = "missile" if out["hit_rate"] >= 0.5 else "evader"
    return out


def duel_matrix(
    sc: Scenario,
    missiles: Iterable[str],
    evaders: Iterable[str],
    repeats: int = 1,
    stride: int = 100_000,
) -> dict:
    """Матрица ячеек по списку сторон; неизвестный id стороны — ValueError
    (молчаливый откат на базовый закон пресечён так же, как в engine)."""
    msl = list(missiles)
    for m in msl:
        if not (m in LAWS or (m.startswith("bio:") and m.split(":", 1)[1] in BRAIN_KINDS)):
            raise ValueError(f"неизвестная сторона ракеты: {m!r}")
    ev = list(evaders)
    for e in ev:
        if e not in EVADER_COLUMNS:
            raise ValueError(f"неизвестный закон уклонения цели: {e!r}")
    return {
        "missiles": msl,
        "evaders": ev,
        "repeats": max(1, repeats),
        "scenario": {"aspect": sc.aspect, "fuse_life_s": sc.fuse_life_s, "duel": sc.duel, "n_target": sc.n_target},
        "cells": [
            {"row": m, "col": e, **_cell(sc, m, e, max(1, repeats), stride)} for m in msl for e in ev
        ],
    }
