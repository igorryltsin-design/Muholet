"""«Ринг чемпионов»: круговой самобой сохранённых мозговых дуэтов.

Дуэт — пара выученных мозгов (ракета + цель), посаженная на полку в
data/queen_ring.json: «выращен и зачтён». Ринг сводит дуэты между собой честно:
дуэт A стреляет по дуэту B, а B защищается СВОИМ выученным уклонистом — мозг
против мозга на фиксированных геометриях (те же три курса, что экзамен королевы).
Ранг — сумма взятий в нападении и отражений в защите: кто сильнее и на атаке,
и на обороне. Глобального мозга процесса ринг не касается: веса дуэтов
живут только в локальных FlyCircuit, текущий чемпион стенда не трогается.
Исключение — `apply_pair`: он собирает пару прямо с полки (ракета атакующего +
уклонист обороняющегося) и сажает её в живые мозг-контейнеры через общие ворота
`redqueen.apply_champions`, чтобы бой можно было посмотреть в сцене.
Второе — `heritage`: полки достаточно, чтобы начать новую кампанию королевы не с
врождённого рефлекса, а с выученных мозгов (наследие), и следующие поколения
вырастают из уже сыгранных. А кого именно считать сильнейшим — хранит `form`:
снимок последнего сводного боя, чтобы наследие шло по рангу, а не по свежести.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np

from navedenie.circuit import FEAT_DIM
from navedenie.redqueen import EXAM_GEOMETRY, apply_champions, brain_battle, init_brain_population
from navedenie.parallel import parallel_map
from navedenie.sim import Scenario
from navedenie.swarm import FlyGenome, _mutate, fly_from_json

RING_FILE = "queen_ring.json"
RING_MAX = 12
# форма полки — последний сводный бой: кто сейчас сильнее, а не кто раньше встал
FORM_FILE = "queen_ring_form.json"
# список правлений — чем форма была раньше: чемпион и порядок каждого прошлого свода
SEASONS_FILE = "queen_ring_seasons.json"
SEASONS_MAX = 24
# геном-боёц обязан быть мозгом ровно нашей размерности (pn-рой на ринге не воюет)
_GENOME_WANT = 2 * FEAT_DIM


def _data_dir():
    from navedenie import circuit

    return circuit.DATA_DIR


def _genome(blob: dict, side: str) -> FlyGenome:
    g = fly_from_json(dict(blob or {}))
    w = np.asarray(g.w, dtype=np.float64)
    if g.kind != "bio" or w.size != _GENOME_WANT or not np.any(w):
        raise ValueError(f"дуэт: {side} обязан быть обученным мозгом (2×{FEAT_DIM} весов)")
    return g


def _load() -> list[dict]:
    path = _data_dir() / RING_FILE
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return raw if isinstance(raw, list) else []


def _store(duels: list[dict]) -> None:
    path = _data_dir() / RING_FILE
    path.write_text(json.dumps(duels, ensure_ascii=False), encoding="utf-8")


def list_duels() -> list[dict]:
    """Полки ринга: [{id, label, missile, evader}] в порядке сохранения."""
    return _load()


def _load_form() -> dict:
    path = _data_dir() / FORM_FILE
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _load_seasons() -> list[dict]:
    path = _data_dir() / SEASONS_FILE
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [s for s in raw if isinstance(s, dict)] if isinstance(raw, list) else []


def save_form(standings: list[dict], base: Scenario, cells: list[dict] | None = None) -> dict:
    """Запомнить форму полки по итогам сводного боя: ранги, очки и на чём именно
    они считаны. Без этого полка — просто очередь сохранения, и наследие
    доставало бы сильнейших наугад. Сверяет расстановку с прежним снимком и
    несёт каждой строке `move`: насколько дуэт поднялся (+) или сел (−)
    относительно прошлого свода; null — в прошлой форме его не было (новичок или первый свод).
    `cells` — пары своего круга: сжатые до (атакующий, обороняющийся, доля взятий),
    они позволяют стенду показывать «кто кого бьёт» до следующего боя."""
    prev = {int(r.get("id", -1)): int(r.get("rank", 0)) for r in (_load_form().get("rows") or [])}
    rows = []
    for s in standings:
        cid, rank = int(s["id"]), int(s.get("rank", 0))
        rows.append(
            {
                "id": cid,
                "label": str(s.get("label") or ""),
                "rank": rank,
                "score": int(s.get("score", 0)),
                "p_attack": float(s.get("p_attack", 0.0)),
                "p_defense": float(s.get("p_defense", 0.0)),
                "move": prev[cid] - rank if cid in prev else None,
            }
        )
    snap = {
        "computed_at": time.time(),
        "scope": {"dt": float(base.dt), "geometries": [str(g["aspect"]) for g in EXAM_GEOMETRY]},
        "rows": rows,
        "cells": [
            {
                "attacker": int(c["attacker"]),
                "defender": int(c["defender"]),
                "p_hit": float(c.get("p_hit", 0.0)),
            }
            for c in (cells or [])
        ],
    }
    (_data_dir() / FORM_FILE).write_text(json.dumps(snap, ensure_ascii=False), encoding="utf-8")
    if rows:
        # полоса правлений: от каждого свода остаётся чемпион и порядок рангов —
        # форма перезаписывается, а история «кто был первым сезон за сезоном» нет
        history = _load_seasons()
        history.append(
            {
                "at": snap["computed_at"],
                "champ_id": int(rows[0]["id"]),
                "champ_label": str(rows[0]["label"]),
                "order": [int(r["id"]) for r in rows],
            }
        )
        (_data_dir() / SEASONS_FILE).write_text(
            json.dumps(history[-SEASONS_MAX:], ensure_ascii=False), encoding="utf-8"
        )
    return snap


def form() -> dict:
    """Форма полки живая: строки ранга, у которых ещё есть дуэт на полке
    (после ✕ или потолка полки снимок обязан устареть честно), плюс базис —
    чем вообще можно наследовать.

    Наследовать можно только тех, кого сводный бой расставил по силам; дуэты,
    которые в том бою не участвовали (сводили часть полки по id), дописываются
    в хвост по свежести — они не сильнее и не слабее известных, о них бой
    просто молчит. Рядом кладёт `seasons` — список правлений: чемпион и порядок
    рангов каждого прошлого свода, от первого до текущего."""
    shelf = {int(d.get("id", -1)) for d in _load()}
    snap = _load_form()
    rows = [r for r in (snap.get("rows") or []) if int(r.get("id", -1)) in shelf]
    rows.sort(key=lambda r: (int(r.get("rank", 0)), -float(r.get("score", 0))))
    ranked = [int(r["id"]) for r in rows]
    rest = sorted((cid for cid in shelf if cid not in ranked), reverse=True)
    if ranked:
        basis = "форма"
    else:
        basis = "свежесть" if shelf else "пусто"
    cells = [
        c
        for c in (snap.get("cells") or [])
        if int(c.get("attacker", -1)) in shelf and int(c.get("defender", -1)) in shelf
    ]
    return {
        "basis": basis,
        "computed_at": snap.get("computed_at"),
        "scope": snap.get("scope") or {},
        "cells": cells,
        "seasons": _load_seasons(),
        "rows": rows,
        "ids": ranked + rest,
    }


def best_duels(limit: int = 2) -> dict:
    """Кого ставить родителями следующей кампании: сильнейших по последней форме
    полки, а если свода ещё не было — самых свежих (полка растёт в силе вместе с
    id). Ограничение — число мест, которые наследие вообще занимает за раз."""
    got = form()
    n = max(1, int(limit))
    return {
        "basis": got["basis"],
        "ids": got["ids"][:n],
        "rows": [r for r in got["rows"][:n]],
        "cells": got["cells"],
        "seasons": got["seasons"],
    }


def save_duel(label: str, missile: dict, evader: dict) -> dict:
    """Поставить дуэт на полку. Веса проходят ту же валидацию, что применение
    чемпионов; id — следующий целый, полка ограничена RING_MAX."""
    mg = _genome(missile, "ракета")
    eg = _genome(evader, "цель")
    duels = _load()
    # id — преемственный максимум, а не len+1: после заполнения полки длина
    # стоит на потолке и новые дуэты получили бы уже занятые id
    next_id = max((int(d.get("id", 0)) for d in duels), default=0) + 1
    entry = {
        "id": next_id,
        "label": (str(label or "").strip() or f"Дуэт {next_id}")[:40],
        "missile": mg.to_json(),
        "evader": eg.to_json(),
    }
    duels.append(entry)
    _store(duels[-RING_MAX:])
    return entry


def delete_duel(duel_id: int) -> bool:
    duels = _load()
    keep = [d for d in duels if int(d.get("id", -1)) != int(duel_id)]
    if len(keep) == len(duels):
        return False
    _store(keep)
    return True


def apply_pair(attacker_id: int, defender_id: int) -> dict:
    """Сборная пара с полки — в живые штурвалы: ракету берём у атакующего дуэта,
    уклониста — у обороняющегося. Так полка перестаёт быть только таблицей
    рангов: её стороны можно сводить в любой комбинации и смотреть бой вживую.
    Один и тот же дуэт с обеих сторон — его собственный бой, это тоже честная
    пара. Чужие стороны не пересекаются: уклонист атакующего и ракета
    обороняющегося в этом бою не летают. Ворота применения те же, что у кнопки
    «Применить чемпионов» (redqueen.apply_champions) — записанные веса не расходятся."""
    shelf = _load()

    def side(cid: int, key: str, who: str) -> dict:
        entry = next((d for d in shelf if int(d.get("id", -1)) == int(cid)), None)
        if entry is None:
            raise ValueError(f"на полке ринга нет дуэта {cid}")
        return _genome(entry.get(key), who).to_json()

    out = apply_champions(
        missile=side(attacker_id, "missile", "ракета"),
        evader=side(defender_id, "evader", "цель"),
    )
    out.update({"attacker": int(attacker_id), "defender": int(defender_id)})
    return out


def heritage(duel_ids: list[int], pop: int, seed: int, *, sigma: float = 0.12) -> dict:
    """Наследие полки: стартовые популяции кампании, привитые к выученным мозгам.

    Гонка вооружений без наследия каждый раз начинается заново — с врождённого
    рефлекса и шума, т.е. сила, добытая прошлыми кампаниями, обнуляется. Здесь
    первые `pop // 2` (но не меньше одного) мест стартовой популяции занимают
    мутанты унаследованных сторон с полки, а остальные — ровно тот же свежий
    старт, что был бы без наследия: прививка, а не замена. `sigma = 0` даёт
    дословных клонов родителей, больше `sigma` — тот же разброс, что у обычного
    мутатора (`swarm._mutate`), поэтому разнообразие популяции не теряется.
    Родителей можно указать несколько — слоты идут по кругу; пустой список
    означает обычный старт. Ракетная сторона стартует с `seed`, цель — с
    `seed + 1`, как в `_run`: у обеих сторон свой шум, а не один и тот же."""
    n = max(2, int(pop))
    ids: list[int] = []
    for cid in duel_ids or []:
        if int(cid) not in ids:
            ids.append(int(cid))
    shelf = {int(d.get("id", -1)): d for d in _load()}
    parents: list[dict] = []
    for cid in ids:
        entry = shelf.get(cid)
        if entry is None:
            raise ValueError(f"на полке ринга нет дуэта {cid}")
        parents.append(entry)
    rng = np.random.default_rng(int(seed))
    out: dict = {"from": [{"id": int(p.get("id", -1)), "label": str(p.get("label") or "")} for p in parents]}
    for side, key, who, off in (("missile", "missile", "ракета", 0), ("evader", "evader", "цель", 1)):
        base = init_brain_population(n, int(seed) + off)
        if parents:
            for j in range(min(n, max(1, n // 2))):
                base[j] = _mutate(_genome(parents[j % len(parents)].get(key), who), float(sigma), rng)
        out[side] = [g.to_json() for g in base]
    return out


def _median(xs: list[float]) -> float:
    return float(np.median(xs))


def _duel_genomes(entry: dict) -> tuple[FlyGenome, FlyGenome]:
    return _genome(entry.get("missile"), "ракета"), _genome(entry.get("evader"), "цель")


def ring_battle(base: Scenario, duels: list[dict], *, dt: float | None = None) -> dict:
    """Круговой самобой: каждая упорядоченная пара дуэтов (атака/оборона) ×
    три фиксированные геометрии. Возврат: cells (по парам), standings (ранг по
    взятиям и отражениям), geometry — на чём сводились."""
    if len(duels) < 2:
        raise ValueError("рингу нужно минимум два дуэта")
    sc_base = replace(base, dt=dt) if dt else base
    crews = []
    for d in duels:
        mg, eg = _duel_genomes(d)
        crews.append((int(d.get("id", 0)), str(d.get("label", "")), mg, eg))

    per_att: dict[int, list[bool]] = {cid: [] for cid, _, _, _ in crews}
    per_def: dict[int, list[bool]] = {cid: [] for cid, _, _, _ in crews}
    pairs = [(ai, di) for ai in range(len(crews)) for di in range(len(crews)) if ai != di]
    flat = parallel_map(
        brain_battle,
        [(replace(sc_base, **g), crews[ai][2], crews[di][3]) for ai, di in pairs for g in EXAM_GEOMETRY],
    )
    cells = []
    for pi, (ai, di) in enumerate(pairs):
        aid, did = crews[ai][0], crews[di][0]
        row = {"attacker": aid, "defender": did, "battles": []}
        for gi in range(len(EXAM_GEOMETRY)):
            sc = replace(sc_base, **EXAM_GEOMETRY[gi])
            b = flat[pi * len(EXAM_GEOMETRY) + gi]
            row["battles"].append(
                {
                    "aspect": sc.aspect,
                    "hit": b["hit"],
                    "t_survived": b["t_survived"],
                    "missile_n_int": b["missile_n_int"],
                    "cpa_m": b["cpa_m"],
                    "missile_fitness": b["missile_fitness"],
                    "evader_fitness": b["evader_fitness"],
                }
            )
            per_att[aid].append(b["hit"])
            per_def[did].append(not b["hit"])
        hits = [x["hit"] for x in row["battles"]]
        row.update(
            {
                "p_hit": float(np.mean(hits)),
                "t_survived_median": _median([x["t_survived"] for x in row["battles"]]),
                "n_int_median": _median([x["missile_n_int"] for x in row["battles"]]),
                "cpa_m_median": _median([x["cpa_m"] for x in row["battles"]]),
                "missile_fitness_median": _median([x["missile_fitness"] for x in row["battles"]]),
                "evader_fitness_median": _median([x["evader_fitness"] for x in row["battles"]]),
            }
        )
        cells.append(row)

    standings = []
    for cid, label, _, _ in crews:
        a_hits, a_n = int(np.sum(per_att[cid])), len(per_att[cid])
        d_saves, d_n = int(np.sum(per_def[cid])), len(per_def[cid])
        standings.append(
            {
                "id": cid,
                "label": label,
                "attack_hits": a_hits,
                "attack_n": a_n,
                "defense_saves": d_saves,
                "defense_n": d_n,
                "p_attack": a_hits / max(1, a_n),
                "p_defense": d_saves / max(1, d_n),
                "score": a_hits + d_saves,
            }
        )
    standings.sort(key=lambda s: (-s["score"], -s["p_attack"], s["id"]))
    for i, s in enumerate(standings):
        s["rank"] = i + 1
    save_form(standings, sc_base, cells)
    return {
        "duels": [{"id": cid, "label": label} for cid, label, _, _ in crews],
        "geometry": [dict(g) for g in EXAM_GEOMETRY],
        "cells": cells,
        "standings": standings,
    }
