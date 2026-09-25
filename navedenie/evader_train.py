"""Школа уклониста: эволюция мозга цели против ракет на законах наведения.

Ученик — тот же геном, что и в рое (веса DN 2×FEAT_DIM + усиление, swarm.FlyGenome
kind='bio'): прогон от отдаёт в EvaderBrainSensor, а ракета летит по выбранному
закону. Приспособленность (меньше — лучше) считает честно, из исхода боя:
- сбили:    1000 + (боевая жизнь − время гибели)   — сбитым быть плохо, позже — легче;
- выжила:   −время жизни − 0.02·n_int_ракеты + 0.002·CPA — цель пользователя:
  «не быть сбитым, создать максимальные перегрузки ракете, выдержать её время».

Геометрия поколения детерминированно новая (seed+gen): ученик не должен
зубрить одну траекторию. Экзамен — тот же набор законов, но фиксированная
эталонная геометрия, чтобы кривая обучения была честной.
"""

from __future__ import annotations

from dataclasses import replace
from statistics import median

import numpy as np

from navedenie.circuit import FlyCircuit, default_W
from navedenie.engine import collect
from navedenie.parallel import parallel_map
from navedenie.pn import LAWS
from navedenie.sim import Scenario
from navedenie.swarm import FlyGenome, W_LIMIT, evolve

EVA_LAWS_DEFAULT = ("pn", "tpn", "apn")
HIT_PENALTY = 1000.0
# вес «накрутки»: сколько секунд-перегрузки ракеты стоит одной единицы фитнеса
K_MISSILE_EFFORT = 0.02
K_CPA = 0.002
# шаг интегрирования школы: тот же порядок, что у ринга (бои массовые)
TRAIN_DT = 0.02
TRAIN_STRIDE = 100_000


def init_evader_population(n: int = 12, seed: int = 7) -> list[FlyGenome]:
    """Старт: врождённый рефлекс «к пеленгу» + шум — бабочка на огонь,
    эволюции есть от чего отталкиваться."""
    rng = np.random.default_rng(seed)
    pop = []
    for _ in range(max(n, 2)):
        w = default_W() + rng.normal(0.0, 0.4, default_W().shape)
        pop.append(FlyGenome(kind="bio", w=np.clip(w, -W_LIMIT, W_LIMIT), gain=float(rng.uniform(0.4, 2.5))))
    return pop


def genome_circuit(base: Scenario, g: FlyGenome) -> FlyCircuit:
    c = FlyCircuit(tau_s=base.tau_s, gain=g.gain, kind="stub")
    c.W_dn = np.asarray(g.w, dtype=float).copy()
    c.trained = True
    return c


def generation_scenario(base: Scenario, gen: int, seed: int) -> Scenario:
    """Детерминированная новая геометрия поколения (как в рое): дуэль идёт на
    всех трёх курсах встречи по кругу, дальность/скорость/смещение — в разумных
    границах; перегрузка цели и боевая жизнь — из базового сценария."""
    rng = np.random.default_rng((int(seed) * 7919 + int(gen) * 104729) % (2**32))
    aspects = ("head-on", "beam", "tail-chase")
    return replace(
        base,
        aspect=aspects[int(gen) % 3],
        range_m=float(rng.uniform(4500.0, 9000.0)),
        v_t=float(rng.uniform(180.0, 330.0)),
        off_axis_m=float(rng.uniform(80.0, 900.0)),
    )


def exam_scenario(base: Scenario) -> Scenario:
    """Экзамен: фиксированная лобовая геометрия — кривая обучения сопоставима
    от поколения к поколению."""
    return replace(base, aspect="head-on", range_m=6000.0, v_t=260.0, off_axis_m=400.0)


def _fitness(res, cap: float) -> float:
    t_end = res.t_end if res.t_end is not None else 0.0
    if res.hit:
        return HIT_PENALTY + max(0.0, cap - t_end)
    return -t_end - K_MISSILE_EFFORT * res.n_int + K_CPA * res.cpa_m


def battle(sc: Scenario, g: FlyGenome) -> dict:
    """Один честный бой: мозг ученика против ракеты на законе sc.law."""
    s = replace(sc, duel=True, evader_law="brain", mode="pn", law=s_law(sc))
    cap = min(s.t_max, float(s.fuse_life_s))
    res = collect(s, stride=TRAIN_STRIDE, evader_circuit=genome_circuit(s, g))
    return {
        "fitness": _fitness(res, cap),
        "hit_by_missile": bool(res.hit),
        "t_survived": float(res.t_survived if res.t_survived is not None else res.t_end or 0.0),
        "missile_n_int": float(res.n_int),
        "cpa_m": float(res.cpa_m),
        "fuse_expired": bool(res.fuse_expired),
    }


def s_law(sc: Scenario) -> str:
    return sc.law if sc.law in LAWS else "pn"


def _battle_job(sc: Scenario, g: FlyGenome, law: str) -> dict:
    """Задача пула: тот же бой, но с законом ракеты, подставленным в сценарий."""
    return battle(replace(sc, law=law), g)


def evaluate_generation(
    base: Scenario,
    population: list[FlyGenome],
    laws: tuple[str, ...] | list[str],
    scen: Scenario | None = None,
) -> list[dict]:
    """Каждый ученик — против каждого закона на данной геометрии; сводка — медианы."""
    sc0 = scen if scen is not None else base
    laws = list(laws)
    flat = parallel_map(_battle_job, [(sc0, g, law) for g in population for law in laws])
    out = []
    for i in range(len(population)):
        rows = flat[i * len(laws) : (i + 1) * len(laws)]
        out.append(
            {
                "fitness": float(median(r["fitness"] for r in rows)),
                "hit_by_missile": float(np.mean([r["hit_by_missile"] for r in rows])) >= 0.5,
                "catch_rate": float(np.mean([r["hit_by_missile"] for r in rows])),
                "t_survived": float(median(r["t_survived"] for r in rows)),
                "missile_n_int": float(median(r["missile_n_int"] for r in rows)),
                "cpa_m": float(median(r["cpa_m"] for r in rows)),
            }
        )
    return out


def train_generation(
    base: Scenario,
    population: list[FlyGenome],
    *,
    laws: tuple[str, ...] | list[str] = EVA_LAWS_DEFAULT,
    gen: int = 0,
    seed: int = 7,
    elite_k: int = 3,
    mutation: float = 0.25,
    exam_every: int = 0,
) -> dict:
    """Поколение школы: тренировочная геометрия (seed, gen) → отбор → следующее
    поколение; опционально — экзамен на эталонной геометрии (валидация)."""
    for law in laws:
        if law not in LAWS:
            raise ValueError(f"неизвестный закон ракеты для школы: {law!r}")
    sc = replace(base, dt=TRAIN_DT, mode="pn")
    sc = generation_scenario(sc, gen, seed)
    rows = evaluate_generation(sc, population, laws, scen=sc)
    fits = [r["fitness"] for r in rows]
    nxt = evolve(population, fits, elite_k=max(1, elite_k), mutation=mutation, seed=seed)
    best = int(np.argmin(fits))
    bio_ws = [np.asarray(g.w, dtype=float).reshape(-1) for g in population if np.size(g.w)]
    diversity = float(np.mean(np.std(np.stack(bio_ws), axis=0))) if len(bio_ws) > 1 else 0.0
    out = {
        "gen": int(gen),
        "scenario": {"aspect": sc.aspect, "range_m": sc.range_m, "v_t": sc.v_t, "off_axis_m": sc.off_axis_m},
        "laws": list(laws),
        "results": [{"fly": g.to_json(), **r} for g, r in zip(population, rows)],
        "next_population": [g.to_json() for g in nxt],
        "stats": {
            "best": fits[best],
            "avg": float(np.mean(fits)),
            "worst": float(np.max(fits)),
            "survive_rate": 1.0 - float(np.mean([r["catch_rate"] for r in rows])),
            "best_t_survived": rows[best]["t_survived"],
            "best_missile_n_int": rows[best]["missile_n_int"],
            "diversity": round(diversity, 4),
            "best_idx": best,
        },
    }
    if exam_every and int(gen) % int(exam_every) == 0:
        ex = evaluate_generation(sc, [population[best]], laws, scen=exam_scenario(sc))
        out["exam"] = ex[0]
    return out
