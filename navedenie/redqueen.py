"""«Красная королева»: совместная эволюция двух нейро-сторон дуэли.

Обе стороны — мозги (один и тот же FlyCircuit-stub, отдельные веса): ракета летит
mode='bio' на своём геноме, цель уходит evader_law='brain' на своём. Гонка вооружений
по Ливанскому правилу: каждый отбор — против ВСЕХ живых оппонентов (медиана фитнеса),
иначе сторона затачивается под одного соперника, а не под вид. Фитнеси честные и
зеркальные по исходу одного и того же боя:

- ракета (меньше — лучше): взяла — t_перехвата + 0.02·n_int (быстрее и мягче — выше);
  промахнулась — 1000 + 0.02·n_int + 0.02·CPA (в шкале цели это «сбитым быть плохо»).
- цель — фитнес школы (evader_train._fitness): сбили — 1000+(жизнь−гибель), выжила —
  −t_survived − 0.02·n_int_ракеты + 0.002·CPA.

Экзамен поколения — бой чемпионов на фиксированных геометриях (три курса): p_hit и
медиана жизни цели. Тот же seed, та же генерация → бит-в-бит тот же результат.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from navedenie.circuit import FEAT_DIM, FlyCircuit
from navedenie.engine import collect
from navedenie.evader_train import (
    TRAIN_DT,
    TRAIN_STRIDE,
    _fitness,
    genome_circuit,
    init_evader_population,
)
from navedenie.sim import Scenario
from navedenie.swarm import FlyGenome, evolve

MISSILE_PENALTY = 1000.0
K_MISSILE_EFFORT = 0.02
# CPA-привязка: без неё у промахнувшейся ракеты нет градиента «поджать» точку
# сближения (0.001 тонул в шуме усилия); 0.02 делает сокращение CPA на ~135 м
# равным выигрышу ~1 с упреждённого перехвата, но ниже планки hit (штраф 1000)
K_MISSILE_CPA = 0.02

# экзамен: три фиксированные геометрии (встреча/бок/догон), честные и постоянные
EXAM_GEOMETRY = (
    dict(aspect="head-on", range_m=6000.0, v_t=260.0, off_axis_m=400.0),
    dict(aspect="beam", range_m=5500.0, v_t=260.0, off_axis_m=300.0),
    dict(aspect="tail-chase", range_m=7000.0, v_t=240.0, off_axis_m=150.0),
)


def init_brain_population(n: int = 8, seed: int = 7) -> list[FlyGenome]:
    """Старт мозговой стороны: врождённый рефлекс + шум (та же схема, что у школы)."""
    return init_evader_population(n, seed)


def generation_scenario(base: Scenario, gen: int, seed: int) -> Scenario:
    """Геометрия поколения — как в школе (три курса по кругу, свежий rng на (seed, gen))."""
    from navedenie.evader_train import generation_scenario as _gs

    return _gs(base, gen, seed)


def _missile_fitness(res, cap: float) -> float:
    if res.hit and res.t_hit is not None:
        return float(res.t_hit) + K_MISSILE_EFFORT * float(res.n_int)
    return MISSILE_PENALTY + K_MISSILE_EFFORT * float(res.n_int) + K_MISSILE_CPA * float(res.cpa_m)


def brain_battle(sc: Scenario, mg: FlyGenome, eg: FlyGenome) -> dict:
    """Один мозг против одного мозга на заданной геометрии sc; обе фитнеси из одного боя."""
    s = replace(sc, duel=True, mode="bio", brain="stub", evader_law="brain")
    cap = min(s.t_max, float(s.fuse_life_s))
    res = collect(s, stride=TRAIN_STRIDE, circuit=genome_circuit(s, mg), evader_circuit=genome_circuit(s, eg))
    return {
        "missile_fitness": _missile_fitness(res, cap),
        "evader_fitness": _fitness(res, cap),
        "hit": bool(res.hit),
        "t_hit": float(res.t_hit) if res.t_hit is not None else None,
        "t_survived": float(res.t_survived if res.t_survived is not None else res.t_end or 0.0),
        "missile_n_int": float(res.n_int),
        "cpa_m": float(res.cpa_m),
    }


def _median(xs: list[float]) -> float:
    return float(np.median(xs))


def queen_generation(
    base: Scenario,
    missile_pop: list[FlyGenome],
    evader_pop: list[FlyGenome],
    *,
    gen: int = 0,
    seed: int = 7,
    elite_k: int = 2,
    mutation: float = 0.25,
) -> dict:
    """Поколение гонки: круговой бой популяций на геометрии поколения → отбор обеих
    сторон → экзамен чемпионов на трёх фиксированных геометриях."""
    if len(missile_pop) < 2 or len(evader_pop) < 2:
        raise ValueError("красной королеве нужно минимум по два генома с каждой стороны")
    sc = generation_scenario(replace(base, dt=TRAIN_DT), gen, seed)

    # круговой бой: каждая пара даёт и фитнес ракеты (медиана по целям), и фитнес цели
    # (медиана по ракетам) — ни одна сторона не «специализируется на одном» сопернике
    m_rows: list[list[float]] = [[] for _ in missile_pop]
    e_rows: list[list[float]] = [[] for _ in evader_pop]
    battles = []
    for mi, mg in enumerate(missile_pop):
        for ei, eg in enumerate(evader_pop):
            b = brain_battle(sc, mg, eg)
            m_rows[mi].append(b["missile_fitness"])
            e_rows[ei].append(b["evader_fitness"])
            battles.append(b)
    m_fit = [_median(r) for r in m_rows]
    e_fit = [_median(r) for r in e_rows]

    m_best = int(np.argmin(m_fit))
    e_best = int(np.argmin(e_fit))
    nxt_m = evolve(missile_pop, m_fit, elite_k=max(1, elite_k), mutation=mutation, seed=seed)
    nxt_e = evolve(evader_pop, e_fit, elite_k=max(1, elite_k), mutation=mutation, seed=seed + 1)

    exam = [brain_battle(replace(sc, **g), missile_pop[m_best], evader_pop[e_best]) for g in EXAM_GEOMETRY]
    return {
        "gen": int(gen),
        "scenario": {"aspect": sc.aspect, "range_m": sc.range_m, "v_t": sc.v_t, "off_axis_m": sc.off_axis_m},
        "missile_population": [g.to_json() for g in nxt_m],
        "evader_population": [g.to_json() for g in nxt_e],
        "champions": {"missile": missile_pop[m_best].to_json(), "evader": evader_pop[e_best].to_json()},
        "stats": {
            "missile_best": m_fit[m_best],
            "missile_median": _median(m_fit),
            "evader_best": e_fit[e_best],
            "evader_median": _median(e_fit),
            "p_hit_ring": float(np.mean([b["hit"] for b in battles])),
            "best_idx": {"missile": m_best, "evader": e_best},
        },
        "exam": {
            "p_hit": float(np.mean([b["hit"] for b in exam])),
            "t_survived_median": _median([b["t_survived"] for b in exam]),
            "missile_n_int_median": _median([b["missile_n_int"] for b in exam]),
            "cpa_m_median": _median([b["cpa_m"] for b in exam]),
        },
    }


def _dn_weights(side: dict, who: str) -> np.ndarray:
    w = np.asarray(side.get("w") or [], dtype=np.float64).reshape(-1)
    if w.size != 2 * FEAT_DIM:
        raise ValueError(f"{who}: нужно {2 * FEAT_DIM} весов DN, получено {w.size}")
    return w


def apply_champions(missile: dict | None = None, evader: dict | None = None) -> dict:
    """Посадить чемпионов за штурвалы: цель — в weights_evader.npz (её летает
    evader_law='brain'), ракета — в живой мозг-контейнер схемы 'stub' (режим bio).
    Кривые веса — ValueError (эндпоинт превращает в 422). Общая ворота для руки
    пользователя и для фонового самообучения, чтобы записи в мозг не расходились.

    В ответ кладутся фактические усиления (`gains`) — живой прогон берёт усиление
    ракеты из сценария (engine → get_circuit(gain=sc.circuit_gain)), и без этой
    подсказки «Дуэль чемпионов» летала бы чемпионом с чужими руками: тот же мозг,
    но с не тем усилием, с каким он выигрывал на ринге."""
    from navedenie import brain_store

    out: dict = {"ok": True, "gains": {}}
    if evader is not None:
        w = _dn_weights(evader, "цель")
        g = float(np.clip(float(evader.get("gain") or 1.0), 0.2, 3.0))
        # цель — своя отдельная схема; ракета — живой контейнер той же схемы,
        # чтобы прочие её поля (kind, tau_s) не терялись от применения весов
        c = FlyCircuit(kind="stub", tau_s=0.025, gain=g)
        c.W_dn = np.clip(w.reshape(2, FEAT_DIM), -4.0, 4.0)
        c.trained = True
        brain_store.set_evader_circuit(c)
        out["evader"] = brain_store.evader_status()
        out["gains"]["evader"] = g
    if missile is not None:
        w = _dn_weights(missile, "ракета")
        g = float(np.clip(float(missile.get("gain") or 1.0), 0.2, 3.0))
        c = brain_store.get_circuit("stub", tau_s=0.025, gain=g)
        c.W_dn = np.clip(w.reshape(2, FEAT_DIM), -4.0, 4.0)
        c.trained = True
        brain_store.set_circuit(c)
        out["missile"] = brain_store.status().get("stub")
        out["gains"]["missile"] = g
    return out
