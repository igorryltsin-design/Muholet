"""Виток 19: бои генетических циклов считаются в пуле процессов.

Зеркалит договорённость: школа/королева/ринг/рой больше не занимают единственный
процесс стенда — каждое поколение идёт через navedenie/parallel.py, а под pytest
и при MUHOLET_WORKERS<2 остаётся последовательным (тот же детерминированный итог).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import navedenie.parallel as par
from navedenie.evader_train import _battle_job, evaluate_generation, init_evader_population
from navedenie.pn import LAWS
from navedenie.sim import Scenario
from navedenie.swarm import rollout

SRC = Path(inspect.getmodule(par).__file__).resolve().parent


def test_parallel_map_preserves_order_and_star_args():
    jobs = [(i, i + 1) for i in range(8)]
    assert par.parallel_map(lambda *a: sum(a), jobs) == [1 + 2 * i for i in range(8)]


def test_serial_fallback_under_pytest_and_env_switch(monkeypatch):
    # под pytest пул не поднимается никогда — это и есть тестовый режим
    assert par._executor_or_none() is None
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("MUHOLET_WORKERS", "1")
    assert par._executor_or_none() is None
    assert par.worker_count() == 1
    monkeypatch.setenv("MUHOLET_WORKERS", "0")
    assert par.worker_count() == 0


def test_ga_loops_go_through_the_pool():
    # маркеры: каждый боевой цикл зовёт parallel_map по своим задачам
    assert "parallel_map(_battle_job" in (SRC / "evader_train.py").read_text(encoding="utf-8")
    ev = (SRC / "redqueen.py").read_text(encoding="utf-8")
    assert "parallel_map(brain_battle" in ev
    assert "parallel_map(" in (SRC / "queen_ring.py").read_text(encoding="utf-8")
    assert "parallel_map(rollout" in (SRC / "swarm.py").read_text(encoding="utf-8")


def test_evaluate_generation_matches_manual_battles():
    sc = Scenario()
    pop = init_evader_population(3, seed=11)
    laws = ["pn"]
    rows = evaluate_generation(sc, pop, laws, scen=sc)
    manual = [_battle_job(sc, g, "pn")["fitness"] for g in pop]
    assert [r["fitness"] for r in rows] == manual


def test_battle_job_substitutes_the_law():
    from dataclasses import replace

    from navedenie.evader_train import battle

    sc = Scenario(law="apn")
    g = init_evader_population(2, seed=5)[0]
    assert _battle_job(sc, g, "pn")["fitness"] == battle(replace(sc, law="pn"), g)["fitness"]


def test_pool_worker_fn_is_top_level_picklable():
    # spawn-дети импортируют функции по имени: лямбды и вложенные функции умрут в пуле
    for fn in (_battle_job, rollout):
        assert fn.__qualname__ == fn.__name__
        assert fn.__module__.startswith("navedenie.")
    from navedenie.redqueen import brain_battle

    assert brain_battle.__qualname__ == brain_battle.__name__
    assert set(LAWS) >= {"pn", "tpn", "apn"}
