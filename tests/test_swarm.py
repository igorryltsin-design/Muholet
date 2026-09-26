import numpy as np

from navedenie.sim import Scenario
from navedenie.swarm import evolve, evaluate_population, fly_from_json, init_population, rollout


def _sc() -> Scenario:
    return Scenario(aspect="head-on", v_m=780, v_t=220, range_m=6000, off_axis_m=200, n_max=30, t_max=14, dt=0.02)


def test_rollout_pn_intercepts() -> None:
    fly = fly_from_json({"kind": "pn", "pn_n": 4.0})
    res = rollout(_sc(), fly)
    assert res["miss_m"] < 60, res["miss_m"]
    assert res["hit"]
    assert len(res["traj_m"]) >= 4
    assert all(len(p) == 3 for p in res["traj_m"])


def test_genome_roundtrip_and_bounds() -> None:
    fly = fly_from_json({"kind": "bio", "w": [99.0] * 16, "gain": 99.0, "pn_n": 99.0})
    assert float(np.max(np.abs(fly.w))) <= 4.0
    assert 0.4 <= fly.gain <= 2.5
    restored = fly_from_json(fly.to_json())
    assert np.allclose(restored.w, fly.w)
    assert restored.gain == fly.gain


def test_evolution_improves_best() -> None:
    sc = _sc()
    flies = init_population(12, seed=5)
    best_hist = []
    for gen in range(4):
        res = evaluate_population(sc, flies)
        fits = [r["fitness"] for r in res]
        best_hist.append(min(fits))
        flies = evolve(flies, fits, elite_k=3, mutation=0.25, seed=100 + gen)
    assert best_hist[-1] <= best_hist[0] + 1e-6
    assert min(best_hist) < 200.0


def test_evolve_keeps_population_and_elite() -> None:
    sc = _sc()
    flies = init_population(10, seed=3)
    res = evaluate_population(sc, flies)
    fits = [r["fitness"] for r in res]
    nxt = evolve(flies, fits, elite_k=3, mutation=0.25, seed=11)
    assert len(nxt) == len(flies)
    best_old = flies[int(np.argmin(fits))]
    assert any(np.allclose(f.w, best_old.w) and f.kind == best_old.kind for f in nxt)
