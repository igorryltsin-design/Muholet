"""Научные прогоны: матрица переносимости, scaling-кривая, изоляция кеша весов."""

import numpy as np

from navedenie.science import transfer_matrix, scaling_curve
from navedenie import train as train_mod


def test_transfer_matrix_structure_and_cache_isolation() -> None:
    # быстрая реплика на схеме: структура, детерминизм, живой кеш не тронут
    train_mod._remember_best("stub", _fake_stub(), 100.0, 0.5)
    cache_before = {k: dict(v) for k, v in train_mod._best_cache.items()}
    res = transfer_matrix("stub", episodes=2, maneuvers=("turn", "weave"))
    assert res["kind"] == "stub"
    assert res["test_maneuvers"] == ["straight", "turn", "weave"]
    assert [r["train"] for r in res["rows"]] == ["turn", "weave"]
    for row in res["rows"]:
        assert set(row["tests"]) == {"straight", "turn", "weave"}
        assert all(np.isfinite(v) and v > 0 for v in row["tests"].values())
        assert np.isfinite(row["train_miss_med"])
    cache_after = {k: dict(v) for k, v in train_mod._best_cache.items()}
    assert cache_before == cache_after  # научный прогон не подменяет снапшоты стенда


def _fake_stub():
    from navedenie.circuit import FlyCircuit

    return FlyCircuit(kind="stub")


def test_transfer_matrix_deterministic() -> None:
    a = transfer_matrix("stub", episodes=2, maneuvers=("dive",))
    b = transfer_matrix("stub", episodes=2, maneuvers=("dive",))
    assert a["rows"] == b["rows"]  # свежие мозги + фиксированные зерна = тот же ответ


def test_transfer_trained_on_maneuver_beats_reflex_on_it() -> None:
    # обучение на вираже должно улучшить вираж относительно необученного рефлекса
    from navedenie.circuit import FlyCircuit
    from navedenie.train import _rollout_update
    from navedenie.science import _test_scenario

    fresh = FlyCircuit(kind="stub")
    before = float(
        np.median([_rollout_update(fresh, _test_scenario("turn", fresh.gain, a), lr=0.0)["miss"] for a in ("head-on", "beam")])
    )
    res = transfer_matrix("stub", episodes=6, maneuvers=("turn",))
    after = res["rows"][0]["tests"]["turn"]
    assert after < before, f"обучение по результату на вираже ({after} м) не улучшило рефлекс ({before} м)"


def test_scaling_curve_params_grow_with_pool() -> None:
    res = scaling_curve("full", sizes=(8, 24), episodes=2)
    assert res["kind"] == "full"
    params = [r["params"] for r in res["rows"]]
    assert params == sorted(params) and params[0] < params[1]
    for r in res["rows"]:
        # полный мозг: W_dn 2×pool + W_pool pool×4096
        assert r["params"] == 2 * r["size"] + r["size"] * 4096
        assert np.isfinite(r["miss_after"]) and np.isfinite(r["ref_dev_after"])
        assert 0.0 <= r["hit_rate_after"] <= 1.0


def test_scaling_connectome_counts_fx_channels() -> None:
    from navedenie.circuit import ConnectomeCircuit
    from navedenie.science import _n_params

    c = ConnectomeCircuit(channels=64)
    from navedenie.circuit import FEAT_DIM
    assert _n_params(c) == 2 * 64 + 64 * FEAT_DIM  # выход + синаптические каналы (пластичны у коннектома)
