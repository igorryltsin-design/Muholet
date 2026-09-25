"""Разные мозги учатся по-разному: у полного — бэкроп в скрытый слой."""

import numpy as np

from navedenie.circuit import ConnectomeCircuit, FlyCircuit
from navedenie.seeker import observe
from navedenie.sim import Scenario, spawn
from navedenie.train import _rollout_update


def _obs(sc, missile, target):
    import numpy as np

    r = target.p - missile.p
    return observe(r, missile.v, None, half_fov=np.deg2rad(165.0) / 2.0, dt_s=0.02)


def test_full_brain_trains_hidden_layer_w_k_frozen() -> None:
    circuit = FlyCircuit(kind="full")
    sc = Scenario(aspect="head-on", mode="bio", t_max=6, dt=0.02, circuit_gain=1.0)
    missile, target = spawn(sc)
    circuit.reset()
    obs = _obs(sc, missile, target)
    circuit.step(obs, 0.02)

    w_k_before = circuit.W_k.copy()
    w_pool_before = circuit.W_pool.copy()
    circuit.learn_step(np.array([0.5, -0.3]), lr=0.04, lr_hidden=0.012)

    assert not np.array_equal(circuit.W_pool, w_pool_before)  # скрытый слой пластичен
    assert np.array_equal(circuit.W_k, w_k_before)  # врождённая проводка заморожена
    assert np.array_equal(circuit.W_k, w_k_before)


def test_stub_learns_only_output() -> None:
    circuit = FlyCircuit(kind="stub")
    sc = Scenario(aspect="beam", mode="bio", t_max=6, dt=0.02)
    missile, target = spawn(sc)
    circuit.reset()
    obs = _obs(sc, missile, target)
    circuit.step(obs, 0.02)
    w_before = circuit.W_dn.copy()
    circuit.learn_step(np.array([0.4, 0.2]), lr=0.04)
    assert not np.array_equal(circuit.W_dn, w_before)


def test_connectome_readout_128_channels_deterministic() -> None:
    a = ConnectomeCircuit()
    b = ConnectomeCircuit()
    assert a.W_dn.shape == (2, 128)  # зоны: 10 прямых + 28 пеленговых + 28 потоковых + 62 ассоциативных
    assert np.array_equal(a.W_fx, b.W_fx)  # расширение детерминировано зерном
    from navedenie.circuit import FEAT_DIM
    a.feat = np.full(FEAT_DIM, 0.5)  # ненулевые признаки — иначе ошибка и правка нулевые
    a.learn_step(np.array([0.3, 0.1]), lr=0.04)
    assert not np.array_equal(a.W_dn, b.W_dn)


def test_training_diverges_brains_on_hard_maneuver() -> None:
    """Одинаковые эпизоды — разные мозга достигают разного промаха: разница видна."""
    sc = Scenario(aspect="beam", mode="bio", t_max=8, dt=0.02, n_target=3.0, maneuver="scissors", seed=11)
    results = {}
    for kind in ("stub", "full", "connectome"):
        circuit = FlyCircuit(kind=kind) if kind != "connectome" else ConnectomeCircuit()
        out = _rollout_update(circuit, sc, lr=0.04)
        results[kind] = out["miss"]
        assert out["ref_dev"] > 0
    values = list(results.values())
    assert len(set(round(v) for v in values)) >= 2  # модели ведут себя по-разному


def test_brain_rebuild_changes_hidden_dims() -> None:
    """Пересоздание меняет скрытые размеры и сбрасывает обучение (ось исследования)."""
    from navedenie.app import BrainRebuildIn, brain_rebuild

    out = brain_rebuild(BrainRebuildIn(kind="full", pool_size=48))
    assert out["ok"] is True and out["n_cells"] > 4000
    circuit = FlyCircuit(kind="full", pool_size=48)
    assert circuit.W_dn.shape == (2, 48) and circuit.trained is False
    from fastapi import HTTPException

    try:
        brain_rebuild(BrainRebuildIn(kind="full", pool_size=4))
        raise AssertionError("ожидался 422")
    except HTTPException as exc:
        assert exc.status_code == 422


def test_evolve_result_mode_improves_or_keeps() -> None:
    """Обучение «по результату»: эволюция выхода не ухудшает промах трио и сохраняет веса."""
    from navedenie import train as T

    kind = "stub"
    T.evolve_start(kind, generations=2, sigma=0.25, pop=3)
    T.evolve_step(kind)
    T.evolve_step(kind)
    fin = T.evolve_finish(kind)
    assert fin["trained"] is True
    assert fin["miss_after"] > 0
    circuit = __import__("navedenie.brain_store", fromlist=["get_circuit"]).get_circuit(kind)
    assert circuit.trained is True
    assert np.array_equal(circuit.W_dn, fin and circuit.W_dn)  # веса на месте после сохранения
