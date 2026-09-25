"""Ночная смена №2: ансамбль-медиана, дистилляция, коэволюция."""

import numpy as np

from navedenie.circuit import EnsembleBrain, FlyCircuit
from navedenie.formula import brain_dn


def test_ensemble_returns_median_command() -> None:
    members = [FlyCircuit(kind="stub") for _ in range(3)]
    for i, m in enumerate(members):
        m.W_dn[1, :] = 0.0
        m.W_dn[1, 0] = float(i - 1)  # рыскание: −1, 0, +1 по beta3
    e = EnsembleBrain(members)
    e.reset()
    for m in members:
        m.state.feat = np.full(8, 0.5)
        m.step({"image": np.zeros((16, 16)), "lock": False, "az": 0, "el": 0}, 0.005)
        m.state.dn = np.array([np.tanh(m.W_dn[0] @ m.state.feat), np.tanh(m.W_dn[1] @ m.state.feat)])
    v = np.array([700.0, 0.0, 0.0])
    a_ens = e.accel_cmd(v, 30.0)
    a_members = [m.accel_cmd(v, 30.0) for m in members]
    median = np.median(np.stack(a_members), axis=0)
    assert np.allclose(a_ens, median, atol=1e-6)


def test_distill_lowers_error_on_grid() -> None:
    # маленькая реплика дистилляции: учитель — коннектом, ученик — линейная схема
    from navedenie.brain_store import get_circuit

    circuit = get_circuit("connectome")
    rng = np.random.default_rng(7)
    from navedenie.circuit import FEAT_DIM
    X = rng.uniform(-1.0, 1.0, size=(800, FEAT_DIM))
    T = brain_dn("connectome", circuit, X)
    w = np.zeros((2, FEAT_DIM))
    err0 = float(np.mean((np.tanh(X @ w.T) - T) ** 2))
    lr0, epochs = 0.8, 300
    for ep in range(epochs):
        lr = lr0 * (1.0 - ep / epochs) + 0.02
        err = T - np.tanh(X @ w.T)
        w += lr * (err.T @ X) / len(X)
        np.clip(w, -4.0, 4.0, out=w)
    err1 = float(np.mean((np.tanh(X @ w.T) - T) ** 2))
    assert err1 < err0 * 0.5  # дистилляция сходится: ошибка падает вдвое+
