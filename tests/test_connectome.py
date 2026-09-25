import time

import numpy as np

from navedenie.circuit import ACT_BINS, CONNECTOME_REGIONS, ConnectomeCircuit, FlyCircuit
from navedenie.sim import Scenario
from navedenie.swarm import canonical_scenarios, sample_generation_scenario


def _obs(lock: bool = True, spot: float = 1.0) -> dict:
    img = np.zeros((8, 8))
    img[3:5, 3:5] = spot
    return {
        "az": 0.2 if lock else 0.0,
        "el": -0.1 if lock else 0.0,
        "size": 0.1,
        "az_dot": 0.3 if lock else 0.0,
        "el_dot": -0.2 if lock else 0.0,
        "size_dot": 0.05 if lock else 0.0,
        "lock": lock,
        "image": img,
    }


def test_connectome_scale_and_activity() -> None:
    c = ConnectomeCircuit()
    assert c.n_cells == sum(n for _, n, _ in CONNECTOME_REGIONS)
    assert c.n_cells > 100_000
    assert c.n_synapses > 1_000_000  # реальная проводка FlyWire: миллионы синапсов
    for _ in range(30):
        c.step(_obs(), 0.02)
    assert float(np.mean(np.abs(c.state.vecs["ламина"]))) > 0.01
    assert float(np.mean(np.abs(c.state.vecs["медулла"]))) > 0.001
    snap = c.state.snapshot(kind="connectome", n_cells=c.n_cells, trained=False)
    assert snap["n_neurons"] == c.n_cells
    assert len(snap["regions"]) == len(CONNECTOME_REGIONS)
    assert sum(r["n"] for r in snap["regions"]) == c.n_cells
    assert 2700 <= len(snap["act_b64"]) <= 2740  # 2048 байт в base64
    a = c.accel_cmd(np.array([700.0, 0, 0]), 30.0)
    assert np.isfinite(a).all() and float(np.linalg.norm(a)) <= 30 * 9.81 + 1e-6


def test_connectome_eye_responds_to_image() -> None:
    c = ConnectomeCircuit()
    c.step(_obs(spot=0.0), 0.02)
    dim = float(np.mean(np.abs(c.state.vecs["ламина"])))
    c.reset()
    for _ in range(10):
        c.step(_obs(spot=1.0), 0.02)
    lit = float(np.mean(np.abs(c.state.vecs["ламина"])))
    assert lit > dim


def test_connectome_step_speed() -> None:
    c = ConnectomeCircuit()
    t0 = time.perf_counter()
    for _ in range(60):
        c.step(_obs(), 0.02)
    assert time.perf_counter() - t0 < 5.0


def test_stub_activity_in_snapshot() -> None:
    c = FlyCircuit(kind="stub")
    c.step(_obs(), 0.02)
    snap = c.state.snapshot(kind="stub", n_cells=c.n_cells, trained=False)
    assert snap["n_neurons"] == sum(v.size for _, v in c.state.neuron_vectors())
    assert 0 < len(snap["regions"]) <= 12
    assert len(snap["act_b64"]) > 100


def test_generation_scenarios_differ() -> None:
    base = Scenario()
    s1a = sample_generation_scenario(base, gen=0, seed=7)
    s1b = sample_generation_scenario(base, gen=0, seed=7)
    s2 = sample_generation_scenario(base, gen=1, seed=7)
    s5 = sample_generation_scenario(base, gen=5, seed=7)
    assert s1a == s1b  # детерминизм
    assert s1a != s2  # новое поколение — новая геометрия
    assert s2.aspect != s5.aspect or s2.range_m != s5.range_m
    canon = canonical_scenarios(base)
    assert [c.aspect for c in canon] == ["head-on", "beam", "tail-chase"]


def test_connectome_bins_bounded() -> None:
    c = ConnectomeCircuit()
    for _ in range(5):
        c.step(_obs(), 0.02)
    snap = c.state.snapshot(kind="connectome", n_cells=c.n_cells, trained=False)
    raw = np.frombuffer(np.asarray(bytearray(__import__("base64").b64decode(snap["act_b64"]))), dtype=np.uint8)
    assert raw.size == ACT_BINS
    assert int(raw.max()) <= 255
