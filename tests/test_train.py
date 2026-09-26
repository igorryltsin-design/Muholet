import numpy as np

from navedenie.circuit import FlyCircuit, default_W
from navedenie.train import train


def test_training_changes_weights_and_reports_miss():
    start = default_W().copy()
    report = train(kind="stub", episodes=9, lr=0.05)
    assert report["trained"] is True
    assert report["n_cells"] >= 80
    assert report["episodes"] == 9
    circuit = FlyCircuit(kind="stub")
    circuit.load()
    assert circuit.trained
    assert float(np.linalg.norm(circuit.W_dn - start)) > 1e-4
    assert np.isfinite(report["miss_after"])
