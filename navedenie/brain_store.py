"""Текущий мозг процесса: схема, полный, коннектом; веса, обучение."""

from __future__ import annotations

import json

import numpy as np

from navedenie.circuit import (
    CONNECTOME_REGIONS,
    DATA_DIR,
    FLYWIRE_NEURONS,
    FLYWIRE_SYNAPSES,
    ConnectomeCircuit,
    FlyCircuit,
)

_circuits: dict[str, object] = {}
_evader: FlyCircuit | None = None

EVADER_WEIGHTS_FILE = "weights_evader.npz"


def get_evader_circuit(tau_s: float = 0.025, gain: float = 1.0) -> FlyCircuit:
    """Мозг-уклонист цели (фаза 3 дуэли): тот же stub-контур, отдельные веса
    weights_evader.npz; без сохранённых — врождённый рефлекс «к пеленгу»."""
    global _evader
    path = DATA_DIR / EVADER_WEIGHTS_FILE
    if _evader is None:
        _evader = FlyCircuit(tau_s=tau_s, gain=gain, kind="stub")
        _evader.load(path)
    _evader.tau = tau_s
    _evader.gain = gain
    _evader.reset()
    return _evader


def set_evader_circuit(circuit: FlyCircuit) -> None:
    """Зарегистрировать выученный мозг-уклонист и сохранить его веса."""
    global _evader
    _evader = circuit
    circuit.save(DATA_DIR / EVADER_WEIGHTS_FILE)


def evader_status() -> dict:
    path = DATA_DIR / EVADER_WEIGHTS_FILE
    trained = bool(_evader.trained) if _evader is not None else False
    if _evader is None and path.exists():
        with np.load(path, allow_pickle=False) as blob:
            trained = bool(blob["trained"][0]) if "trained" in blob else True
    return {
        "n_cells": int(_evader.n_cells) if _evader is not None else 279,
        "trained": trained,
        "kind": "stub",
        "weights_file": path.name,
    }


def get_circuit(kind: str = "stub", tau_s: float = 0.025, gain: float = 1.0):
    key = kind if kind in ("stub", "full", "connectome") else "stub"
    circuit = _circuits.get(key)
    if circuit is None:
        if key == "connectome":
            circuit = ConnectomeCircuit(tau_s=tau_s, gain=gain)
            circuit.load()
        else:
            circuit = FlyCircuit(tau_s=tau_s, gain=gain, kind=key)
            circuit.load()
        _circuits[key] = circuit
    circuit.tau = tau_s
    circuit.gain = gain
    circuit.reset()
    return circuit


def replace(kind: str, circuit) -> None:
    """Подменить экземпляр мозга (после пересоздания с новыми размерами)."""
    _circuits[kind] = circuit


def set_circuit(circuit) -> None:
    _circuits[circuit.kind] = circuit
    circuit.save()


def status() -> dict:
    from navedenie.circuit import FEATURE_SCHEMA_VERSION

    out = {"feature_schema_version": FEATURE_SCHEMA_VERSION}
    for kind in ("stub", "full"):
        c = _circuits.get(kind) or FlyCircuit(kind=kind)
        if kind not in _circuits:
            c.load()
        out[kind] = {
            "n_cells": c.n_cells,
            "trained": c.trained,
            "kind": c.kind,
        }
    # коннектом не строим ради статуса: размеры известны из конфигурации регионов,
    # флаг обучения читаем прямо из сохранённых весов, источник проводки — из circuit_v1
    n_cells = sum(n for _, n, _ in CONNECTOME_REGIONS)
    if _circuits.get("connectome") is not None:
        conn_trained = bool(_circuits["connectome"].trained)
        conn_wiring = {
            "real": bool(getattr(_circuits["connectome"], "real_wiring", False)),
            "n_synapses": int(getattr(_circuits["connectome"], "n_synapses", 0)),
        }
    else:
        conn_trained = False
        wpath = DATA_DIR / "weights_connectome.npz"
        if wpath.exists():
            with np.load(wpath, allow_pickle=False) as blob:
                conn_trained = bool(blob["trained"][0])
        conn_wiring = _read_real_wiring_meta()
    out["connectome"] = {
        "n_cells": n_cells,
        "trained": conn_trained,
        "kind": "connectome",
        "flywire_neurons": FLYWIRE_NEURONS,
        "flywire_synapses": FLYWIRE_SYNAPSES,
        "wiring": conn_wiring,
    }
    out["evader"] = evader_status()
    return out


def _read_real_wiring_meta() -> dict:
    """Источник проводки коннектома без построения сети: из circuit_v1.npz."""
    path = DATA_DIR / "circuit_v1.npz"
    if not path.exists():
        return {"real": False, "n_synapses": 0}
    try:
        with np.load(path, allow_pickle=False) as blob:
            meta = json.loads(str(blob["meta"]))
        return {"real": True, "n_synapses": int(meta.get("n_synapses", 0)), "source": meta.get("source", "")}
    except Exception:  # noqa: BLE001
        return {"real": False, "n_synapses": 0}
