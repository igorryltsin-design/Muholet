from navedenie.app import BrainImportIn, brain_export, brain_import
from navedenie.circuit import FEAT_DIM


def test_brain_export_import_roundtrip() -> None:
    exported = brain_export("stub")
    assert exported["kind"] == "stub"
    assert len(exported["w"]) == 2 and len(exported["w"][0]) == FEAT_DIM
    assert "gain" in exported
    assert exported["wiring"]["seed"] == 23

    out = brain_import(
        BrainImportIn(kind="stub", w=exported["w"], trained=False, gain=exported["gain"], wiring=exported["wiring"])
    )
    assert out["ok"] is True
    assert out["trained"] is False

    again = brain_export("stub")
    assert again["trained"] is False
    assert again["w"] == exported["w"]


def test_brain_import_rejects_wrong_shape() -> None:
    out = brain_import(BrainImportIn(kind="stub", w=[[1.0, 2.0]], trained=True))
    assert out["ok"] is False
    assert "ожидалась матрица" in out["error"]


def test_connectome_export_contains_wiring_recipe() -> None:
    exported = brain_export("connectome")
    wiring = exported["wiring"]
    assert wiring["seed"] == 42
    assert wiring["basis"] == 192
    assert wiring["n_synapses"] > 1_000_000  # миллионы РЕАЛЬНЫХ синапсов FlyWire
    assert len(wiring["hash"]) == 16
    assert len(wiring["regions"]) == 8

    ok = brain_import(BrainImportIn(kind="connectome", w=exported["w"], wiring=wiring))
    assert ok["ok"] is True

    tampered = dict(wiring)
    tampered["hash"] = "0" * 16
    bad = brain_import(BrainImportIn(kind="connectome", w=exported["w"], wiring=tampered))
    assert bad["ok"] is False
    assert "проводка" in bad["error"]


def test_full_brain_export_carries_hidden_blocks() -> None:
    exported = brain_export("full")
    wiring = exported["wiring"]
    assert "blocks" in wiring
    assert set(wiring["blocks"]) == {"W_k", "W_pool"}
    assert wiring["hash"]

    ok = brain_import(BrainImportIn(kind="full", w=exported["w"], wiring=wiring))
    assert ok["ok"] is True
    restored = brain_export("full")
    assert restored["wiring"]["hash"] == wiring["hash"]
