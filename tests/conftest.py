"""Тесты не трогают прод-веса в data/: все save/load мозгов идут во временный каталог.

Без этой песочницы тесты brain_import/train перезаписывали data/weights_*.npz —
после прогона pytest на стенде оказывалась необученная схема.
"""

import pytest


@pytest.fixture(autouse=True)
def _sandbox_data_dir(tmp_path, monkeypatch):
    from navedenie import brain_store, circuit

    monkeypatch.setattr(circuit, "DATA_DIR", tmp_path)
    monkeypatch.setattr(brain_store, "DATA_DIR", tmp_path)
    yield
