"""Фаза 3: phys-параметры трёхстепенной модели в UI «Дополнительно».

Имена и дефолты полей TS (types.ts) обязаны зеркалить Python ScenarioIn —
прецедент сверки LAW_RU (test_glossary.py): независимых таблиц не держим,
расходимости ловим тестом. Плюс проверка, что все поля реально провязаны
в AdvancedScenarioSettings.tsx.
"""
from __future__ import annotations

import re
from pathlib import Path

from navedenie.app import ScenarioIn

REPO = Path(__file__).resolve().parents[1]


def _ts_defaults() -> dict[str, str]:
    src = (REPO / "web" / "src" / "types.ts").read_text(encoding="utf-8")
    block = re.search(r"DEFAULT_SCENARIO\s*[:=].*?\{(.*?)\n\}", src, re.S)
    assert block, "не найден DEFAULT_SCENARIO в types.ts"
    return dict(re.findall(r"(phys_\w+)\s*:\s*([^,\n]+)", block.group(1)))


def _ts_interface_fields() -> set[str]:
    src = (REPO / "web" / "src" / "types.ts").read_text(encoding="utf-8")
    return set(re.findall(r"(phys_\w+)\s*:\s*(?:boolean|number)", src))


def test_ts_phys_names_match_interface_and_python():
    ts = _ts_defaults()
    assert ts, "в DEFAULT_SCENARIO нет phys-полей"
    assert set(ts) == _ts_interface_fields(), "интерфейс Scenario и дефолты разошлись"
    assert set(ts) <= set(ScenarioIn.model_fields), (
        f"TS-поля неизвестны Python: {set(ts) - set(ScenarioIn.model_fields)}"
    )


def test_ts_phys_defaults_match_python():
    for name, ts_raw in _ts_defaults().items():
        ts_val = ts_raw.strip().rstrip(";")
        py = ScenarioIn.model_fields[name].default
        if isinstance(py, bool):
            assert ts_val in ("true", "false"), f"{name}: не bool-литерал {ts_val!r}"
            assert (ts_val == "true") == py, f"{name}: TS {ts_val} ≠ Python {py}"
        else:
            assert abs(float(ts_val) - float(py)) < 1e-12, f"{name}: TS {ts_val} ≠ Python {py}"


def test_settings_component_wires_every_phys_field():
    src = (REPO / "web" / "src" / "shell" / "AdvancedScenarioSettings.tsx").read_text(encoding="utf-8")
    wired = set(re.findall(r"set\('(phys_\w+)'", src))
    assert wired == set(_ts_defaults()), f"в UI не провязаны: {set(_ts_defaults()) - wired}"
    assert "point_mass_3dof" in src, "блок не условный по трёхстепенной модели"
