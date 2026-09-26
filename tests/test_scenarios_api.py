"""Фаза 6: серверная библиотека сценариев (/api/scenarios, data/scenarios).

CRUD по образцу экспериментов: POST сохраняет нормализованный ScenarioIn,
GET/DELETE — по имени, прохождение имён через _NAME_RE (никаких «/» и «..»).
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import navedenie.app as api


@pytest.fixture()
def scn_dir(tmp_path, monkeypatch):
    d = tmp_path / "scenarios"
    monkeypatch.setattr(api, "_SCN_DIR", d)
    return d


def _body(name: str, **over):
    return api.ScenarioSaveIn(name=name, scenario=api.ScenarioIn(**over))


def test_roundtrip_post_get_list(scn_dir):
    assert api.scenarios_list() == {"scenarios": []}
    api.scenario_save(_body("лобовая 90", aspect="free", free_tx=12345.0, t_max=17.0))
    items = api.scenarios_list()["scenarios"]
    assert [i["name"] for i in items] == ["лобовая 90"]
    assert items[0]["saved_at"]
    got = api.scenario_get("лобовая 90")
    assert got["scenario"]["aspect"] == "free"
    assert got["scenario"]["free_tx"] == 12345.0
    assert got["scenario"]["t_max"] == 17.0
    # файл — plain JSON без секретов и лишнего
    raw = (scn_dir / "лобовая 90.json").read_text(encoding="utf-8")
    assert "free_tx" in raw and "token" not in raw.lower()


def test_overwrite_same_name(scn_dir):
    api.scenario_save(_body("рабочий", v_m=700.0))
    api.scenario_save(_body("рабочий", v_m=900.0))
    assert len(api.scenarios_list()["scenarios"]) == 1
    assert api.scenario_get("рабочий")["scenario"]["v_m"] == 900.0


@pytest.mark.parametrize("bad", ["a/b", "..", "../etc", "x" * 65, "имя; rm", ""])
def test_bad_names_rejected(scn_dir, bad):
    with pytest.raises(HTTPException) as e:
        api.scenario_save(_body(bad))
    assert e.value.status_code == 422
    with pytest.raises(HTTPException) as e:
        api.scenario_delete(bad)
    assert e.value.status_code == 422
    assert not scn_dir.exists() or list(scn_dir.glob("*.json")) == []


def test_scenario_body_validated(scn_dir):
    with pytest.raises(ValidationError):
        api.ScenarioSaveIn(name="х", scenario={"t_max": "не число", "v_m": "тоже"})


def test_get_and_delete_missing_404(scn_dir):
    with pytest.raises(HTTPException) as e:
        api.scenario_get("нет такого")
    assert e.value.status_code == 404
    with pytest.raises(HTTPException) as e:
        api.scenario_delete("нет такого")
    assert e.value.status_code == 404
    api.scenario_save(_body("-temp"))
    assert api.scenario_delete("-temp")["ok"] is True
    assert api.scenarios_list()["scenarios"] == []
