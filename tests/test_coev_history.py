"""Фаза 4: история поколений коэволюции (/api/coevolve/*).

step теперь пишет в историю не только фитнес сильнейшей цели (miss_fly),
но и медиану популяции и флаг «муха дообучена» (ставит train_fly).
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from navedenie.app import coevolve_reset, coevolve_start, coevolve_step, coevolve_train_fly


def test_step_history_grows_monotonically_with_gen():
    coevolve_start()
    h1 = list(coevolve_step()["history"])  # копия: step отдаёт живой список состояния
    h2 = coevolve_step()["history"]
    assert len(h2) == len(h1) + 1 == 2
    assert [e["gen"] for e in h2] == [1, 2]
    coevolve_reset()


def test_entry_has_median_and_retrain_fields():
    coevolve_start()
    e = coevolve_step()["history"][-1]
    assert e["retrained"] is False
    assert "median_miss" in e and "miss_fly" in e
    # медиана популяции не опаснее сильнейшей цели
    assert e["median_miss"] <= e["miss_fly"] + 1e-9
    coevolve_reset()


def test_reset_empties_history_and_step_needs_start():
    coevolve_start()
    coevolve_step()
    coevolve_reset()
    with pytest.raises(HTTPException) as exc:
        coevolve_step()
    assert exc.value.status_code == 409
    # старт после сброса — новая история, старая не видна
    coevolve_start()
    assert coevolve_step()["gen"] == 1
    coevolve_reset()


def test_train_fly_marks_last_generation_retrained():
    coevolve_start()
    coevolve_step()
    coevolve_step()
    out = coevolve_train_fly({"generations": 1, "rounds": 1})
    marked = [e for e in out["history"] if e.get("retrained")]
    assert len(marked) == 1 and marked[-1]["gen"] == 2
    assert {"fly_before", "fly_after"} <= set(marked[-1])
    assert out["fly_miss_after"] >= 0.0
    coevolve_reset()
