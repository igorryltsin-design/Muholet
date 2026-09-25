"""§4: канонический словарь — единый источник терминов и автоматическая
проверка согласованности Python↔TypeScript.

Проверяется, что:
  - словарь самосогласован (каждый закон/метрика/термин имеет имя, единицу, статус);
  - в пользовательских подписях нет буквы K для навигационного коэффициента (§3);
  - закон ZEM вынесен из законов в метрики (§13);
  - таблица названий TypeScript (web/src/shell/labels.ts) совпадает с Python LAW_LABEL.
"""

from __future__ import annotations

import re
from pathlib import Path

from navedenie.glossary import LAWS, LAW_LABEL, METRICS, SENSOR, TERMS

REPO = Path(__file__).resolve().parents[1]


def test_glossary_fields_present() -> None:
    for t in TERMS.values():
        assert t.id and t.rus_full, t
        assert t.status, f"{t.id}: пустой статус"
        assert t.unit, f"{t.id}: пустая единица"


def test_navigation_coefficient_is_n_not_k() -> None:
    """§3: в пользовательских названиях законов — навигационный коэффициент N,
    а не «коэффициент K» / «K_eff» / «расписание K»."""
    for lid, label in LAW_LABEL.items():
        assert not re.search(r"K_eff|коэффициент\s*K|переменный\s*K|расписание\s*K", label), (lid, label)
    # pn_sched назван через «навигационным коэффициентом N» (§5.5)
    assert "навигационным коэффициентом N" in LAW_LABEL["pn_sched_oracle"]
    assert "навигационным коэффициентом N" in LAW_LABEL["pn_sched_sensor"]


def test_zem_is_metric_not_law() -> None:
    """§13: ZEM — прогнозная метрика, а не закон наведения."""
    assert not any("zem" in k.lower() for k in LAWS)
    assert "h0" in METRICS  # «Прогнозируемый промах без дальнейшего управления»
    assert METRICS["h0"].status == "диагностический показатель, не закон наведения"
    assert METRICS["h0"].rus_full == "Прогнозируемый промах без дальнейшего управления"


def test_law_ids_stable() -> None:
    """§22: внутренние id законов не ломаются (legacy-совместимость)."""
    assert set(LAW_LABEL) == set(LAWS)
    for legacy in ("pn", "apn", "pure", "clos", "pn_gsn"):
        assert legacy in LAWS


def _ts_law_ru() -> dict[str, str]:
    src = (REPO / "web" / "src" / "shell" / "labels.ts").read_text(encoding="utf-8")
    block = re.search(r"LAW_RU\s*:\s*Record<string,\s*string>\s*=\s*\{(.*?)\}", src, re.S)
    assert block, "не найден LAW_RU в labels.ts"
    return dict(re.findall(r"(\w+)\s*:\s*'([^']*)'", block.group(1)))


def test_ts_law_labels_match_python() -> None:
    """§4: в Python и TypeScript не должно быть независимых таблиц названий —
    сверяем их автоматически."""
    ts = _ts_law_ru()
    assert set(ts) == set(LAW_LABEL), "наборы id законов в TS и Python разошлись"
    for lid, ru in LAW_LABEL.items():
        assert ts[lid] == ru, f"{lid}: TS {ts[lid]!r} ≠ Python {ru!r}"
