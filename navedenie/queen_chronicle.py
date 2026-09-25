"""«Хроника войн»: память о кампаниях самообучения, переживающая рестарт.

Лог поколений живёт в фоновой задаче — стенд перезапустился, и от кампании не
остаётся ничего: ни кривой взятий, ни того, чем она кончилась. Для игры важен
длинный след: виден ли рост силы от войны к войне. Здесь каждая доигранная
кампания server-королевы получает запись в `data/queen_chronicle.json` (голый
список, как полка ринга), с кривой взятий по поколениям и финалом — что ушло в
веса и на ринг. Запись пишет только завершённую кампанию с хотя бы одним
досчитанным поколением: прерванная на нуле война — не событие.
"""

from __future__ import annotations

import json
import time
from typing import Any

CHRONICLE_FILE = "queen_chronicle.json"
CHRON_MAX = 20
_CURVE_FIELDS = ("gen", "p_hit_ring", "exam_p_hit", "missile_best", "evader_best")


def _data_dir():
    from navedenie import circuit

    return circuit.DATA_DIR


def _load() -> list[dict]:
    path = _data_dir() / CHRONICLE_FILE
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return raw if isinstance(raw, list) else []


def _store(entries: list[dict]) -> None:
    path = _data_dir() / CHRONICLE_FILE
    path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")


def list_campaigns() -> list[dict]:
    """Кампании в порядке записи (старшие первыми)."""
    return _load()


def _curve(log: list[dict]) -> list[dict]:
    return [{k: row.get(k) for k in _CURVE_FIELDS} for row in log or []]


def record(snap: dict[str, Any]) -> dict | None:
    """Взять снимок фоновой задачи (та же форма, что отдаёт GET /api/queen/train)
    и положить в хронику. Пустой кампании не бывает: без досчитанного поколения
    воевать нечем, и запись молча пропускается."""
    done = int(snap.get("generations_done") or 0)
    if done < 1:
        return None
    curve = _curve(list(snap.get("log") or []))
    if not curve:
        return None
    entries = _load()
    # id — преемственный максимум, как на полке ринга: после потолка длина
    # стоит на месте, и len+1 повторял бы уже занятые номера
    next_id = max((int(e.get("id", 0)) for e in entries), default=0) + 1
    ring = (snap.get("saved") or {}).get("ring") or {}
    entry = {
        "id": next_id,
        "finished_at": float(snap.get("finished_at") or time.time()),
        "label": (str(ring.get("label") or "").strip() or f"Кампания {next_id}")[:40],
        "generations": done,
        "planned": int(snap.get("generations") or 0),
        "pop": int(snap.get("pop") or 0),
        "seed": int(snap.get("seed") or 0),
        "seconds": float(snap.get("seconds") or 0.0),
        "scenario": dict(snap.get("scenario") or {}),
        "inherited": [int(i) for i in (snap.get("inherit") or [])],
        "curve": curve,
        "p_hit_first": curve[0].get("p_hit_ring"),
        "p_hit_last": curve[-1].get("p_hit_ring"),
        "ring_id": ring.get("id"),
        "shelved": bool(ring),
        "applied": bool((snap.get("saved") or {}).get("weights")),
        "stopped": bool(snap.get("stop_requested")),
        "error": snap.get("error"),
    }
    entries.append(entry)
    _store(entries[-CHRON_MAX:])
    return entry


def delete_campaign(campaign_id: int) -> bool:
    entries = _load()
    keep = [e for e in entries if int(e.get("id", -1)) != int(campaign_id)]
    if len(keep) == len(entries):
        return False
    _store(keep)
    return True
