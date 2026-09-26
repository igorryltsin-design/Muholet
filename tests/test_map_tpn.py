"""Фаза 2: закон tpn («истинный МПС») в «Карте преимуществ» (/api/map).

map_advantage строит сетку «манёвр × закон»; tpn добавлен вторым законом.
Тесты проверяют состав сетки, метку из единого источника терминов и то, что
сетка реально различает команды tpn и pn (иначе новый закон — декорация).
"""
from __future__ import annotations

from navedenie.app import MapIn, map_advantage
from navedenie.glossary import LAWS

MANEUVERS = ["прямолинейно", "вираж", "змейка", "ножницы", "горка/пике"]
# быстрый профиль: дальнобойность не нужна, важна структура сетки
_FAST = dict(v_m=900.0, v_t=300.0, range_m=6000.0, dt=0.02, brain="stub", repeats=1)


def test_map_contains_five_laws_for_every_maneuver():
    rows = map_advantage(MapIn(**_FAST))["rows"]
    assert len(rows) == len(MANEUVERS) * 5
    for m in MANEUVERS:
        ids = {r["law_id"] for r in rows if r["maneuver"] == m}
        assert ids == {"pn", "tpn", "apn", "pure", "clos"}


def test_map_tpn_label_matches_glossary():
    rows = map_advantage(MapIn(**_FAST))["rows"]
    tpn = [r for r in rows if r["law_id"] == "tpn"]
    assert tpn, "нет строк tpn"
    # единый источник русских имён законов: метка карты = rus_short из LAWS
    assert {r["law"] for r in tpn} == {LAWS["tpn"].rus_short}


def test_map_tpn_cells_complete_and_ordered_like_pn():
    rows = map_advantage(MapIn(**_FAST))["rows"]
    for m in MANEUVERS:
        cell_pn = next(r for r in rows if r["maneuver"] == m and r["law_id"] == "pn")
        cell_tpn = next(r for r in rows if r["maneuver"] == m and r["law_id"] == "tpn")
        # ячейка tpn — та же схема сравнения: промах/попадание/энергия эталона
        for k in ("miss_pn", "hit_pn", "n_int_pn", "miss_fly", "advantage"):
            assert k in cell_tpn
        assert cell_tpn["miss_fly"] == cell_pn["miss_fly"]  # один и тот же мозг и сценарий


def test_map_tpn_differs_from_pn_somewhere():
    """Если ни в одной ячейке сетки tpn не отличается от pn — сетка не читает закон."""
    rows = map_advantage(MapIn(**_FAST))["rows"]
    d = [
        abs(next(r for r in rows if r["maneuver"] == m and r["law_id"] == "tpn")["miss_pn"]
             - next(r for r in rows if r["maneuver"] == m and r["law_id"] == "pn")["miss_pn"])
        for m in MANEUVERS
    ]
    assert max(d) > 0.05, f"tpn нигде не отличается от pn: {d}"
