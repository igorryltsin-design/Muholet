"""Вероятностная зона перехвата (capture_zone с n_runs > 1): каждая ячейка сетки
«дальность × перегрузка цели» — серия Monte-Carlo прогонов со seed = seed_start + i
(та же стохастика движка, что в /api/science/monte-carlo: шум измерений ГСН).
Ячейка несёт p_hit с биномиальным 95 % ДИ (Уилсон), медиану R_min; бинарное поле
hit — большинство серии; бисекция границ в этом режиме честно пропускается
(refine_skipped), а детерминированный сценарий помечается series_deterministic.

Сценарий с дробными долями: pn_gsn + шум пеленга 0,7°, сфера срабатывания
150 м, дальности 5/7/9 км — сетка намеренно захватывает границу зоны, где
часть серии попадает, а часть нет (проверено фиксированными seed)."""

import pytest

from navedenie.app import CaptureZoneIn, capture_zone_endpoint
from navedenie.science import _wilson_ci

_MC = dict(mode="pn", law="pn_gsn", aspect="tail-chase", v_m=780.0, v_t=260.0,
           off_axis_m=420.0, n_max=30.0, pn_n=4.0, dt=0.02, t_max=30.0,
           kill_radius_m=150.0, noise_az_deg=0.7)
_RANGES = [5000.0, 7000.0, 9000.0]
_GS = [0.0, 8.0]


def _mc_scan(**kw) -> dict:
    base = dict(_MC, ranges_m=_RANGES, target_gs=_GS, n_runs=8, seed_start=1000)
    base.update(kw)
    return capture_zone_endpoint(CaptureZoneIn(**base))


@pytest.fixture(scope="module")
def scan() -> dict:
    return _mc_scan()


def _cells(s: dict):
    return [c for row in s["rows"] for c in row["cells"]]


def test_mc_fields_and_majority(scan):
    """Каждая ячейка серии: p_hit с ДИ Уилсона, медиана CPA, hit — большинство."""
    assert scan["n_runs"] == 8 and scan["seed_start"] == 1000
    assert scan["refine_skipped"] is True
    for c in _cells(scan):
        k = round(c["p_hit"] * 8)
        assert abs(c["p_hit"] * 8 - k) < 1e-9  # доля кратна 1/n_runs
        lo, hi = c["p_hit_ci95"]
        assert lo - 1e-9 <= c["p_hit"] <= hi + 1e-9
        assert c["hit"] == (c["p_hit"] >= 0.5)
        assert c["median_cpa_m"] > 0.0
    # refine пропущен честно: ни одна строка не получила границ
    for row in scan["rows"]:
        assert "boundary_m" not in row and "inner_boundary_m" not in row


def test_wilson_ci_matches_reference(scan):
    """Д И ячеек — ровно формула Уилсона для k/n при n=8 (сверка с тем же хелпером)."""
    for c in _cells(scan):
        k = round(c["p_hit"] * 8)
        ref_lo, ref_hi = _wilson_ci(k, 8)
        assert c["p_hit_ci95"] == [round(ref_lo, 4), round(ref_hi, 4)]


def test_fractional_p_hit_on_zone_boundary(scan):
    """Сетка захватила границу зоны: есть ячейки с 0 < p_hit < 1 (не бинарна)."""
    ps = [c["p_hit"] for c in _cells(scan)]
    assert any(0.0 < p < 1.0 for p in ps)


def test_noise_breaks_determinism(scan):
    """Со стохастикой серии не вырождены: хотя бы одна ячейка не deterministic."""
    assert scan["series_deterministic"] is False
    assert any(not c["deterministic_cell"] for c in _cells(scan))


def test_zero_stochastics_matches_single_run():
    """Детерминированный контроль: pn без шума и отказов → все серии побитово
    совпадают (series_deterministic, p_hit ∈ {0;1}) и мажоритарный hit совпадает
    с одиночным прогоном той же сетки."""
    det_kw = dict(_MC, law="pn", noise_az_deg=0.0)
    single = capture_zone_endpoint(CaptureZoneIn(**det_kw, ranges_m=_RANGES,
                                                 target_gs=_GS, refine=False))
    series = capture_zone_endpoint(CaptureZoneIn(**det_kw, ranges_m=_RANGES,
                                                 target_gs=_GS, n_runs=4,
                                                 seed_start=700))
    assert series["series_deterministic"] is True
    for r_s, r_m in zip(single["rows"], series["rows"]):
        for cs, cm in zip(r_s["cells"], r_m["cells"]):
            assert cm["p_hit"] in (0.0, 1.0)
            assert cm["hit"] == cs["hit"]
            assert abs(cm["r_min_m"] - cs["r_min_m"]) < 1e-9


def test_reproducible_by_seed_and_shifted_seed_differs():
    """Один seed_start — бит-в-бит повтор; сдвиг seed на боковой шум меняет доли."""
    a = _mc_scan()
    b = _mc_scan()
    assert a["rows"] == b["rows"]
    c = _mc_scan(seed_start=91000)
    assert any(ca["p_hit"] != cc["p_hit"]
               for ra, rc in zip(a["rows"], c["rows"])
               for ca, cc in zip(ra["cells"], rc["cells"]))


def test_single_run_shape_unchanged():
    """Регресс формы: n_runs=1 не несёт MC-полей ни в ответе, ни в ячейках."""
    r = capture_zone_endpoint(CaptureZoneIn(**dict(_MC, ranges_m=[6000.0],
                                                   target_gs=[5.0], refine=False)))
    assert "n_runs" not in r and "refine_skipped" not in r
    assert "seed_start" not in r and "series_deterministic" not in r
    for c in _cells(r):
        assert "p_hit" not in c and "deterministic_cell" not in c
        assert {"hit", "r_min_m", "t_end_s", "time_limited"} <= set(c)
