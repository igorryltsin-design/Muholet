"""Monte-Carlo рассеивание (POST /api/science/monte-carlo): статистика точки
прицела при возмущающих факторах — классический приём оценки боевого рассеивания
(Weapons Handbook: dispersion of aim point). Реализация: n_runs прогонов одного
сценария с seed = seed_start + i; источники стохастики движка — шум измерений
ГСН (noise_az_deg/noise_range_m), срыв сопровождения (lock_drop_p), отказы
сетчатки (retina_death_p) — все тянутся из np.random.default_rng(seed·977+13).

Тесты сверяют статистику с аналитикой: детерминированный контроль (нулевая
стохастика → СКО = 0 побитно), формула Уилсона на примере 57/100 →
(0.47215; 0.66267), согласованность p_hit с per_run, воспроизводимость серии."""

import numpy as np
import pytest

from navedenie.app import MonteCarloIn, monte_carlo_endpoint
from navedenie.science import _wilson_ci


def _mc(**kw) -> dict:
    base = dict(mode="pn", law="pn_gsn", maneuver="weave", aspect="head-on",
                t_max=8.0, n_runs=8, seed_start=2000)
    base.update(kw)
    return monte_carlo_endpoint(MonteCarloIn(**base))


def test_wilson_ci_textbook():
    """Уилсон для 57/100: учебный пример — (0.4704, 0.6650)."""
    lo, hi = _wilson_ci(57, 100)
    # точные значения формулы (p + z²/2n ± z√(p(1−p)/n + z²/4n²))/(1 + z²/n), z=1.959964
    assert abs(lo - 0.47215) < 1e-4 and abs(hi - 0.66267) < 1e-4


def test_wilson_ci_edges():
    """k=0 и k=n: интервал лежит в [0,1] и не вырождается."""
    lo0, hi0 = _wilson_ci(0, 100)
    assert lo0 < 1e-9 and 0.0 < hi0 < 0.1
    lo1, hi1 = _wilson_ci(100, 100)
    assert hi1 == 1.0 and 0.9 < lo1 < 1.0


def test_zero_stochastic_is_deterministic():
    """Детерминированный контроль: без шума/срывов все прогоны совпадают → СКО = 0."""
    r = _mc(noise_az_deg=0.0, noise_range_m=0.0)
    assert r["deterministic"] is True
    assert r["r_min_std_m"] == 0.0
    assert len({p["r_min_m"] for p in r["per_run"]}) == 1


def test_bearing_noise_creates_dispersion():
    """Шум пеленга 2° порождает ненулевой разброс R_min (СКО > 0, квартили различаются)."""
    r = _mc(noise_az_deg=2.0)
    assert r["deterministic"] is False
    assert r["r_min_std_m"] > 0.0
    assert r["r_min_p90_m"] > r["r_min_median_m"]


def test_summary_consistent_with_per_run():
    """Сводка согласована с рядом прогонов: p_hit = доля попаданий, медиана — из выборки."""
    r = _mc(noise_az_deg=1.5)
    k = sum(1 for p in r["per_run"] if p["hit"])
    assert r["p_hit"] == round(k / r["n_runs"], 4)
    lo, hi = _wilson_ci(k, r["n_runs"])
    assert r["p_hit_ci95"] == [round(lo, 4), round(hi, 4)]
    arr = np.array([p["r_min_m"] for p in r["per_run"]])
    assert r["cep50_m"] == round(float(np.median(arr)), 3)
    assert r["r_95_m"] == pytest.approx(round(float(np.percentile(arr, 95)), 3))
    # CEP_50 ≤ R_95: 50 %-радиус никогда не больше 95 %-радиуса
    assert r["cep50_m"] <= r["r_95_m"]


def test_series_reproducible_by_seed_start():
    """Один и тот же seed_start — воспроизводимая серия (все числа идентичны)."""
    a = _mc(noise_az_deg=2.0, seed_start=777, n_runs=4)
    b = _mc(noise_az_deg=2.0, seed_start=777, n_runs=4)
    assert a["per_run"] == b["per_run"]
    assert a["r_min_std_m"] == b["r_min_std_m"]
    c = _mc(noise_az_deg=2.0, seed_start=778, n_runs=4)
    assert c["per_run"] != a["per_run"]  # сдвиг seed — другая выборка
