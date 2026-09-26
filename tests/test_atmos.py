"""Тесты стандартной атмосферы ICAO/US1976 (navedenie/atmos.py) и её внедрения
в трёхстепенную модель (point_mass_3dof, phys_atmos).

Сверяемся с независимыми величинами:
  - определение ρ(0) = 1.225 кг/м³ (стандарт);
  - опубликованные опорные плотности слоёв US1976 (ρ на 11 и 20 км);
  - каноническая тропосферная формула ICAO ρ = ρ₀·(1 − 2.25577·10⁻⁵·H)^4.25588;
  - аналитическое решение торможения квадратичным сопротивлением v(t)=v₀/(1+k·v₀·t)
    (тяга/тяжесть выключены, ρ постоянна на слое-эталоне).
"""

from __future__ import annotations

import math

import numpy as np

from navedenie.atmos import (
    RHO_SEA_LEVEL,
    air_density,
    air_temperature,
)
from navedenie.physics import FlightState, PhysicsParams, drag_accel, rho_at_h, step_physics
from navedenie.sim import Scenario
from navedenie.engine import collect


# ── сама атмосферная модель ────────────────────────────────────────────────


def test_sea_level_is_definition() -> None:
    assert air_density(0.0) == RHO_SEA_LEVEL == 1.225


def test_us1976_layer_bases() -> None:
    """Опубликованные опорные плотности слоёв US1976 (геопотенциальные высоты)."""
    assert math.isclose(air_density(11000.0), 0.363916, rel_tol=1e-4)
    assert math.isclose(air_density(20000.0), 0.088035, rel_tol=1e-4)


def test_troposphere_matches_ICAO_power_law() -> None:
    """Каноническая формула ICAO для тропосферы 0…11 км (независимый источник констант):
    ρ = 1.225·(1 − 2.25577·10⁻⁵·H)^4.25588."""
    for h in (0.0, 1000.0, 4500.0, 9000.0, 11000.0):
        ref = 1.225 * (1.0 - 2.25577e-5 * h) ** 4.25588
        assert math.isclose(air_density(h), ref, rel_tol=1e-5), h
    # табличное значение на 5 км — 0.73643 кг/м³
    assert math.isclose(air_density(5000.0), 0.73643, rel_tol=3e-3)


def test_temperature_profile_layers() -> None:
    assert math.isclose(air_temperature(0.0), 288.15, abs_tol=1e-9)
    assert math.isclose(air_temperature(9000.0), 288.15 - 6.5 * 9, abs_tol=1e-9)
    assert math.isclose(air_temperature(11000.0), 216.65, abs_tol=1e-9)
    assert math.isclose(air_temperature(15000.0), 216.65, abs_tol=1e-9)  # изотермический слой
    assert math.isclose(air_temperature(25000.0), 216.65 + 1.0 * 5, abs_tol=1e-9)
    assert math.isclose(air_temperature(40000.0), 228.65 + 2.8 * 8, abs_tol=1e-9)
    assert math.isclose(air_temperature(75000.0), 214.65, abs_tol=1e-9)


def test_density_continuous_at_layer_bases() -> None:
    """Нет скачков на границах слоёв 11/20/32/47/51/71 км: разность ρ на ±1 м
    не превосходит естественного градиента (~0.2 % на 1 м в верхних слоях)."""
    for hb in (11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0):
        lo, hi = air_density(hb - 1.0), air_density(hb + 1.0)
        assert abs(lo - hi) < 5e-3 * hi, hb


def test_density_monotone_and_positive() -> None:
    hs = [float(h) for h in range(0, 84001, 250)]
    rhos = [air_density(h) for h in hs]
    assert all(b < a for a, b in zip(rhos, rhos[1:]))
    assert all(r > 0.0 for r in rhos)


def test_out_of_range_clamped() -> None:
    assert air_density(-10000.0) == air_density(0.0)
    assert air_density(120000.0) == air_density(84852.0) > 0.0


# ── внедрение в трёхстепенную модель ───────────────────────────────────────


def test_atmos_off_is_altitude_independent() -> None:
    """Регресс (atmos=False = прежнее поведение): траектории на нуле и на 11 км
    с постоянной ρ совпадают по скорости — высота ни на что не влияет."""
    p_off = PhysicsParams(rho_air=0.736, ref_area_m2=0.05, drag_cx=0.3, thrust_n=0.0, g=0.0, tau_a_s=0.0)
    assert rho_at_h(p_off, 0.0) == 0.736 and rho_at_h(p_off, 20000.0) == 0.736

    def _coast(z0: float) -> FlightState:
        st = FlightState(p=np.array([0.0, 0.0, z0]), v=np.array([400.0, 0.0, 0.0]))
        t = 0.0
        for _ in range(800):
            st = step_physics(st, np.zeros(3), p_off, t, 0.005)
            t += 0.005
        return st

    a, b = _coast(0.0), _coast(11000.0)
    assert np.allclose(a.v, b.v, rtol=1e-12)
    assert math.isclose(a.speed, b.speed, rel_tol=1e-12)


def test_atmos_on_drag_scales_with_std_density() -> None:
    """Отношение торможения на нуле и на 11 км равно табличному отношению ρ."""
    p = PhysicsParams(rho_air=0.0, ref_area_m2=0.05, drag_cx=0.3, thrust_n=0.0, g=0.0, tau_a_s=0.0, atmos=True)
    v = np.array([400.0, 0.0, 0.0])
    a0 = float(np.linalg.norm(drag_accel(v, p, 0.0)))
    a11 = float(np.linalg.norm(drag_accel(v, p, 11000.0)))
    assert math.isclose(a0 / a11, air_density(0.0) / air_density(11000.0), rel_tol=1e-12)
    # отношение ρ(0)/ρ(11 км) = 3.366 (US1976)
    assert math.isclose(a0 / a11, 3.366, rel_tol=5e-3)


def test_atmos_coast_matches_analytic_quadratic_drag() -> None:
    """Аналитический эталон: квадратичное сопротивление при постоянной ρ даёт
    v(t) = v₀/(1 + k·v₀·t), k = 0.5·ρ·S·Cx/m. Интегрируем с малым шагом на
    НЕ constante ρ — но высота за 3 с падает на доли метра (g выключен,
    v_z = 0) — ρ фактически постоянна и решение обязано совпасть с формулой."""
    from navedenie.atmos import air_density as ad

    h0, v0, T, dt = 11000.0, 400.0, 3.0, 0.005
    rho = ad(h0)
    p = PhysicsParams(mass_kg=150.0, ref_area_m2=0.05, drag_cx=0.3, rho_air=0.0,
                      thrust_n=0.0, g=0.0, tau_a_s=0.0, atmos=True)
    st = FlightState(p=np.array([0.0, 0.0, h0]), v=np.array([v0, 0.0, 0.0]))
    t = 0.0
    while t < T - 1e-12:
        st = step_physics(st, np.zeros(3), p, t, dt)
        t += dt
    k = 0.5 * rho * 0.05 * 0.3 / 150.0
    v_exact = v0 / (1.0 + k * v0 * T)
    assert math.isclose(st.speed, v_exact, rel_tol=2e-3)
    assert st.speed > v0 / (1.0 + (0.5 * ad(0.0) * 0.05 * 0.3 / 150.0) * v0 * T)  # на 11 км торможение слабее


# ── сквозная проверка через Scenario / движок ──────────────────────────────


def _phys_scenario(**kw) -> Scenario:
    base = dict(
        mode="bio", brain="stub", model="point_mass_3dof",
        maneuver="straight", dt=0.02, t_max=6.0,
        phys_gravity=False, phys_atmos=True, phys_rho_air=0.0, phys_tau_a_s=0.0,
    )
    base.update(kw)
    return Scenario(**base)


def test_run_frames_carry_atm_rho_and_descent_increases_it() -> None:
    """Кадры физ-режима несут atm_rho; цель на высоте выше старта — ракета
    набирает/теряет высоту, но главное: atm_rho меняется с высотой и равна
    табличному значению на высоте последнего кадра."""
    sc = _phys_scenario(alt_m=11000.0)
    res = collect(sc, stride=8)
    assert res.frames, "прогон не дал кадров"
    f0, f1 = res.frames[0], res.frames[-1]
    assert f0.atm_rho is not None
    assert math.isclose(f0.atm_rho, air_density(11000.0), rel_tol=1e-9)
    # atm_rho остаётся плотностью стандартной атмосферы на фактической высоте
    assert math.isclose(f1.atm_rho, air_density(float(f1.missile[2])), rel_tol=1e-6)


def test_kinematic_run_has_no_atm_rho() -> None:
    sc = Scenario(mode="bio", brain="stub", model="kinematic_legacy", dt=0.02, t_max=3.0)
    res = collect(sc, stride=8)
    assert all(fr.atm_rho is None for fr in res.frames)


def test_high_altitude_flight_is_faster_than_sea_level() -> None:
    """Физическое следствие высоты: при atmos=True баллистический участок на
    11 км теряет скорость заметно меньше, чем у земли (ρ в 3.4 раза ниже)."""
    kw = dict(alt_m=0.0)
    lo = collect(_phys_scenario(phys_gravity=False, **kw), stride=32)
    hi = collect(_phys_scenario(phys_gravity=False, alt_m=11000.0), stride=32)
    assert hi.frames[-1].speed_ms > lo.frames[-1].speed_ms * 1.02


def test_scenario_in_accepts_phys_atmos_and_alt() -> None:
    """API-модель принимает новые поля и прокидывает их в Scenario."""
    from navedenie.app import ScenarioIn, _sc

    sc = _sc(ScenarioIn(model="point_mass_3dof", phys_atmos=True, alt_m=9000.0))
    assert sc.phys_atmos is True and sc.alt_m == 9000.0
    # дефолт сохраняет прежнее поведение
    d = _sc(ScenarioIn())
    assert d.phys_atmos is False and d.alt_m == 4000.0
