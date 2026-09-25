"""Волновой кризис лобового сопротивления Cx(M) в трёхстепенной модели.

Проверяемое (Бертин, Каммингс, «Аэродинамика. Основы»: рост волнового
сопротивления в околозвуковой области; аппроксимация — S-образный tanh-переход):
  - табличные опоры скорости звука стандартной атмосферы a(H) = √(γRT(H));
  - плато и перегиб Cx(M): подзвук → Cx₀, M = M_кр → Cx₀ + ΔCx/2, сверхзвук → Cx₀ + ΔCx;
  - монотонный рост с максимумом крутизны в M_кр;
  - регресс: cx_wave = 0 воспроизвляет постоянный Cx побитно;
  - аналитика: торможение при насыщенном (сверхзвуковом) Cx совпадает с
    v(t) = v₀/(1 + k·v₀·t); при переходе через кризис торможение сильнее.
"""

from dataclasses import replace

import numpy as np
import pytest

from navedenie.atmos import speed_of_sound
from navedenie.engine import collect, frame_to_dict, run
from navedenie.physics import PhysicsParams, drag_cx_at, mach_at, step_physics, FlightState
from navedenie.sim import Scenario


# ── скорость звука: табличные опоры US1976 / ISA ────────────────────────────

def test_speed_of_sound_table():
    assert abs(speed_of_sound(0.0) - 340.294) < 1e-3
    assert abs(speed_of_sound(5000.0) - 320.53) < 0.05
    # тропопауза 11 км: T = 216.65 K → a = 295.0665 м/с (печатная таблица ISA)
    assert abs(speed_of_sound(11000.0) - 295.0665) < 0.01
    # изотермический слой 11…20 км — a не меняется
    assert speed_of_sound(20000.0) == speed_of_sound(11000.0)
    # за границами модели — заморозка на границе
    assert speed_of_sound(-1000.0) == speed_of_sound(0.0)


def test_mach_number_definition():
    h = 7000.0
    a = speed_of_sound(h)
    assert abs(mach_at(a, h) - 1.0) < 1e-12
    assert abs(mach_at(2.5 * a, h) - 2.5) < 1e-12


# ── форма кривой Cx(M) ───────────────────────────────────────────────────────

def _wave_params() -> PhysicsParams:
    return PhysicsParams(drag_cx=0.30, cx_wave=0.35, mach_kr=1.0, mach_band=0.10)


def test_cx_plateaus_and_inflection():
    p = _wave_params()
    a0 = speed_of_sound(0.0)
    # глубокое подзвуковое плато: tanh((0.2−1)/0.1) ≈ −1 + 3·10⁻⁸
    cx_sub = drag_cx_at(0.2 * a0, 0.0, p)
    assert abs(cx_sub - 0.30) < 1e-6
    # ровно на M_кр — половина прироста (tanh 0 = 0)
    assert abs(drag_cx_at(a0, 0.0, p) - (0.30 + 0.175)) < 1e-12
    # сверхзвуковое плато
    cx_sup = drag_cx_at(1.8 * a0, 0.0, p)
    assert abs(cx_sup - 0.65) < 1e-6
    # монотонный рост и максимум крутизны в M_кр
    ms = np.linspace(0.2, 1.8, 321)
    cx = np.array([drag_cx_at(m * a0, 0.0, p) for m in ms])
    assert np.all(np.diff(cx) >= 0.0)
    grad = np.gradient(cx, ms)
    assert ms[int(np.argmax(grad))] == pytest.approx(1.0, abs=0.02)


def test_cx_wave_zero_is_constant():
    p = PhysicsParams(drag_cx=0.42, cx_wave=0.0)
    a0 = speed_of_sound(0.0)
    for m in (0.05, 0.9, 1.0, 1.5, 4.0):
        assert drag_cx_at(m * a0, 0.0, p) == 0.42


# ── интеграция в движок ──────────────────────────────────────────────────────

def _phys_scenario(**kw) -> Scenario:
    base = dict(model="point_mass_3dof", aspect="head-on", alt_m=7000.0,
                phys_gravity=False, phys_thrust_n=0.0, phys_tau_a_s=0.0)
    base.update(kw)
    return Scenario(**base)


def test_wave_zero_reproduces_constant_cx_bitwise():
    a = collect(_phys_scenario())
    b = collect(_phys_scenario(phys_cx_wave=0.0, phys_mach_kr=0.9, phys_mach_band=0.2))
    assert a.cpa_m == b.cpa_m
    assert [f.speed_ms for f in a.frames] == [f.speed_ms for f in b.frames]


def test_supersonic_decay_matches_analytic_constant_cx():
    """Сверхзвук M ≫ M_кр: tanh насыщен, Cx = Cx₀+ΔCx постоянен →
    квадратичное торможение v(t) = v₀/(1 + k·v₀·t), k = ρS·Cx/(2m)."""
    p = PhysicsParams(mass_kg=150.0, ref_area_m2=0.05, drag_cx=0.30, cx_wave=0.35,
                      rho_air=0.736, thrust_n=0.0, tau_a_s=0.0, g=0.0,
                      mach_kr=1.0, mach_band=0.10)
    v0 = np.array([2600.0, 0.0, 0.0])
    st = FlightState(p=np.array([0.0, 0.0, 7000.0]), v=v0.copy())
    dt, t_end = 0.002, 4.0
    k = 0.5 * p.rho_air * p.ref_area_m2 * (p.drag_cx + p.cx_wave) / p.mass_kg
    n_steps = int(t_end / dt)
    zero = np.zeros(3)
    for _ in range(n_steps):
        st = step_physics(st, zero, p, 0.0, dt)
        assert st.v[1] == 0.0 and st.v[2] == 0.0  # прямолинейное торможение
    v_num = float(np.linalg.norm(st.v))
    v_an = 2600.0 / (1.0 + k * 2600.0 * t_end)
    assert abs(v_num - v_an) / v_an < 2e-3
    # весь участок остался глубоко сверхзвуковым (tanh насыщен: M > 4 = M_кр+30δ)
    assert mach_at(v_num, 7000.0) > 4.0


def test_transonic_wave_drag_brakes_harder():
    """За одно и то же время околозвукового торможения волновой рост Cx забирает
    скорость сильнее постоянного подзвукового Cx₀."""
    p0 = PhysicsParams(rho_air=0.736, thrust_n=0.0, tau_a_s=0.0, g=0.0,
                       drag_cx=0.30, cx_wave=0.0)
    pw = replace(p0, cx_wave=0.35)
    def end_speed(p, t_end=6.0, dt=0.002):
        st = FlightState(p=np.array([0.0, 0.0, 7000.0]),
                         v=np.array([1200.0, 0.0, 0.0]))  # M ≈ 4.1 на 7 км
        zero = np.zeros(3)
        for _ in range(int(t_end / dt)):
            st = step_physics(st, zero, p, 0.0, dt)
        return float(np.linalg.norm(st.v))
    v_const = end_speed(p0)
    v_wave = end_speed(pw)
    assert v_wave < v_const  # волновой кризис честнее тормозит у M ≈ 1


def test_mach_telemetry_in_frames():
    res = collect(_phys_scenario(phys_cx_wave=0.35))
    fr0 = frame_to_dict(res.frames[0])
    assert fr0["mach"] is not None
    f0 = res.frames[0]
    # speed_ms в кадре хранится округлённым — сверяем с тем же допуском, что и графикостроение
    assert abs(fr0["mach"] - mach_at(f0.speed_ms, float(f0.missile[2]))) < 1e-6
    # телеметрия M живёт в физ-режиме даже при постоянном Cx...
    res_const = collect(_phys_scenario())
    assert frame_to_dict(res_const.frames[0])["mach"] is not None
    # ...и отсутствует в кинематическом режиме
    kin = collect(Scenario(aspect="head-on"))
    assert frame_to_dict(kin.frames[0])["mach"] is None


def test_wave_drag_changes_cpa_honestly():
    """Волновой кризис меняет результат прогона (не нулевой эффект), но остаётся
    в разумных пределах демонстрационной баллистики."""
    a = collect(_phys_scenario())
    b = collect(_phys_scenario(phys_cx_wave=0.35))
    assert a.cpa_m != b.cpa_m
    assert 0.0 < b.cpa_m < 1e4
