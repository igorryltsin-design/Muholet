"""Граница динамического полёта: располагаемая перегрузка из скоростного напора.

Проверяемое (Зархан, Tactical Missiles Guidance: available g_T = q·S·C_N/(W);
классика динамики полёта — Мануйленко/Удин):
  - аналитическое значение n_расп(V, H) = ½·ρ·V²·S·Cn_max/(m·g);
  - конструкционный предел: n_расп = min(n_avail_max, n_аэро);
  - пропорциональность ρ(H)V² (на динамическом напоре, не на числе Маха);
  - регресс: cn_max = 0 воспроизводит константный предел побитно;
  - физика: на малой скорости команде неоткуда взяться (a_act ≤ n_расп·g),
    промах на медленной ракете растёт.
"""

from dataclasses import replace

import numpy as np

from navedenie.atmos import air_density
from navedenie.engine import collect
from navedenie.physics import (FlightState, PhysicsParams, n_available,
                               step_physics)
from navedenie.sim import G, Scenario


def _p(**kw) -> PhysicsParams:
    base = dict(mass_kg=150.0, ref_area_m2=0.05, rho_air=0.736,
                n_avail_max=30.0, cn_max=2.0, tau_a_s=0.0, thrust_n=0.0, g=0.0)
    base.update(kw)
    return PhysicsParams(**base)


# ── аналитика формулы ────────────────────────────────────────────────────────

def test_available_g_analytic_value():
    # n = ½·ρ·V²·S·Cn/(m·g) = 0.5·0.736·300²·0.05·2/(150·9.81) = 2.25084… g
    n_ref = 0.5 * 0.736 * 300.0 ** 2 * 0.05 * 2.0 / (150.0 * G)
    assert abs(n_ref - 2.25084) < 1e-4
    assert abs(n_available(300.0, 0.0, _p()) - n_ref) < 1e-12


def test_structural_limit_caps():
    # V = 3000 м/с → аэро = 225 g, но конструкция держит 30 g
    assert n_available(3000.0, 0.0, _p()) == 30.0
    # ровно на стыке: V² = 2·m·g·n_max/(ρ·S·Cn) → предел = n_max
    v_cap = np.sqrt(2 * 150.0 * G * 30.0 / (0.736 * 0.05 * 2.0))
    assert n_available(v_cap * 0.999, 0.0, _p()) < 30.0
    assert n_available(v_cap * 1.001, 0.0, _p()) == 30.0


def test_available_g_scales_with_dynamic_pressure():
    p = _p()
    # V²: четырёхкратная скорость — четырёхкратная располагаемая (вне насыщения)
    assert abs(n_available(150.0, 0.0, p) * 4.0 - n_available(300.0, 0.0, p)) < 1e-12
    # ρ(H): на 11 км плотность ниже во столько же раз (atmos включён)
    pa = _p(atmos=True)
    ratio = air_density(11000.0) / air_density(0.0)
    assert abs(n_available(300.0, 11000.0, pa) - n_available(300.0, 0.0, pa) * ratio) < 1e-12


def test_cn_zero_keeps_constant_limit():
    p0 = _p(cn_max=0.0)
    for v in (1.0, 100.0, 300.0, 3000.0):
        assert n_available(v, 5000.0, p0) == 30.0


# ── исполнение в шаге модели ─────────────────────────────────────────────────

def test_step_respects_available_g():
    """Медленная ракета (100 м/с) не может выполнить команду 10 g — реализуется
    не больше n_available(100 м/с)·g ≈ 0.25 g."""
    p = _p(drag_cx=0.0)  # чистый механизм ограничения, без торможения
    n_lim = n_available(100.0, 0.0, p)
    assert 0.0 < n_lim < 1.0  # ~0.25 g — «повернуть ракету нечем»
    st = FlightState(p=np.array([0.0, 0.0, 0.0]), v=np.array([100.0, 0.0, 0.0]))
    a_cmd = np.array([0.0, 10.0 * 9.81, 0.0])  # команда 10 g по оси Y
    for _ in range(20):
        st = step_physics(st, a_cmd, p, 0.0, 0.01)
        assert float(np.linalg.norm(st.a_act)) <= n_lim * G * 1.0000001 + 1e-12
    speed = float(np.linalg.norm(st.v))
    assert abs(speed - 100.0) < 1e-6  # модуль скорости сохранён (без тяги/сопр.)


def test_cn_zero_step_is_bitwise_old():
    """Регресс: cn_max = 0 даёт побитово прежний полёт с константным пределом."""
    sc = Scenario(model="point_mass_3dof", aspect="beam", alt_m=7000.0,
                  phys_gravity=False, phys_thrust_n=0.0, phys_tau_a_s=0.0)
    a = collect(sc)
    b = collect(replace(sc, phys_cn_max=0.0, phys_n_avail_max=30.0))
    assert a.cpa_m == b.cpa_m
    assert [f.missile.tolist() for f in a.frames] == [f.missile.tolist() for f in b.frames]


def test_low_speed_cannot_follow_command():
    """Медленная ракета (180 м/с): с границей динамического полёта реализуемая
    перегрузка не превышает аэродинамически доступную ~0.8 g, тогда как прежде
    она исполняла команду закона до ~7.8 g. Знак промаха при этом геометрийный
    — проверяем именно неисполнение команды."""
    sc = Scenario(model="point_mass_3dof", aspect="beam", alt_m=7000.0, v_m=180.0,
                  phys_gravity=False, phys_thrust_n=0.0, phys_tau_a_s=0.0,
                  phys_ref_area_m2=0.05, phys_mass_kg=150.0, phys_rho_air=0.736)
    const = collect(sc)                      # cn_max = 0 — прежние 30 g всегда
    aero = collect(replace(sc, phys_cn_max=2.0))
    n_aero_v0 = n_available(180.0, 7000.0, _p())
    assert 0.5 < n_aero_v0 < 1.5
    act_const = max(f.n_act for f in const.frames)
    act_aero = max(f.n_act for f in aero.frames)
    assert act_const > 3.0                    # прежде команда исполнялась почти целиком
    assert act_aero <= n_aero_v0 * 1.05       # теперь потолок — динамика полёта
    assert act_aero < act_const / 3.0


def test_n_avail_telemetry_becomes_dynamic():
    sc = Scenario(model="point_mass_3dof", aspect="head-on", alt_m=7000.0,
                  phys_gravity=False, phys_thrust_n=0.0, phys_tau_a_s=0.0,
                  phys_cn_max=2.0)
    res = collect(sc)
    avs = [f.n_avail for f in res.frames]
    assert all(a is not None for a in avs)
    assert max(avs) <= 30.0 + 1e-12
    # где-то граница реальна (меньше константы): телеметрия не плоская
    assert min(avs) < 29.0
    # n_avail считается ПРЕД шагом (этим и ограничивалась команда шага), а
    # speed_ms — постшаг; сверяем с границей на post-step скорости с допуском
    # на приращение скорости за шаг (dt·a ≤ 0.3 g)
    for f in res.frames[:20]:
        n_cap_here = n_available(f.speed_ms, float(f.missile[2]), _p())
        assert abs(f.n_avail - n_cap_here) < 0.5
