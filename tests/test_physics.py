"""Тесты трёхстепенной физической модели point_mass_3dof (navedenie/physics.py).

Проверяются физические инварианты и аналитические случаи, а НЕ конкретная
реализация: свободное падение, тяга, сопротивление, сохранение модуля скорости
при чисто нормальном ускорении, отклик исполнительного контура, ограничения и
сходимость при уменьшении dt."""

from __future__ import annotations

import numpy as np

from navedenie.physics import (
    FlightState,
    PhysicsParams,
    drag_accel,
    gravity_accel,
    n_available,
    step_physics,
    thrust_accel,
)
from navedenie.sim import G


def _zero_forces(**kw) -> PhysicsParams:
    """Параметры без тяги/сопротивления/тяги — только заданное нормальное ускорение."""
    base = dict(thrust_n=0.0, rho_air=0.0, g=0.0, tau_a_s=0.0, n_avail_max=1000.0)
    base.update(kw)
    return PhysicsParams(**base)  # type: ignore[arg-type]


def test_straight_line_no_forces() -> None:
    """Без сил и без команды — прямолинейное равномерное движение."""
    st = FlightState(p=np.array([0.0, 0.0, 0.0]), v=np.array([300.0, 0.0, 0.0]))
    params = _zero_forces()
    dt = 0.01
    for _ in range(100):
        st = step_physics(st, np.zeros(3), params, 0.0, dt)
    assert np.allclose(st.v, [300.0, 0.0, 0.0], atol=1e-9)
    assert abs(st.p[0] - 300.0) < 1e-6  # 300 м/с · 1 с
    assert abs(st.p[1]) < 1e-9 and abs(st.p[2]) < 1e-9


def test_free_fall_parabola() -> None:
    """Только тяжесть: горизонтальная скорость стоит, вертикаль — g·t, путь — парабола."""
    st = FlightState(p=np.array([0.0, 0.0, 1000.0]), v=np.array([200.0, 0.0, 0.0]))
    params = _zero_forces(g=G)
    dt = 0.005
    t = 0.0
    for _ in range(400):  # 2 с
        st = step_physics(st, np.zeros(3), params, t, dt)
        t += dt
    assert np.isclose(st.v[0], 200.0, atol=1e-6)
    assert np.isclose(st.v[2], -G * t, rtol=2e-3)
    assert np.isclose(1000.0 - st.p[2], 0.5 * G * t * t, rtol=5e-3)


def test_constant_thrust_increases_speed() -> None:
    """Постоянная тяга вдоль скорости: модуль растёт как (P/m)·t."""
    st = FlightState(p=np.zeros(3), v=np.array([200.0, 0.0, 0.0]))
    params = _zero_forces(thrust_n=3000.0, burn_time_s=10.0, mass_kg=150.0)
    dt = 0.005
    t = 0.0
    for _ in range(200):  # 1 с
        st = step_physics(st, np.zeros(3), params, t, dt)
        t += dt
    a = 3000.0 / 150.0
    assert np.isclose(st.speed, 200.0 + a * t, rtol=2e-3)
    # направление не изменилось
    assert abs(st.v[1]) < 1e-9 and abs(st.v[2]) < 1e-9


def test_drag_brakes_and_dissipates_energy() -> None:
    """Сопротивление тормозит и убивает кинетическую энергию; тяги нет."""
    st = FlightState(p=np.zeros(3), v=np.array([400.0, 0.0, 0.0]))
    params = _zero_forces(rho_air=0.736, ref_area_m2=0.05, drag_cx=0.3, mass_kg=150.0)
    dt = 0.005
    t = 0.0
    ke0 = 0.5 * params.mass_kg * st.speed**2
    for _ in range(400):
        st = step_physics(st, np.zeros(3), params, t, dt)
        t += dt
    ke1 = 0.5 * params.mass_kg * st.speed**2
    assert st.speed < 400.0
    assert ke1 < ke0
    # замедление монотонно убывает по мере падения скорости
    a_hi = float(np.linalg.norm(drag_accel(np.array([400.0, 0.0, 0.0]), params)))
    a_lo = float(np.linalg.norm(drag_accel(np.array([200.0, 0.0, 0.0]), params)))
    assert a_hi > a_lo > 0.0


def test_pure_normal_accel_preserves_speed() -> None:
    """Чисто нормальное ускорение без продольных сил сохраняет модуль скорости точно."""
    st = FlightState(p=np.zeros(3), v=np.array([300.0, 0.0, 0.0]))
    params = _zero_forces()
    a_cmd = np.array([0.0, 0.0, 10.0 * G])  # вертикальная команда, ⊥ скорости
    dt = 0.01
    for _ in range(300):
        st = step_physics(st, a_cmd, params, 0.0, dt)
    assert np.isclose(st.speed, 300.0, rtol=1e-9)  # модуль не изменился


def test_actuator_first_order_lag_time_constant() -> None:
    """Ступенчатый отклик: a_act догоняет команду по экспоненте с постоянной tau_a."""
    st = FlightState(p=np.zeros(3), v=np.array([300.0, 0.0, 0.0]))
    tau = 0.1
    params = _zero_forces(tau_a_s=tau, n_avail_max=1000.0)
    a_cmd = np.array([0.0, 0.0, 5.0 * G])
    dt = 0.001
    # через время tau приращения должно быть ~63%
    target = 5.0 * G
    reached = 0.0
    t = 0.0
    for _ in range(int(tau / dt)):
        st = step_physics(st, a_cmd, params, t, dt)
        t += dt
        reached = float(np.linalg.norm(st.a_act))
    assert reached == target * (1.0 - np.exp(-1.0)) or abs(reached - target * 0.632) < target * 0.03


def test_command_rate_limit() -> None:
    """Ограничение скорости изменения команды не даёт a_act прыгнуть мгновенно."""
    st = FlightState(p=np.zeros(3), v=np.array([300.0, 0.0, 0.0]))
    params = _zero_forces(tau_a_s=0.0, da_dt_max=100.0, n_avail_max=1000.0)
    a_cmd = np.array([0.0, 0.0, 30.0 * G])  # большой шаг команды
    st1 = step_physics(st, a_cmd, params, 0.0, dt=0.01)
    # приращение за шаг ограничено da_dt_max·dt = 1 м/с²
    assert float(np.linalg.norm(st1.a_act)) <= 100.0 * 0.01 + 1e-9


def test_available_overload_limit_and_n_available() -> None:
    """Команда ограничивается располагаемой перегрузкой; n_available — константный предел."""
    params = PhysicsParams(n_avail_max=20.0)
    assert n_available(400.0, 5000.0, params) == 20.0
    st = FlightState(p=np.zeros(3), v=np.array([300.0, 0.0, 0.0]))
    a_cmd = np.array([0.0, 0.0, 100.0 * G])  # заведомо выше предела
    p = _zero_forces(tau_a_s=0.0, n_avail_max=20.0)
    st1 = step_physics(st, a_cmd, p, 0.0, dt=0.001)
    assert float(np.linalg.norm(st1.a_act)) <= 20.0 * G + 1e-6


def test_gravity_accel_direction() -> None:
    g = gravity_accel(PhysicsParams(g=G))
    assert np.allclose(g, [0.0, 0.0, -G])


def test_physics_trajectory_converges_with_dt() -> None:
    """Сходимость интегратора: фиксированная команда нормального ускорения,
    прогон до одного и того же времени при dt, dt/2, dt/4. Разница конечных
    точек монотонно убывает при уменьшении шага (сходимость порядка dt), а
    модуль скорости сохраняется точно на всех шагах.

    Тестируется именно численная модель, отвязанная от разрыва «перехват /
    промах» терминального события — на пограничном сценарии CPA сходимость
    не обязана быть монотонной, и это не баг интегратора."""
    V, a_cmd_mag, T = 300.0, 8.0 * G, 5.0
    params = _zero_forces(n_avail_max=1000.0)

    def _final(dt: float) -> np.ndarray:
        st = FlightState(p=np.zeros(3), v=np.array([V, 0.0, 0.0]))
        a_cmd = np.array([0.0, 0.0, a_cmd_mag])
        t = 0.0
        for _ in range(int(round(T / dt))):
            st = step_physics(st, a_cmd, params, t, dt)
            t += dt
            assert np.isclose(st.speed, V, rtol=1e-9)  # нормальное ускорение не меняет |V|
        return st.p.copy()

    p1 = _final(0.05)
    p2 = _final(0.025)
    p4 = _final(0.0125)
    d12 = float(np.linalg.norm(p1 - p2))
    d24 = float(np.linalg.norm(p2 - p4))
    assert d12 > 0.0                    # шаг влияет на решение
    assert d24 < d12                    # уменьшение шага уменьшает расхождение
    assert d24 < 0.75 * d12             # порядка dt: расхождение падает не медленнее
