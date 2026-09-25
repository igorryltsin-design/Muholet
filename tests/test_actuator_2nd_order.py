"""Рулевой привод 2-го порядка в трёхстепенной модели: звено
ä + 2ζω_n·ȧ + ω_n²·a = ω_n²·a_cmd (Зархан, Tactical Missiles Guidance;
[Гусев 1996] гл. 5) с точным ЗОХ-переходником на шаге dt.

Тесты сверяют реализацию с АНАЛИТИКОЙ теории управления: переходная функция
при ступенчатом воздействии для ζ<1, ζ=1, ζ>1; перерегулирование
M_p = exp(−πζ/√(1−ζ²)); установившееся усиление = 1; при wn_act = 0 поведение
сводится к прежнему 1-му звену побитно."""

import numpy as np

from navedenie.app import ScenarioIn, run_once
from navedenie.physics import FlightState, PhysicsParams, _second_order_step, step_physics

G0 = 9.80665


def _params(**kw):
    base = dict(mass_kg=150.0, ref_area_m2=0.0, drag_cx=0.0, rho_air=0.0,
                thrust_n=0.0, g=0.0, tau_a_s=0.0, n_avail_max=40.0)
    base.update(kw)
    return PhysicsParams(**base)  # type: ignore[arg-type]


def _step_response(u, params_kw, t_end=3.0, dt=0.001):
    """Характеристика привода внутри step_physics: v велика (поворот корпуса
    за окно теста пренебрежим), команде — ступенька u по оси Z.
    Возвращает массив a_act_z(t) и dt."""
    p = _params(**params_kw)
    st = FlightState(p=np.zeros(3), v=np.array([100000.0, 0.0, 0.0]))
    cmd = np.array([0.0, 0.0, u])
    out = []
    t = 0.0
    while t < t_end - 1e-12:
        st = step_physics(st, cmd, p, t, dt)
        out.append(st.a_act[2])
        t += dt
    return np.array(out), dt


def test_wn_zero_is_bitwise_first_order():
    """wn_act = 0 — прежнее 1-е звено tau_a_s: тот же промах и та же телеметрия."""
    body = ScenarioIn(model="point_mass_3dof", maneuver="turn", n_target=4.0,
                      t_max=8.0, dt=0.02, seed=3)
    a = run_once(body)
    b = run_once(body.model_copy(update={"phys_wn_act": 0.0, "phys_zeta_act": 1.0}))
    assert [f["n_act"] for f in a["frames"]] == [f["n_act"] for f in b["frames"]]
    assert a["cpa_m"] == b["cpa_m"]


def test_step_response_underdamped_matches_analytics():
    """ζ=0.5, ω_n=20: a_act(t)/u совпадает с 1 − e^(−ζω_n t)(cos ω_d t + ζω_n/ω_d·sin ω_d t)."""
    wn, zeta, u = 20.0, 0.5, 1.0
    y, dt = _step_response(u, {"wn_act": wn, "zeta_act": zeta}, t_end=0.5)
    a = zeta * wn
    wd = wn * np.sqrt(1 - zeta * zeta)
    t = np.arange(1, len(y) + 1) * dt  # точные моменты шагов (без накопления t+=dt)
    ref = 1 - np.exp(-a * t) * (np.cos(wd * t) + a / wd * np.sin(wd * t))
    assert np.max(np.abs(y - ref)) < 1e-3


def test_overshoot_equals_theory():
    """Перерегулирование ζ=0.5: пик = u·(1 + exp(−πζ/√(1−ζ²)))."""
    wn, zeta, u = 20.0, 0.5, 300.0
    y, _ = _step_response(u, {"wn_act": wn, "zeta_act": zeta}, t_end=1.5)
    theory = u * (1 + np.exp(-np.pi * zeta / np.sqrt(1 - zeta * zeta)))
    assert abs(y.max() - theory) < 0.01 * u


def test_critical_and_overdamped_monotone():
    """ζ=1 и ζ=2 — монотонный апериодический вход без перерегулирования."""
    for zeta in (1.0, 2.0):
        y, _ = _step_response(100.0, {"wn_act": 15.0, "zeta_act": zeta}, t_end=2.0)
        assert np.all(np.diff(y) > -1e-4), f"ζ={zeta}: есть убывание"
        assert y.max() <= 100.0 + 1e-4
        assert abs(y[-1] - 100.0) < 2.0  # установившееся усиление = 1 (ζ=2 — медленная мода)


def test_steady_state_gain_is_one():
    """После переходного процесса a_act → a_cmd (ω_n=20, ζ=0.7, окно 3 с)."""
    y, _ = _step_response(250.0, {"wn_act": 20.0, "zeta_act": 0.7}, t_end=3.0)
    assert abs(y[-1] - 250.0) < 250.0 * 1e-3


def test_second_order_step_helper_direct():
    """ЗОХ-переходник: три режима демпфирования за 500 шагов == аналитика до 1e-12."""
    import math
    wn, u, h = 20.0, np.array([1.0, 1.0, 1.0]), 0.001
    for zeta in (0.5, 1.0, 2.0):
        x = np.zeros(3)
        v = np.zeros(3)
        for _ in range(500):
            x, v = _second_order_step(x, v, u, wn, zeta, h)
        t = 500 * h
        a = zeta * wn
        if zeta < 1:
            wd = wn * math.sqrt(1 - zeta * zeta)
            ref = 1 - math.exp(-a * t) * (math.cos(wd * t) + a / wd * math.sin(wd * t))
        elif zeta == 1:
            ref = 1 - math.exp(-wn * t) * (1 + wn * t)
        else:
            b = wn * math.sqrt(zeta * zeta - 1)
            s1, s2 = -a + b, -a - b
            ref = 1 - ((0 - s2) / (s1 - s2) * math.exp(s1 * t) + (s1 / (s1 - s2)) * math.exp(s2 * t))
        assert abs(x[0] - ref) < 1e-12


def test_scenario_in_wn_zeta_wire_through_api():
    """phys_wn_act/phys_zeta_act доходят до привода: с ω_n=5 кадра медленнее,
    чем с ω_n=50, а при ω_n=50 n_act близок к n_cmd."""
    base = dict(model="point_mass_3dof", maneuver="turn", n_target=4.0,
                t_max=8.0, dt=0.02, seed=7, phys_wn_act=50.0, phys_zeta_act=0.9)
    fast = run_once(ScenarioIn(**base))
    slow = run_once(ScenarioIn(**{**base, "phys_wn_act": 5.0}))
    ideal = run_once(ScenarioIn(**{**base, "phys_wn_act": 0.0, "phys_tau_a_s": 0.0}))
    dev_f = np.mean([abs(f["n_act"] - f["n_cmd"]) for f in fast["frames"] if f["n_act"] is not None])
    dev_s = np.mean([abs(f["n_act"] - f["n_cmd"]) for f in slow["frames"] if f["n_act"] is not None])
    dev_i = np.mean([abs(f["n_act"] - f["n_cmd"]) for f in ideal["frames"] if f["n_act"] is not None])
    assert dev_i == 0.0  # без привода исполнение точное
    assert dev_f < dev_s  # больше полоса привода — меньше рассогласование
    assert dev_f < 0.25 * dev_s
