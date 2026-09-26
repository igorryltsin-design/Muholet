"""η — «угол встречи»: терминальная метрика качества наведения.

Определение (закреплено здесь и в docs/notation.md): угол между векторами
скоростей ракеты и цели в момент наибольшего сближения (аргмин накопленного
CPA; при перехвате — кадр срабатывания БЧ), 0…180°. 180° — лобовой курс,
0° — догон. Классическая терминальная метрика PNG-литературы (Зархан,
Tactical Missiles Guidance; Kim–Lee–Grider 1993, impact angle control).
"""

from __future__ import annotations

import math

import numpy as np

from navedenie.app import ScenarioIn, run_once
from navedenie.engine import collect
from navedenie.sim import G, Scenario

BASE = dict(
    mode="pn", law="pn", v_m=780.0, v_t=260.0, n_max=30.0, pn_n=4.0,
    dt=0.02, t_max=30.0, kill_radius_m=45.0, range_m=8000.0,
)


def _run(**kw) -> Scenario:
    s = dict(BASE)
    s.update(kw)
    return Scenario(**s)


def test_head_on_and_tail_chase_anchor_on_course():
    """Аналитическая привязка к геометрии курса: прямо летящая цель в лолу
    даёт η близкий к 180° (скорости антипараллельны), в догоне — близкий к 0°
    (параллельны). Отклонения — только от начального смещения и траверса."""
    eta_head = collect(_run(aspect="head-on", off_axis_m=0.0), stride=100_000).impact_angle_deg
    eta_tail = collect(_run(aspect="tail-chase", off_axis_m=0.0), stride=100_000).impact_angle_deg
    assert eta_head is not None and eta_tail is not None
    assert 170.0 < eta_head <= 180.0, eta_head
    assert 0.0 <= eta_tail < 5.0, eta_tail


def test_turning_target_eta_matches_turn_rate_times_time():
    """Догон + цель с ПОСТОЯННОЙ перегрузкой (вираж): к моменту встречи скорость
    цели повернулась на угол ω_ц·t = (n·g/v_t)·t_hit, ракета к CPA ещё почти
    на прежнем курсе — η должен совпасть с учебной формулой поворота вектора
    скорости в пределах угла, добавляемого траекторией ракеты (±20°)."""
    n_t = 6.0
    res = collect(_run(aspect="tail-chase", maneuver="turn", n_target=n_t), stride=100_000)
    assert res.hit and res.t_hit is not None
    eta_expected = math.degrees(n_t * G / 260.0 * res.t_hit)  # ω_ц = n·g/v_t, rad/s
    assert abs(res.impact_angle_deg - eta_expected) <= 20.0, (res.impact_angle_deg, eta_expected)
    assert res.impact_angle_deg > 90.0  # цель успела развернуться более чем на четверть оборота


def test_impact_angle_dt_stable():
    """η считается по непрерывному CPA — метрика устойчива к шагу интегрирования:
    dt 0.02 и 0.01 дают один и тот же угол с точностью до 1°."""
    e20 = collect(_run(aspect="head-on", off_axis_m=0.0), stride=100_000).impact_angle_deg
    e10 = collect(_run(aspect="head-on", off_axis_m=0.0, dt=0.01), stride=100_000).impact_angle_deg
    assert abs(e20 - e10) <= 1.0, (e20, e10)


def test_consistent_with_frames_argmin():
    """Поверка проводки: η, пересчитанный по ПОЛНОЙ ленте кадров (аргмин
    накопленного кадра fr.miss и скорости того же кадра), совпадает с полем
    RunResult — поле действительно берётся из момента наибольшего сближения."""
    res = collect(_run(aspect="beam"), stride=1)
    best = math.inf
    vm = vt = None
    for fr in res.frames:
        if fr.miss < best - 1e-12:
            best = fr.miss
            vm, vt = fr.missile_v, fr.target_v
    assert vm is not None
    cos = float(np.dot(vm, vt)) / (float(np.linalg.norm(vm)) * float(np.linalg.norm(vt)))
    eta = round(math.degrees(math.acos(max(-1.0, min(1.0, cos)))), 2)
    assert abs(eta - res.impact_angle_deg) <= 0.02


def test_beam_geometry_ordering():
    """Бортовая геометрия: η строго между догонным и лобовым якорями — метрика
    различает аспекты встречи, не вырождаясь."""
    eta = collect(_run(aspect="beam"), stride=100_000).impact_angle_deg
    assert 5.0 < eta < 170.0, eta


def test_api_echoes_impact_angle():
    """Проводка API: /api/run отдаёт поле impact_angle_deg с тем же значением,
    что считает collect на том же сценарии."""
    body = ScenarioIn(**{**BASE, "aspect": "tail-chase", "off_axis_m": 0.0})
    out = run_once(body)
    assert out["impact_angle_deg"] is not None
    ref = collect(Scenario(**{**BASE, "aspect": "tail-chase", "off_axis_m": 0.0}), stride=100_000)
    assert abs(out["impact_angle_deg"] - ref.impact_angle_deg) <= 0.01
