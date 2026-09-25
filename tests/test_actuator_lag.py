"""Тесты лага рулевого привода в кинематическом контуре (Scenario.tau_act_s).

Модель — первое апериодическое звено на исполнении команды закона наведения:
τ·d(a_act)/dt + a_act = a_cmd (неявная устойчивая дискретизация, как в
navedenie/physics.py). Порядок «закон → ограничитель n_max → привод → корпус»
согласуется с [Гусев] гл. 5 (лаг привода — классическая причина роста промаха)
и Zarchan, Tactical Missiles Guidance (effect of first-order lag).

Проверяются честные инварианты:
  - tau_act_s = 0 — прежнее поведение (телескопия n_act/n_cmd выключена);
  - ступенчатый отклик: доля реализованной команды за τ ≈ 63 %;
  - постоянный вираж цели 6 g: промах строго растёт с лагом (классика);
  - модуль скорости сохраняется (привод не совершает работу);
  - API-модель принимает поле и прокидывает его в кадры.
"""

from __future__ import annotations

import math

import numpy as np

from navedenie.engine import collect
from navedenie.sim import Scenario, G


def _sc(tau: float, **kw) -> Scenario:
    base = dict(
        mode="pn", law="pn", aspect="beam", maneuver="turn", n_target=6,
        dt=0.02, t_max=16, tau_act_s=tau, seed=1,
    )
    base.update(kw)
    return Scenario(**base)


def test_zero_lag_is_legacy() -> None:
    """При tau_act_s = 0 кадровые поля привода не заполняются — трасса и метрики
    остаются прежнего поведения (фильтр не вносится вовсе)."""
    res = collect(_sc(0.0), stride=32)
    assert res.frames
    assert all(fr.n_act is None and fr.n_cmd is None for fr in res.frames)


def test_step_response_time_constant() -> None:
    """Ступенчатая команда (дальняя фава beam-сближения почти постоянна):
    за время τ реализованная перегрузка догоняет заданную на 63.2 % ± допуск
    неявной дискретизации (dt = τ/20 даёт ~64 %)."""
    tau = 0.1
    res = collect(_sc(tau, n_target=0, maneuver="straight"), stride=1)
    n_at_tau = int(round(tau / 0.02))
    fr = res.frames[n_at_tau]
    assert fr.n_cmd is not None and fr.n_act is not None
    assert fr.n_cmd > 1.0, "команда на упреждении слишком мала для ступенчатого теста"
    ratio = fr.n_act / fr.n_cmd
    assert 0.55 < ratio < 0.72, ratio


def test_miss_grows_monotonically_with_lag_on_const_turn() -> None:
    """Постоянный вираж цели 6 g: лаг привода вносит догоняющую ошибку
    сопровождения — промах строго растёт с τ (на случайных манёврах монотонность
    не обязана держаться, здесь геометрия стационарна)."""
    miss = []
    for tau in (0.0, 0.1, 0.2):
        r = collect(_sc(tau), stride=64)
        miss.append(r.cpa_m)
    assert miss[0] < miss[1] < miss[2], miss
    # честность: рост мал, перехваты не превращаются в фиктивные промахи
    assert miss[2] - miss[0] < 10.0


def test_speed_preserved_with_lag() -> None:
    """Привод не совершает работу: |V| ракеты постоянен и при лаге
    (кинематический интегратор поворачивает скорость)."""
    res = collect(_sc(0.15, v_m=780.0), stride=8)
    speeds = [float(np.linalg.norm(fr.missile_v)) for fr in res.frames]
    for s in speeds:
        assert abs(s - 780.0) < 1e-6


def test_executed_never_exceeds_available() -> None:
    """Выпуклая комбинация ограниченных команд не превышает предел n_max:
    реализованная перегрузка всегда в пределах допускаемой."""
    res = collect(_sc(0.2, n_max=12.0), stride=8)
    for fr in res.frames:
        if fr.n_act is not None:
            assert fr.n_act <= 12.0 + 1e-9


def test_scenario_in_and_frames_wire_tau_act() -> None:
    """API-модель: дефолт 0 сохраняет прежнее поведение; ненулевой лаг
    пробрасывается в Scenario и появляется в телеметрии кадров."""
    from navedenie.app import ScenarioIn, _sc as sc_from_in, run_once

    assert sc_from_in(ScenarioIn()).tau_act_s == 0.0
    body = ScenarioIn(mode="pn", aspect="beam", maneuver="turn", n_target=6,
                      dt=0.02, t_max=6, tau_act_s=0.1)
    out = run_once(body)
    with_lag = [f for f in out["frames"] if f.get("n_act") is not None]
    assert with_lag, "кадры с лагом привода должны нести n_act"
    kin = run_once(ScenarioIn(mode="pn", aspect="beam", maneuver="turn", n_target=6,
                              dt=0.02, t_max=6))
    assert all(f.get("n_act") is None for f in kin["frames"])
