"""Реактивные законы уклонения цели — дуэль «муха-ракета против мухи-самолёта».

Цель в режиме дуэли (Scenario.duel=True) больше не летит «по нотам»: она
«видит» ракету — положение и скорость (аналог приёмника облучения), — но не
знает её ускорения и не видит её будущего. Каждый закон возвращает нормальное
ускорение цели: ортогональное вектору скорости цели и ограниченное n_target·g —
та же конвенция, что у законов наведения ракеты (см. pn.py).

Обозначения (от лица уклониста): s = p_ракеты − p_цели (вектор «на ракету»),
v = v_ракеты − v_цели (скорость изменения s), R = |s|,closing = −(ŝ·v) —
скорость сближения ракеты с целью (>0, пока ракета налетает).

Публичные русские названия — glossary.EVADER_LABEL; паспорт законов —
docs/law_passports.md (§ уклонисты).
"""

from __future__ import annotations

import numpy as np

from navedenie.sim import G, Body, Scenario, Vec, _perp, _unit, clip_accel, los_omega

EVADERS = ("away", "negpn", "cpa_max", "brain")

_UP = np.array([0.0, 0.0, 1.0])


def _closing_speed(s: Vec, v_rel: Vec) -> float:
    """Скорость сближения ракеты с целью (s — «от цели к ракете», v_rel — её
    производная): положительна, пока ракета налетает."""
    u = _unit(s)
    return float(-np.dot(u, v_rel))


def _side_direction(body: Body, want: Vec, mag: float) -> Vec:
    """Единичное направление «полного банка» вбок от скорости цели: проекция
    желаемого направления на плоскость ⊥ v_ц; при вырождении — детерминированный
    «нырок в бок» (перпендикуляр к v через вертикаль, как в target_accel).
    Возвращает вектор длиной mag (уклонист оттягивает доступную перегрузку целиком)."""
    d = _perp(want, body.v)
    dn = float(np.linalg.norm(d))
    if dn < 1e-6:
        lift = _unit(np.cross(body.v, _UP))
        if float(np.linalg.norm(lift)) < 0.2:
            lift = _unit(np.cross(body.v, np.array([0.0, 1.0, 0.0])))
        return lift * mag
    return d * (mag / dn)


def away_accel(target: Body, missile: Body, mag: float) -> Vec:
    """«Уклон от точки встречи»: прямая экстраполяция даёт прогнозную точку
    сближения; команда разворачивает цель так, чтобы в момент t_go = R/V_закр
    уходить от позиции ракеты на максимальное боковое расстояние.

    d = (p_ц + v_ц·t_go) − (p_р + v_р·t_go) — прогнозируемый «вектор промаха
    ракеты»; смещение цели за t_go под нормальной командой ~ ½·a·t_go², значит
    рост |d| даёт a ∝ перпендикулярная часть d. Строго коллизионный курс
    (|d| ≈ 0) — градиент вырожден: уходим вбок от линии визирования."""
    s = missile.p - target.p
    v_rel = missile.v - target.v
    rng = float(np.linalg.norm(s))
    vc = _closing_speed(s, v_rel)
    if vc <= 1e-6 or rng < 1e-6:
        return np.zeros(3)  # ракета не налетает — уклоняться не от чего
    t_go = rng / vc
    d = (target.p + target.v * t_go) - (missile.p + missile.v * t_go)
    if float(np.linalg.norm(d)) < 0.05 * rng:
        d = s  # коллизионный курс: шаг с ЛВ (перпендикуляр снимет наводку)
    return _side_direction(target, d, mag)


def negpn_accel(target: Body, missile: Body, n_const: float) -> Vec:
    """«ПН наоборот»: тот же закон ППН a = N·(ω×v), но с отрицательным
    коэффициентом относительно линии визирования «цель → ракета»: вместо
    доворота скорости повороту ЛВ цель доворачивает ПРОТИВ него — угловая
    скорость ЛВ разрастается, ПН-ракета требует всё упреждение и уходит
    в насыщение."""
    s = missile.p - target.p
    v_rel = missile.v - target.v
    omega = los_omega(s, v_rel)
    a = -float(n_const) * np.cross(omega, target.v)
    return _perp(a, target.v)


def cpa_max_accel(target: Body, missile: Body, mag: float) -> Vec:
    """«Градиент CPA»: CPA² = R² − (s·v̂)²; для закрывающейся геометрии
    (s·v̂ < 0) изменение относительной скорости на −a·τ (цель довернула на a)
    даёт d(CPA²) = +k·τ·(−CPA_vec)·a, k > 0, где CPA_vec = s − (s·v̂)v̂ —
    вектор наименьшего сближения. Значит рост CPA максимален при a ∝
    ⊥(−CPA_vec). Вырождение (чисто лобовой: CPA_vec ≈ 0) — тот же запасной
    «нырок в бок», что у away."""
    s = missile.p - target.p
    v_rel = missile.v - target.v
    vv = float(np.dot(v_rel, v_rel))
    if vv < 1e-12:
        return np.zeros(3)
    vn = float(np.sqrt(vv))
    if float(np.dot(s, v_rel)) >= 0.0:
        return np.zeros(3)  # сближения нет — прироста CPA не будет
    cpa_vec = s - (float(np.dot(s, v_rel)) / vv) * v_rel
    return _side_direction(target, -cpa_vec, mag)


def evader_accel(target: Body, missile: Body, sc: Scenario) -> Vec:
    """Команда уклонения выбранного закона (sc.evader_law) для цели в дуэли.
    Ограничение — нормальная перегрузка sc.n_target·g; при n_target = 0 цель
    неманёвренна и летит прямо (нечем уходить)."""
    if sc.n_target <= 0.0:
        return np.zeros(3)
    kind = sc.evader_law
    if kind == "away":
        a = away_accel(target, missile, sc.n_target * G)
    elif kind == "negpn":
        a = negpn_accel(target, missile, sc.pn_n)
    elif kind == "cpa_max":
        a = cpa_max_accel(target, missile, sc.n_target * G)
    elif kind == "brain":
        # мозг-уклонист — состояние (сетчатка, контур) живёт в движке
        # (engine.run строит EvaderBrainSensor); плоский вызов сюда — ошибка проводки
        raise ValueError("закон 'brain' обслуживает engine через EvaderBrainSensor")
    else:
        raise ValueError(f"неизвестный закон уклонения цели: {kind!r}")
    return clip_accel(_perp(a, target.v), sc.n_target)
