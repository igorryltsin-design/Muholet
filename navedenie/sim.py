"""Кинематическая 3D-модель точки-массы: постоянная скорость, нормальное ускорение.

Ограничения модели (честно): нет тяги, сопротивления, изменения массы, гравитации,
угловой динамики корпуса и автопилота. |a|/g — это НОРМАЛЬНОЕ ускорение (поворот
вектора скорости), а не полная перегрузка с учётом гравитации. Модуль скорости
сохраняется при любом манёвре — численного разгона от интегрирования нет.

Цель дополнительно может иметь ПРОДОЛЬНОЕ ускорение (target_speed_mode ≠
constant): оно меняет модуль скорости цели в заданных границах и НЕ считается
нормальной перегрузкой (n_target — по-прежнему только вираж).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

G = 9.81
Vec = np.ndarray

Maneuver = Literal[
    "straight",
    "turn",
    "weave",
    "weave_var",
    "break",
    "scissors",
    "dive",
    "combo",
]
Aspect = Literal["head-on", "beam", "tail-chase", "free"]
GuideMode = Literal["pn", "bio", "both"]
# профиль скорости цели внутри эпизода: постоянная | разгон | торможение |
# короткий продольный импульс | синусоидальное изменение модуля
TargetSpeedMode = Literal["constant", "accelerate", "decelerate", "pulse", "sine"]
# документированные учебные пределы продольного ускорения цели, g (настраиваемо
# через target_longitudinal_g, распределение обучения живёт в train.PROTOCOL)
TARGET_LONG_G_LIMITS = (0.0, 2.0)


def _unit(v: Vec) -> Vec:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(3)
    return v / n


def _perp(a: Vec, v: Vec) -> Vec:
    """Составляющая a, ортогональная скорости v."""
    s2 = float(np.dot(v, v)) + 1e-12
    return a - v * (float(np.dot(a, v)) / s2)


def clip_accel(a: Vec, n_max: float) -> Vec:
    mag = float(np.linalg.norm(a))
    limit = n_max * G
    if mag > limit and mag > 0:
        return a * (limit / mag)
    return a


@dataclass
class Scenario:
    aspect: Aspect = "head-on"
    v_m: float = 780.0
    v_t: float = 260.0
    range_m: float = 8000.0  # НАЧАЛЬНАЯ наклонная дальность «ракета—цель», м (точно для всех аспектов)
    off_axis_m: float = 400.0
    # ── «Свободная расстановка» (aspect="free"): ручной сценарий без типовых
    # геометрий. СК стенда: старт ракеты — начало отсчёта, +X — вперёд по
    # начальному курсу базовой геометрии; курс (азимут) считается от +X против
    # часовой (к +Y «бок»), подъём — угол вектора скорости над горизонтом, град.
    free_tx: float = 8000.0  # цель X от пуска, м
    free_ty: float = 0.0  # цель Y (боковое смещение), м
    free_talt: float = 4000.0  # цель высота (Z), м над уровнем моря
    free_mhdg: float = 0.0  # начальный азимут скорости ракеты, град.
    free_mclimb: float = 0.0  # начальный подъём ракеты, град.
    free_thdg: float = 180.0  # начальный азимут скорости цели, град. (180 — лоб)
    free_tclimb: float = 0.0  # начальный подъём цели, град.
    alt_m: float = 4000.0
    n_max: float = 30.0  # предел нормального ускорения ракеты, g (не полная перегрузка)
    n_target: float = 0.0  # нормальное ускорение цели, g
    maneuver: Maneuver = "straight"
    pn_n: float = 4.0
    # ── scheduled PN (контрольный закон, отдельный от классической ПН):
    # N_sched = clip(N0 + k_rho·rho, N_min, N_max); oracle-вариант использует
    # истинное Vc/R, сенсорный — только измеренное rho (см. pn.py)
    pn_sched_n0: float = 3.0
    pn_sched_k_rho: float = 0.8
    pn_sched_n_min: float = 2.0
    pn_sched_n_max: float = 6.0
    # ── профиль скорости цели внутри эпизода (версионированные поля):
    target_speed_mode: TargetSpeedMode = "constant"  # constant сохраняет прежнее поведение
    target_longitudinal_g: float = 1.0  # амплитуда продольного ускорения, g (модуль)
    target_speed_min: float = 120.0  # нижняя граница модуля скорости цели, м/с
    target_speed_max: float = 520.0  # верхняя граница модуля скорости цели, м/с
    target_speed_period_s: float = 6.0  # период sine / длительность окна pulse, с
    target_speed_phase: float = 0.0  # фаза профиля, с (сдвиг во времени)
    dt: float = 0.005
    t_max: float = 40.0
    fov_deg: float = 12.0  # узкое поле кадра для oracle-ПН и центрального разрешения BIO
    bio_fov_deg: float = 165.0  # физическое поле зрения BIO (фовеальная сетчатка: плотный центр, редкая периферия)
    seeker_delay_s: float = 0.02
    kill_radius_m: float = 45.0  # радиус срабатывания БЧ (trigger range)
    noise_az_deg: float = 0.0  # σ шума пеленга/угла места, град
    noise_range_m: float = 0.0  # σ шума дальности, м
    lock_drop_p: float = 0.0  # вероятность пропуска захвата на кадр
    seeker_jitter_s: float = 0.0  # случайная доп. задержка ГСЧ: верхняя граница U[0, jitter], с
    retina_death_p: float = 0.0  # ПОСТОЯННАЯ доля «умерших» омматидиев (seed-стабильная маска на прогон)
    retina_dropout_p: float = 0.0  # независимый временный дропаут рецепторов на кадр
    # фотометрия цели: множитель яркости пятна (альбедо/подсветка/фон). Проверка
    # «скрытого дальномера»: theta декодируется из суммарной яркости в известной
    # модели оптики — если политика держится при brightness ≠ 1, она не читает
    # фиксированную яркость как дальность (см. diagnostics.photometric_robustness)
    target_brightness: float = 1.0
    mode: GuideMode = "pn"
    law: str = "pn"  # закон для режима pn: pn | tpn | apn | pure | clos | pn_gsn | pn_sched_oracle | pn_sched_sensor
    # ── модель движения центра масс (раздельный выбор; kinematic_legacy — прежнее
    # поведение без изменений, на нём держатся существующие тесты и эксперименты):
    #   kinematic_legacy — постоянный модуль скорости, ускорение только поворачивает V;
    #   point_mass_3dof    — тяга/сопротивление/тяжесть + инерция исполнительного контура.
    model: str = "kinematic_legacy"
    phys_mass_kg: float = 150.0        # масса (постоянна), кг
    phys_ref_area_m2: float = 0.05     # характерная площадь S, м²
    phys_drag_cx: float = 0.30         # коэффициент лобового сопротивления Cx (подзвуковая база)
    phys_cx_wave: float = 0.0          # прирост ΔCx волнового кризиса на сверхзвуковом плато (0 — Cx постоянен)
    phys_mach_kr: float = 1.0          # число Маха перегиба волнового роста M_кр
    phys_mach_band: float = 0.10       # ширина околозвукового перехода δM
    phys_rho_air: float = 0.736        # плотность воздуха (постоянная), кг/м³
    phys_atmos: bool = False           # True: ρ(H) по стандартной атмосфере ICAO/US1976 (rho_air игнорируется)
    phys_thrust_n: float = 0.0         # тяга P, Н (0 — безмоторный участок)
    phys_burn_time_s: float = 3.0      # время горения, с
    phys_tau_a_s: float = 0.05         # постоянная времени исполнительного контура, с
    phys_wn_act: float = 0.0           # звено 2-го порядка привода: ω_n, рад/с (0 — прежнее 1-е звено)
    phys_zeta_act: float = 1.0         # звено 2-го порядка привода: ζ (демпфирование)
    phys_n_avail_max: float = 30.0     # предел располагаемой нормальной перегрузки, g
    phys_cn_max: float = 0.0           # макс. коэфф. нормальной силы Cn_max: n_расп = min(n_avail_max, q̄·S·Cn/(m·g)); 0 — константа
    phys_da_dt_max: float = 0.0        # предел скорости изменения команды, м/с³ (0 — нет)
    phys_gravity: bool = True          # учитывать силу тяжести
    circuit_gain: float = 1.0
    tau_s: float = 0.025
    # инерция рулевого привода в кинематическом контуре: 1-е апериодическое звено
    # τ·d(a_act)/dt + a_act = a_cmd (по порядку — [Гусев] гл.5: лаг привода как
    # причина промаха); 0 — мгновенное исполнение (прежнее поведение побитно)
    tau_act_s: float = 0.0
    seed: int = 1
    brain: str = "stub"
    # ── дуэль «муха-ракета против мухи-самолёта»: цель уходит реактивно
    # (законы — navedenie/evader.py); вне дуэли поведение не меняется
    duel: bool = False
    evader_law: str = "away"  # away | negpn | cpa_max
    fuse_life_s: float = 30.0  # «боевая жизнь» ракеты в дуэли, с: не взяла — выдохлась


@dataclass
class Body:
    p: Vec
    v: Vec


@dataclass
class Frame:
    t: float
    missile: Vec
    target: Vec
    missile_v: Vec
    target_v: Vec
    a_cmd: Vec
    a_pn: Vec
    n_req: float
    n_lim: float
    range_m: float
    v_c: float
    omega_los: float
    az: float
    el: float
    lock: bool
    miss: float
    seeker: dict
    circuit: dict
    layers: dict[str, float]
    event: str | None = None
    ghost: Vec | None = None  # позиция «призрака» — эталонной ПН на той же геометрии
    ghost_t_hit: float | None = None  # интерполированное время входа призрака в сферу БЧ
    # ── фаза сближения (декодирована из изображения, с задержкой кадра) ──
    theta: float = 0.0  # измеренный угловой размер цели, рад
    theta_dot: float = 0.0  # скорость расширения изображения, рад/с
    rho: float = 0.0  # θ̇/θ — оптическая оценка Vc/R, 1/с
    tau_contact: float = 30.0  # θ/θ̇ — ограниченная оценка времени до контакта, с
    # ── диагностика N_eff (истинная геометрия, постфактум; НЕ для управления) ──
    n_eff: float | None = None
    n_eff_valid: bool = False
    n_eff_reason: str | None = None  # no_geom | saturation | nonfinite
    n_eff_yaw: float | None = None
    n_eff_pitch: float | None = None
    sat: bool = False  # команда на насыщении n_max
    # ── телеметрия физической модели (point_mass_3dof); None в кинематическом режиме
    n_cmd: float | None = None   # заданная нормальная перегрузка команды, g
    n_act: float | None = None   # фактическая (после инерции контура) перегрузка, g
    n_avail: float | None = None # располагаемая перегрузка, g
    speed_ms: float | None = None  # модуль скорости ракеты, м/с (меняется в физ. режиме)
    tgo: float | None = None  # DEPRECATED alias: то же, что t_radial = R/Vc, с
    t_radial: float | None = None  # радиальная оценка оставшегося времени R/Vc, с
    t_cpa: float | None = None  # время до ближайшего сближения −r·Vотн/|Vотн|², с
    speed_mode: str = "constant"  # профиль скорости цели в этом прогоне
    target_speed: float = 0.0  # фактический модуль скорости цели, м/с
    # плотность воздуха ρ_возд, применённая в силе сопротивления на этом шаге,
    # кг/м³ (только физ. режим; `rho` в кадре — оптический θ̇/θ, не путать)
    atm_rho: float | None = None
    # число Маха M = V/a(H) по post-step скорости кадра (рядом с speed_ms);
    # при cx_wave = 0 сохраняется прежний постоянный Cx, но телеметрия M честна
    # всегда в физ-режиме
    mach: float | None = None
    # снимок контура цели-уклониста (дуэль, evader_law='brain'): слои энергии,
    # DN и корзина активности её нейронов; None — цель летит по закону/нотам
    evader: dict | None = None


@dataclass
class RunResult:
    frames: list[Frame] = field(default_factory=list)
    miss_m: float = 1e9  # DEPRECATED alias: то же, что cpa_m (минимальное геометрическое расстояние)
    cpa_m: float = 1e9  # R_min — минимальное расстояние сближения (непрерывный минимум за полёт), м
    trigger_range_m: float = 45.0  # R_trigger — радиус/дальность срабатывания (legacy kill_radius), м
    hit: bool = False  # «Перехват»: сфера срабатывания пересечена (межшагово), без моделирования поражения
    t_hit: float | None = None  # интерполированное время входа в сферу срабатывания
    t_end: float | None = None  # время конца прогона
    fov_lock_frac: float = 0.0
    lock_time_s: float = 0.0  # суммарное время сопровождения, с
    lock_fraction: float = 0.0  # доля времени сопровождения = lock_time / t_end, 0…1
    n_int: float = 0.0  # J_n — интеграл модуля заданной нормальной перегрузки ∫|a_cmd|/g dt, g·с
    n_mean_g: float = 0.0  # среднее нормальное ускорение = effort / фактическая длительность, g
    n_peak: float = 0.0  # пиковое нормальное ускорение, g
    reason: str = "timeout"
    t_guide: float | None = None  # время наведения: t_hit перехвата; None — перехвата не было
    ref_dev_m: float | None = None  # среднее отклонение траектории от призрака-ПН, м
    ref_rms_m: float | None = None  # СКО рассогласования по точкам, м
    ref_nrms: float | None = None  # СКО рассогласования, нормированное НАЧАЛЬНОЙ дальностью
    t_ref: float | None = None  # время наведения призрака-ПН; None — не перехватил
    terminal_zem_m: float | None = None  # LEGACY alias == h_cv_m (геометрический прогноз при неизменных скоростях), м
    h_cv_m: float | None = None  # h_cv «Прогноз минимального расстояния при неизменных скоростях» (прямая экстраполяция), м
    h0_m: float | None = None  # h₀ «Прогнозируемый промах при нулевой дальнейшей команде» (интегрирование free-полёта), м
    end_range_m: float | None = None  # R_end — конечная дистанция при завершении прогона, м
    # η «Угол встречи»: угол между векторами скоростей ракеты и цели в момент
    # наибольшего сближения (при перехвате — кадр срабатывания БЧ), град.;
    # 180° — лобовой курс, 0° — догон (терминальная метрика PNG — Зархан,
    # Tactical Missiles Guidance; Kim–Lee–Grider 1993, impact angle control)
    impact_angle_deg: float | None = None
    metrics_version: int = 4
    model: str = "kinematic_legacy"  # модель движения прогона: kinematic_legacy | point_mass_3dof
    # ── агрегаты N_eff за прогон (только валидные отсчёты) ──
    n_eff_median: float | None = None
    n_eff_q25: float | None = None
    n_eff_q75: float | None = None
    n_eff_min: float | None = None
    n_eff_max: float | None = None
    n_eff_valid_frac: float = 0.0  # доля валидных отсчётов N_eff от числа шагов наведения
    sat_frac: float = 0.0  # доля кадров с насыщением команды
    corr_n_eff_rho: float | None = None  # корреляция N_eff с rho (если выборка достаточна)
    corr_n_eff_tgo: float | None = None  # корреляция N_eff с истинным t_go
    n_eff_count: int = 0
    # ── вердикт дуэли (только Scenario.duel; в обычном прогоне None/False) ──
    duel: bool = False
    duel_result: str | None = None  # "missile" — перехват; "evader" — цель выжила
    fuse_expired: bool = False  # боевая жизнь ракеты истекла без перехвата
    t_survived: float | None = None  # сколько цель продержалась до конца прогона, с


def _percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    return float(np.percentile(np.asarray(xs), p))


def _corr(xs: list[float], ys: list[float]) -> float | None:
    """Персон-корреляция; None, если выборка мала или вырождена."""
    if len(xs) < 12:
        return None
    a = np.asarray(xs, dtype=np.float64)
    b = np.asarray(ys, dtype=np.float64)
    if float(np.std(a)) < 1e-9 or float(np.std(b)) < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def summarize_n_eff(frames: list["Frame"], guide_steps: int | None = None) -> dict:
    """Агрегаты N_eff по кадрам прогона: медиана, квартили, min/max по валидным,
    доли валидности и насыщения, корреляции с rho и t_go."""
    vals: list[float] = []
    rhos: list[float] = []
    tgos: list[float] = []
    steps = 0
    sat = 0
    for fr in frames:
        steps += 1
        sat += int(fr.sat)
        if fr.n_eff_valid and fr.n_eff is not None and np.isfinite(fr.n_eff):
            vals.append(fr.n_eff)
            rhos.append(fr.rho)
            tgos.append(fr.tgo if fr.tgo is not None else np.nan)
    pair_rho = [(v, r) for v, r in zip(vals, rhos) if np.isfinite(r)]
    pair_tgo = [(v, tg) for v, tg in zip(vals, tgos) if np.isfinite(tg)]
    return {
        "n_eff_median": _percentile(vals, 50),
        "n_eff_q25": _percentile(vals, 25),
        "n_eff_q75": _percentile(vals, 75),
        "n_eff_min": min(vals) if vals else None,
        "n_eff_max": max(vals) if vals else None,
        "n_eff_valid_frac": (len(vals) / steps) if steps else 0.0,
        "sat_frac": (sat / steps) if steps else 0.0,
        "corr_n_eff_rho": _corr([p[0] for p in pair_rho], [p[1] for p in pair_rho]),
        "corr_n_eff_tgo": _corr([p[0] for p in pair_tgo], [p[1] for p in pair_tgo]),
        "n_eff_count": len(vals),
    }


def vel_from_angles(speed: float, hdg_deg: float, climb_deg: float) -> Vec:
    """Вектор скорости по модулю, азимуту (от +X против часовой к +Y) и подъёму
    над горизонтом: V·(cosγ·cosψ, cosγ·sinψ, sinγ). Основа «свободной расстановки»."""
    hdg = math.radians(hdg_deg)
    climb = math.radians(climb_deg)
    ch = speed * math.cos(climb)
    return np.array([ch * math.cos(hdg), ch * math.sin(hdg), speed * math.sin(climb)])


def spawn(sc: Scenario) -> tuple[Body, Body]:
    """Ракеты и цели в условной СК: +X — вперёд по начальному курсу ракеты.

    range_m — ТОЧНАЯ начальная наклонная дальность для всех аспектов: геометрия
    аспекта задаёт НАПРАВЛЕНИЕ на цель, затем вектор нормируется к range_m.
    v_t — фактический начальный МОДУЛЬ скорости цели для всех аспектов; аспект
    меняет только НАПРАВЛЕНИЕ вектора скорости (скрытых множителей нет).

    Исключение — аспект "free" («свободная расстановка», ручной сценарий): цель
    берётся в точках (free_tx, free_ty, free_talt) от точки пуска, а начальные
    скорости — по азимуту/подъёму free_mhdg/free_mclimb и free_thdg/free_tclimb;
    range_m в этой ветке не используется (дальность задаётся самой расстановкой)."""
    p_m = np.array([0.0, 0.0, sc.alt_m])
    v_m = np.array([sc.v_m, 0.0, 0.0])
    if sc.aspect == "free":
        p_t = np.array([float(sc.free_tx), float(sc.free_ty), float(sc.free_talt)])
        v_m = vel_from_angles(sc.v_m, sc.free_mhdg, sc.free_mclimb)
        v_t = vel_from_angles(sc.v_t, sc.free_thdg, sc.free_tclimb)
        return Body(p_m, v_m), Body(p_t, v_t)
    if sc.aspect == "head-on":
        raw = np.array([1.0, sc.off_axis_m / max(sc.range_m, 1.0), 80.0 / max(sc.range_m, 1.0)])
        v_t = np.array([-sc.v_t, 0.0, 0.0])
    elif sc.aspect == "beam":
        raw = np.array([0.55, -0.65, 0.0])
        v_t = np.array([0.0, sc.v_t, 0.0])
    else:
        raw = np.array([0.45, 0.4 * sc.off_axis_m / max(sc.range_m, 1.0), 40.0 / max(sc.range_m, 1.0)])
        v_t = np.array([sc.v_t, 0.0, 0.0])
    p_t = p_m + raw * (sc.range_m / float(np.linalg.norm(raw)))
    return Body(p_m, v_m), Body(p_t, v_t)


def target_long_accel(body: Body, sc: Scenario, t: float) -> float:
    """Продольное ускорение цели, м/с² (вдоль вектора скорости — меняет МОДУЛЬ).

    Профили (все детерминированы при заданном сценарии, фаза — target_speed_phase):
    constant — 0 (прежнее поведение в точности);
    accelerate — постоянный разгон +A;
    decelerate — постоянное торможение −A;
    pulse — один короткий импульс длительностью 0.15·периода, старт на фазе;
    sine — A·sin(2π(t−фаза)/период)."""
    if sc.target_speed_mode == "constant":
        return 0.0
    a = abs(float(sc.target_longitudinal_g)) * G
    mode = sc.target_speed_mode
    if mode == "accelerate":
        return a
    if mode == "decelerate":
        return -a
    if mode == "pulse":
        tau = t - float(sc.target_speed_phase)
        period = max(float(sc.target_speed_period_s), 0.2)
        return a if 0.0 <= tau < 0.15 * period else 0.0
    if mode == "sine":
        period = max(float(sc.target_speed_period_s), 0.2)
        return a * float(np.sin(2.0 * np.pi * (t - float(sc.target_speed_phase)) / period))
    return 0.0


def integrate_target(body: Body, sc: Scenario, t: float, dt: float, a_norm: Vec | None = None) -> None:
    """Шаг цели: нормальное ускорение (вираж, |v| сохраняется) + продольное
    (меняет модуль в границах target_speed_min/max). Позиция — трапеция.
    Продольная составляющая НЕ входит в нормальную перегрузку n_target.
    При target_speed_mode='constant' поведение в точности как integrate().
    a_norm — внешняя команда нормального ускорения (дуэль: закон-уклонист из
    evader.py); None — прежний слепой манёвр target_accel(body, sc, t)."""
    a_long = target_long_accel(body, sc, t)
    v0 = body.v
    speed = float(np.linalg.norm(v0))
    p_before = body.p.copy()
    integrate(body, target_accel(body, sc, t) if a_norm is None else a_norm, dt)
    if speed < 1e-9 or abs(a_long) < 1e-12:
        return
    speed1 = float(np.clip(speed + a_long * dt, float(sc.target_speed_min), float(sc.target_speed_max)))
    new_speed = float(np.linalg.norm(body.v))
    if new_speed > 1e-9 and abs(speed1 - new_speed) > 1e-12:
        body.v = body.v * (speed1 / new_speed)
    body.p = p_before + (v0 + body.v) * (0.5 * dt)


def target_accel(body: Body, sc: Scenario, t: float) -> Vec:
    """Манёвр цели. Горизонтальная перегрузка — в плоскости, перпендикулярной курсу;
    «dive/combo» добавляют вертикальную составляющую (горка/пике)."""
    if sc.maneuver == "straight" or sc.n_target <= 0:
        return np.zeros(3)
    lift = _unit(np.cross(body.v, np.array([0.0, 0.0, 1.0])))
    if float(np.linalg.norm(lift)) < 0.2:
        lift = _unit(np.cross(body.v, np.array([0.0, 1.0, 0.0])))
    vert = np.array([0.0, 0.0, 1.0])
    n = sc.n_target * G
    m = sc.maneuver

    if m == "turn":
        a = lift * n
    elif m == "weave":
        a = lift * n * np.sin(2.0 * np.pi * t / 4.0)
    elif m == "weave_var":
        # переменная змейка: частота нарастает (чирп) + выход на амплитуду
        phase = 2.0 * np.pi * (t / 4.0 + 0.055 * t * t)
        ramp = min(1.0, 0.35 + t / 6.0)
        a = lift * n * ramp * np.sin(phase)
    elif m == "break":
        # форсированный вираж: разгон прямо, затем резкий разворот на полной перегрузке
        a = lift * n if t > 2.5 else np.zeros(3)
    elif m == "scissors":
        # «ножницы»: чередование виражей влево/вправо с плавным фронтом
        sign = 1.0 if int(t // 1.6) % 2 == 0 else -1.0
        frac = t % 1.6
        front = min(1.0, frac / 0.35, (1.6 - frac) / 0.35)
        a = lift * n * sign * front
    elif m == "dive":
        # горка/пике: вертикальная перегрузка качается вверх-вниз
        a = vert * n * np.sin(2.0 * np.pi * t / 6.0)
    elif m == "combo":
        # комбо: вираж с модуляцией + вертикальная «горка» в противофазе
        a = lift * n * (0.6 + 0.4 * np.sin(2.0 * np.pi * t / 5.0))
        a = a + vert * n * 0.45 * np.sin(2.0 * np.pi * t / 3.0 + 1.0)
    else:
        a = np.zeros(3)
    # полный вектор ускорения цели ограничен n_target·g (включая «всё сразу»)
    mag = float(np.linalg.norm(a))
    if mag > n > 0.0:
        a = a * (n / mag)
    return _perp(a, body.v)


def los_omega(r: Vec, v_rel: Vec) -> Vec:
    r2 = float(np.dot(r, r)) + 1e-12
    return np.cross(r, v_rel) / r2


def closing_speed(r: Vec, v_rel: Vec) -> float:
    return float(-np.dot(_unit(r), v_rel))


def time_to_cpa(r: Vec, v_rel: Vec) -> float:
    """Время до ближайшего сближения (CPA) при неизменных скоростях:
    t_cpa = −(r · Vотн) / |Vотн|². Отрицательно, если сближение уже позади.
    Это НЕ радиальная оценка R/Vc — другая величина (см. radial_time)."""
    vv2 = float(np.dot(v_rel, v_rel))
    if vv2 < 1e-12:
        return 0.0
    return float(-np.dot(r, v_rel) / vv2)


def radial_time(r: Vec, v_rel: Vec) -> float | None:
    """Радиальная оценка оставшегося времени t_radial = R / Vc (Vc — скорость
    сближения). None, если сближения нет (Vc ≤ 0). Отличается от t_cpa тем,
    что не учитывает боковую составляющую промаха."""
    vc = closing_speed(r, v_rel)
    if vc <= 1e-6:
        return None
    return float(np.linalg.norm(r) / vc)


def integrate(body: Body, a: Vec, dt: float) -> None:
    """Шаг кинематической точки-массы: постоянный модуль скорости, ускорение
    поворачивает ВЕКТОР СКОРОСТИ (нормальное ускорение), позиция — трапеция.

    Это устраняет численный рост скорости от интегрирования ортогонального
    ускорения (было: v += a·dt при a ⊥ v увеличивало |v| на O(dt²))."""
    v0 = body.v
    speed = float(np.linalg.norm(v0))
    a_mag = float(np.linalg.norm(a))
    if speed < 1e-9:
        return
    if a_mag < 1e-12:
        body.p = body.p + v0 * dt
        return
    # ось поворота = v × a (поворот вектора скорости НАВСТРЕЧУ ускорению), угол = |a|·dt/|v|
    axis = np.cross(v0, a) / max(a_mag * speed, 1e-12)
    theta = min(a_mag * dt / speed, np.pi)  # шаг угла ограничен π
    c, s = float(np.cos(theta)), float(np.sin(theta))
    v1 = v0 * c + np.cross(axis, v0) * s + axis * (float(np.dot(axis, v0)) * (1.0 - c))
    body.p = body.p + (v0 + v1) * (0.5 * dt)  # трапеция
    body.v = v1


def step_encounter(
    p_m0: Vec, p_m1: Vec, p_t0: Vec, p_t1: Vec, kill_radius: float
) -> dict:
    """Непрерывная встреча на одном шаге: относительный отрезок r(α) = r0 + α·dr.

    Возвращает честные, слабо зависящие от dt величины:
    - cpa — минимальное расстояние внутри шага (непрерывный минимум);
    - hit — пересечена ли сфера БЧ (включая случай, когда шаг начался внутри сферы);
    - alpha_in — доля шага, на которой происходит вход в сферу (0…1), для t_hit."""
    r0 = p_t0 - p_m0
    dr = (p_t1 - p_m1) - r0
    a = float(np.dot(dr, dr))
    b = 2.0 * float(np.dot(r0, dr))
    c0 = float(np.dot(r0, r0))
    r0_in = c0 <= kill_radius * kill_radius

    # CPA: минимум квадратичной функции на [0, 1]
    if a > 1e-18:
        alpha_star = min(1.0, max(0.0, -b / (2.0 * a)))
    else:
        alpha_star = 0.0
    r_star = r0 + dr * alpha_star
    cpa = float(np.linalg.norm(r_star))
    if a <= 1e-18:
        cpa = float(np.linalg.norm(r0))

    # пересечение сферы: |r0 + α·dr|² = R²
    hit = False
    alpha_in = None
    if r0_in:
        hit = True
        alpha_in = 0.0
    elif a > 1e-18:
        disc = b * b - 4.0 * a * (c0 - kill_radius * kill_radius)
        if disc >= 0.0:
            sq = float(np.sqrt(disc))
            alpha1 = (-b - sq) / (2.0 * a)
            if 0.0 <= alpha1 <= 1.0:
                hit = True
                alpha_in = alpha1

    return {"cpa": cpa, "hit": hit, "alpha_in": alpha_in}
