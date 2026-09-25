"""Трёхстепенная физическая модель движения центра масс (`point_mass_3dof`).

Отдельный режим рядом с кинематической моделью (`kinematic_legacy`, см. sim.py).
Кинематическая модель сохраняется без изменений: на неё опираются существующие
тесты и сохранённые эксперименты.

Что моделируется (по русским учебным источникам по динамике полёта — Мануйленко/
Удин, гл. 1, 3, 4; здесь — упрощённая точка-масса без динамики корпуса):
  - тяга P вдоль вектора скорости (ограничена временем горения);
  - аэродинамическое сопротивление X = 0.5·rho·V²·S·Cx, направлено против скорости;
  - сила тяжести (g вдоль −Z в земной системе);
  - нормальное управляющее ускорение a_н — команда закона наведения;
  - заданная команда a_cmd и фактически реализованная a_act разделены;
  - конечное быстродействие исполнительного контура: по умолчанию апериодическое
    звено 1-го порядка tau_a·d(a_act)/dt + a_act = a_cmd; при wn_act > 0 —
    колебательное звено 2-го порядка
    d²a_act/dt² + 2ζω_n·da_act/dt + ω_n²·a_act = ω_n²·a_cmd
    (стандартная модель рулевого привода: Зархан, Tactical Missiles Guidance;
    [Гусев 1996] гл. 5);
  - ограничение располагаемой перегрузки (константа либо из скоростного напора
    при cn_max > 0) и предела скорости изменения команды.

Честные ограничения (важно):
  - масса постоянна (расход топлива под тягой НЕ моделируется);
  - плотность воздуха: постоянная rho_air (по умолчанию) либо стандартная
    атмосфера ICAO/US1976 ρ(H) при atmos=True (atmos.py);
  - аэродинамика — демонстрационная: Cx постоянный либо с S-образным волновым
    кризисом Cx(M) при cx_wave > 0; от угла атаки не зависит;
  - угловая динамика корпуса и рулей отсутствует (управление — прямая команда ускорения);
  - это НЕ модель конкретной реальной ракеты и не оценка боевой эффективности.

Разложение полного ускорения на пути: нормальная составляющая (⊥ V) поворачивает
вектор скорости, сохраняя модуль; продольная составляющая (∥ V) меняет модуль.
При нулевой продольной силе модуль скорости сохраняется точно (проверено тестом).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from navedenie.atmos import air_density, speed_of_sound
from navedenie.sim import G, Vec, _perp, _unit


@dataclass
class PhysicsParams:
    """Демонстрационные параметры модели. Значения по умолчанию НЕ привязаны к
    конкретному реальному изделию и служат для наглядной демонстрации физики."""

    mass_kg: float = 150.0
    ref_area_m2: float = 0.05          # характерная площадь S, м²
    drag_cx: float = 0.30              # коэффициент лобового сопротивления Cx (подзвуковая база)
    cx_wave: float = 0.0               # прирост ΔCx волнового кризиса на сверхзвуковом плато
                                       # (0 — Cx постоянен, прежнее поведение побитно)
    mach_kr: float = 1.0               # число Маха перегиба волнового роста M_кр
    mach_band: float = 0.10            # ширина околозвукового перехода δM, с
    rho_air: float = 0.736             # плотность воздуха, кг/м³ (ПОСТОЯННАЯ, выс. ~7 км)
    atmos: bool = False                # True: ρ берётся из стандартной атмосферы ICAO/US1976
                                       # по текущей высоте (rho_air игнорируется)
    thrust_n: float = 0.0             # тяга двигателя P, Н (0 — безмоторный участок)
    burn_time_s: float = 3.0          # время работы двигателя, с
    tau_a_s: float = 0.05             # постоянная времени исполнительного контура, с
    wn_act: float = 0.0               # собственная частота привода 2-го порядка ω_n, рад/с
                                      # (0 — звено 1-го порядка tau_a_s, прежнее поведение)
    zeta_act: float = 1.0             # коэффициент демпфирования ζ привода 2-го порядка
    n_avail_max: float = 30.0         # предел располагаемой нормальной перегрузки, g
    cn_max: float = 0.0               # макс. коэффициент нормальной силы Cn_max (на α_max/руле)
                                      # (0 — константный предел n_avail_max, прежнее поведение;
                                      # >0 — располагаемая перегрузка ограничена скоростным напором)
    da_dt_max: float = 0.0            # предел скорости изменения команды, м/с³ (0 — без ограничения)
    g: float = G                      # ускорение свободного падения; 0 — отключить тяжесть


def rho_at_h(params: PhysicsParams, h_m: float) -> float:
    """Плотность воздуха на высоте h_m, м: константа params.rho_air либо
    стандартная атмосфера ICAO/US1976 при params.atmos = True."""
    return air_density(h_m) if params.atmos else float(params.rho_air)


def mach_at(speed_ms: float, h_m: float) -> float:
    """Число Маха M = V/a(H) с локальной скоростью звука стандартной атмосферы
    (температурная структура US1976 берётся всегда — это не зависимость от флага atmos)."""
    return float(speed_ms) / speed_of_sound(h_m)


def drag_cx_at(speed_ms: float, h_m: float, params: PhysicsParams) -> float:
    """Полный коэффициент лобового сопротивления Cx(M) = Cx₀ + ΔCx_волн(M).

    При cx_wave = 0 — прежний постоянный Cx (побитно). Иначе околозвуковой
    волновой кризис сопротивления аппроксимирован S-образным ростом от подзвукового
    плато Cx₀ к сверхзвуковому Cx₀ + ΔCx (Бертин, Каммингс, Аэродинамика.
    Основы: рост волнового сопротивления в околозвуковой области):

        Cx(M) = Cx₀ + ΔCx · ½·(1 + tanh((M − M_кр)/δM))

    Честно: локальный максимум (пик около M ≈ 1.05…1.2) и последующее снижение
    с ростом M моделью НЕ воспроизводятся — только монотонный S-образный рост.
    Зависимость Cx от угла атаки — тоже вне рамок демонстрационной модели."""
    if params.cx_wave <= 0.0:
        return float(params.drag_cx)
    m = mach_at(speed_ms, h_m)
    s = 0.5 * (1.0 + float(np.tanh((m - params.mach_kr) / max(params.mach_band, 1e-9))))
    return float(params.drag_cx) + float(params.cx_wave) * s


@dataclass
class FlightState:
    """Состояние точки-массы: положение, скорость, фактическое нормальное ускорение."""

    p: Vec
    v: Vec
    a_act: Vec = field(default_factory=lambda: np.zeros(3))
    da_act: Vec = field(default_factory=lambda: np.zeros(3))  # da_act/dt — нужна только звену 2-го порядка

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.v))


def drag_accel(v: Vec, params: PhysicsParams, h_m: float = 0.0) -> Vec:
    """Ускорение сопротивления: −(0.5·rho·|V|²·S·Cx(M) / m)·V̂ (против вектора скорости).
    При params.atmos плотность берётся из стандартной атмосферы на высоте h_m;
    Cx(M) — постоянный либо с волновым кризисом при cx_wave > 0 (см. drag_cx_at)."""
    speed = float(np.linalg.norm(v))
    rho = rho_at_h(params, h_m)
    if speed < 1e-9 or rho <= 0.0 or params.ref_area_m2 <= 0.0:
        return np.zeros(3)
    q = 0.5 * rho * speed * speed * params.ref_area_m2 * drag_cx_at(speed, h_m, params)
    return -_unit(v) * (q / max(params.mass_kg, 1e-9))


def thrust_accel(v: Vec, params: PhysicsParams, t: float) -> Vec:
    """Ускорение тяги вдоль вектора скорости, пока идёт участок горения."""
    if params.thrust_n <= 0.0 or t >= params.burn_time_s:
        return np.zeros(3)
    return _unit(v) * (params.thrust_n / max(params.mass_kg, 1e-9))


def gravity_accel(params: PhysicsParams) -> Vec:
    """Ускорение тяжести в земной системе (внутреннее хранение: Z — вверх)."""
    return np.array([0.0, 0.0, -params.g])


def n_available(V: float, H: float, params: PhysicsParams) -> float:
    """Располагаемая нормальная перегрузка, g.

    При cn_max = 0 — ПРОСТОЙ КОНСТАНТНЫЙ предел params.n_avail_max (прежнее
    поведение побитно; честно, без имитации полного аэродинамического расчёта).

    При cn_max > 0 включается граница динамического полёта: подъёмная сила
    конечна, и реализумая перегрузка растёт со скоростным напором (Зархан,
    Tactical Missiles Guidance: available g = q·S·C_N/m; классика динамики
    полёта — Мануйленко/Удин):

        n_расп(V, H) = min( n_констр , ½·ρ(H)·V²·S·Cn_max / (m·g) )

    на малых V команда больше не исполняется («ракету нечем повернуть»),
    на больших — работает конструкционный предел n_avail_max. Плотность —
    та же воздушная модель, что и в сопротивлении (rho_air или ρ(H) при atmos).
    """
    n_struct = float(params.n_avail_max)
    if params.cn_max <= 0.0:
        return n_struct
    rho = rho_at_h(params, H)
    v = max(float(V), 0.0)
    n_aero = 0.5 * rho * v * v * params.ref_area_m2 * params.cn_max \
        / (max(params.mass_kg, 1e-9) * G)
    return min(n_struct, n_aero)


def _limit_mag(a: Vec, limit: float) -> Vec:
    mag = float(np.linalg.norm(a))
    if limit > 0.0 and mag > limit:
        return a * (limit / mag)
    return a


def _second_order_step(x: Vec, v: Vec, u: Vec, wn: float, zeta: float, h: float) -> tuple[Vec, Vec]:
    """Точное на интервале h решение звена 2-го порядка
    ẍ = ω_n²(u − x) − 2ζω_n·ẋ при постоянном u (держановский ЗОХ-переходник).

    Формулы по трём режимам демпфирования (ζ<1 колебательный, ζ=1 критический,
    ζ>1 передемпфированный) — аналитика из теории управления; устойчива при любом
    шаге dt, в отличие от явной Эйлера-схемы (ОДН-переходник)."""
    y0 = x - u
    a = zeta * wn
    if abs(zeta - 1.0) < 1e-9:  # критический режим
        e = np.exp(-wn * h)
        c = v + wn * y0
        y1 = e * (y0 + c * h)
        v1 = e * (v - wn * c * h)
    elif zeta < 1.0:  # колебательный режим
        wd = wn * float(np.sqrt(1.0 - zeta * zeta))
        e = np.exp(-a * h)
        cs, sn = np.cos(wd * h), np.sin(wd * h)
        y1 = e * (y0 * cs + ((v + a * y0) / wd) * sn)
        v1 = e * (v * cs - ((wn * wn * y0 + a * v) / wd) * sn)
    else:  # передемпфированный режим
        b = wn * float(np.sqrt(zeta * zeta - 1.0))
        s1, s2 = -a + b, -a - b
        A = (v - s2 * y0) / (s1 - s2)
        B = y0 - A
        e1, e2 = np.exp(s1 * h), np.exp(s2 * h)
        y1 = A * e1 + B * e2
        v1 = A * s1 * e1 + B * s2 * e2
    return u + y1, v1


def step_physics(
    state: FlightState,
    a_cmd: Vec,
    params: PhysicsParams,
    t: float,
    dt: float,
) -> FlightState:
    """Один шаг трёхстепенной модели при заданной команде нормального ускорения a_cmd.

    Порядок (документирован и закреплён тестами):
      1. команда закона наведения a_cmd (уже ⊥ V, ограничена законом);
      2. проекция команды на плоскость, перпендикулярную скорости;
      3. ограничение располагаемой перегрузки n_available(V, H)·g
         (константа n_avail_max либо граница динамического полёта при cn_max > 0);
      4. ограничение скорости изменения команды da_dt_max;
      5. динамика исполнительного контура: 1-е звено tau_a·d(a_act)/dt + a_act = a_cmd,
         а при wn_act > 0 — 2-е звено ä_act + 2ζω_n·ȧ_act + ω_n²·a_act = ω_n²·a_cmd;
      6. реальные силы (тяга, сопротивление, тяжесть) прибавляются к скорости как
         ускорения (меняют и модуль, и направление);
      7. нормальное управляющее ускорение a_act поворачивает вектор скорости,
         сохраняя его модуль; позиция — трапеция.
    Возвращает НОВОЕ состояние (исходное не мутируется)."""
    v0 = state.v
    speed0 = float(np.linalg.norm(v0))
    if speed0 < 1e-9:
        return FlightState(p=state.p.copy(), v=v0.copy(), a_act=state.a_act.copy(),
                           da_act=state.da_act.copy())

    # 2–3: нормальная команда, ограниченная располагаемой перегрузкой
    #      (константный предел либо граница динамического полёта n_available(V, H))
    a_cmd_n = _perp(np.asarray(a_cmd, dtype=float), v0)
    a_cmd_n = _limit_mag(a_cmd_n, n_available(speed0, float(state.p[2]), params) * G)

    # 4: ограничение скорости изменения команды (по модулю приращения за шаг)
    prev_cmd = state.a_act
    delta = a_cmd_n - prev_cmd
    if params.da_dt_max > 0.0:
        max_delta = params.da_dt_max * dt
        delta = _limit_mag(delta, max_delta)
    a_cmd_lim = prev_cmd + delta

    # 5: исполнительный контур. По умолчанию — апериодическое звено 1-го порядка
    #    (неявно — устойчиво при любом dt). При wn_act > 0 — звено 2-го порядка
    #    ẍ = ω_n²(u−x) − 2ζω_nẋ с точным ЗОХ-переходником на шаге dt.
    if params.wn_act > 1e-9:
        a_act, da_act = _second_order_step(prev_cmd, state.da_act, a_cmd_lim,
                                           float(params.wn_act), float(params.zeta_act), dt)
        a_act = _perp(a_act, v0)   # фактическое ускорение — нормальное (⊥ скорости)
        da_act = _perp(da_act, v0)
    else:
        da_act = state.da_act
        if params.tau_a_s > 1e-9:
            k = dt / (params.tau_a_s + dt)
        else:
            k = 1.0  # мгновенное исполнение
        a_act = prev_cmd + (a_cmd_lim - prev_cmd) * k
        a_act = _perp(a_act, v0)  # фактическое ускорение — нормальное (⊥ скорости)

    # 6: РЕАЛЬНЫЕ силы (тяга, сопротивление, тяжесть) интегрируются прибавлением
    #    вектора ускорения — они меняют и направление, и модуль скорости (совершают
    #    работу). Нормальное УПРАВЛЯЮЩЕЕ ускорение a_act работы не совершает:
    #    оно лишь поворачивает вектор скорости, сохраняя его модуль.
    a_force = thrust_accel(v0, params, t) + drag_accel(v0, params, float(state.p[2])) + gravity_accel(params)
    v_mid = v0 + a_force * dt
    speed_mid = float(np.linalg.norm(v_mid))
    if speed_mid < 1e-9:
        return FlightState(p=state.p.copy(), v=np.zeros(3), a_act=a_act, da_act=da_act)

    a_mag = float(np.linalg.norm(a_act))
    if a_mag < 1e-12:
        v1 = v_mid
    else:
        # поворот v_mid навстречу a_act на угол |a_act|·dt/|v_mid| (точно сохраняет модуль)
        axis = np.cross(v_mid, a_act) / (a_mag * speed_mid)
        theta = min(a_mag * dt / speed_mid, np.pi)
        c, s = float(np.cos(theta)), float(np.sin(theta))
        v1 = v_mid * c + np.cross(axis, v_mid) * s + axis * (float(np.dot(axis, v_mid)) * (1.0 - c))
        v1 = _unit(v1) * speed_mid
    p1 = state.p + (v0 + v1) * (0.5 * dt)
    return FlightState(p=p1, v=v1, a_act=a_act, da_act=da_act)
