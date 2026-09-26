"""Законы наведения: метод пропорциональной навигации (ПН), истинная ПН (TPN,
команда по нормали к линии визирования), ПН с компенсацией ускорения цели
(экспериментальный), метод погони, метод трёх
точек (CLOS), ПН по измерениям ГСН, экспериментальное ПН с переменным
коэффициентом (oracle и сенсорная).

Публичные русские названия и статусы — в LAW_LABEL и в docs/law_passports.md;
устойчивые внутренние id (pn/apn/tpn/pure/clos/pn_gsn/pn_sched_*) не меняются.

Размерности и конвенции (важно):
- ω_LOS = (r × v_rel)/|r|² — вектор угловой скорости линии визирования, рад/с;
- PPN:  a = N · (ω_LOS × v_m)  — произведение (ω × v) уже имеет размерность
  ускорения (рад/с · м/с = м/с²), никаких дополнительных множителей Vc;
- TPN:  a = N · V_c · (ω_LOS × r̂) — та же ω_LOS, но нормаль к линии
  визирования и множитель V_c (закрывающая скорость) вместо скорости ракеты;
- APN:  a = a_PPN + (N/2) · a_t,⊥  — добавка (N/2)·a_t, где a_t,⊥ — компонента
  ускорения цели, ОРТОГОНАЛЬНАЯ скорости ракеты (конвенция стенда; другие
  источники используют ортогональность линии визирования — эффект тот же при
  малых углах);
- «ПН (oracle)» использует точную геометрию r, v — это эталон с подсказкой, а
  не равноправный сенсорный участник; для честного сравнения есть «ПН через
  ГСН» (pn_gsn), которая считает команду только из измеренных углов;
- scheduled PN — ОТДЕЛЬНЫЙ экспериментальный закон с переменным
  N_sched = clip(N0 + k_rho·rho, N_min, N_max): oracle-вариант берёт
  истинное Vc/R (эквивалент 1/t_go), сенсорный — только измеренное rho.
  Классическая ПН с постоянным N остаётся baseline и не подменяется.

Все законы возвращают ускорение, ортогональное скорости ракеты, с ограничением
n_max·g (нормальное ускорение).
"""

from __future__ import annotations

import numpy as np

from navedenie.sim import Vec, clip_accel, closing_speed, los_omega, _perp, _unit

LAWS = ("pn", "tpn", "apn", "pure", "clos", "pn_gsn", "pn_sched_oracle", "pn_sched_sensor")
# Устойчивые внутренние id (pn/apn/…) сохранены ради совместимости данных и API;
# публичные названия — русские, по терминологии профильных учебных изданий
# (см. docs/notation.md и docs/law_passports.md). «ПН» = метод пропорционального
# сближения; «oracle»/«сенсорная» помечают информационный бюджет закона.
# Публичные русские названия законов — единый источник в navedenie.glossary (§4);
# re-export сохранён для совместимости импорта `from navedenie.pn import LAW_LABEL`.
from navedenie.glossary import LAW_LABEL  # noqa: E402



def ppn_accel(r: Vec, v_m: Vec, v_t: Vec, n_const: float, n_max: float) -> Vec:
    """a_ПН = N · (ω_ЛВ × V_к) — размерностно корректный метод пропорциональной навигации.

    Навигационный коэффициент безразмерен и во всём стенде обозначается N.
    Эквивалент закону [Гусев] §5.3, ур. (5.15) `ψ̇_к = K·φ̇` (с. 102–103).
    В обозначениях Гусева коэффициент обозначен K; в обозначениях стенда тот же
    навигационный коэффициент есть N (N ≡ K), поэтому закон записан как
    `ψ̇_к = N·φ̇`. Скорость поворота вектора скорости пропорциональна угловой
    скорости линии визирования; переход к нормальный ускорению даёт
    a_н = V·ψ̇_к = N·V·φ̇ = N·(ω_ЛВ × V_к)."""
    omega = los_omega(r, v_t - v_m)
    a = n_const * np.cross(omega, v_m)
    return clip_accel(_perp(a, v_m), n_max)


def tpn_accel(r: Vec, v_m: Vec, v_t: Vec, n_const: float, n_max: float) -> Vec:
    """Истинная ПН (TPN): a = N·V_c·(ω_ЛВ × r̂) — нормальное УСКОРЕНИЕ,
    перпендикулярное ЛИНИИ ВИЗИРОВАНИЯ (r̂ — единичный вектор ЛВ), а не скорости
    ракеты, и пропорциональное закрывающей скорости V_c = −(r̂ · V_отн).

    Классический закон истинного пропорционального наведения (Зархан, Tactical
    Missiles Guidance, гл. 2: true proportional navigation a_c = N·V_c·ω_LOS
    по нормали к ЛВ; Siourinas et al., Proportional Navigation and its
    Applications, 2021). Отличие от «обычного» ПН выше: множитель V_c вместо
    |V_к| и направление по ⊥ ЛВ вместо ⊥ V_к — при боковой геометрии (большом
    угле встречи ЛВ к скорости) команды расходятся. При убегании (V_c < 0)
    закрывающая скорость обрезается до нуля — команда не разворачивает ракету
    «задом наперёд»; конвенция стенда: итоговая команда, как и у других законов,
    ортогонализуется по V_к и ограничивается n_max·g."""
    omega = los_omega(r, v_t - v_m)
    vc = max(closing_speed(r, v_t - v_m), 0.0)
    a = n_const * vc * np.cross(omega, _unit(r))
    return clip_accel(_perp(a, v_m), n_max)


def n_eff_basis(r: Vec, v_m: Vec, v_t: Vec) -> Vec:
    """Базовый вектор команды ПН при N=1: q = perp(ω_LOS × V_m, V_m).

    (ω×v) уже ⊥ v, so perp оставляет его без изменений; функция фиксирует
    конвенцию. Диагностика N_eff: N_eff = <a_cmd, q>/<q, q> — какому
    коэффициенту ПН эквивалентна проекция команды в текущем состоянии."""
    omega = los_omega(r, v_t - v_m)
    return _perp(np.cross(omega, v_m), v_m)


def n_eff_from(
    a_cmd: Vec,
    r: Vec,
    v_m: Vec,
    v_t: Vec,
    *,
    n_max: float,
    q_min: float = 0.5,
    sat_frac: float = 0.98,
) -> tuple[float | None, str | None, Vec]:
    """Эффективная постоянная навигации N_eff = <a_cmd, q>/<q, q> + причина
    невалидности. Считается ТОЛЬКО для анализа (истинная геометрия используется
    постфактум и никогда не поступает в сенсорные законы/BIO).

    Валидность: геометрия не вырождена (|q| ≥ q_min, м/с² на единицу N),
    команда не на насыщении (|a_cmd| < sat_frac·n_max·g), значения конечны.
    Возвращает (n_eff | None, reason | None, q)."""
    q = n_eff_basis(r, v_m, v_t)
    qn = float(np.linalg.norm(q))
    a_mag = float(np.linalg.norm(a_cmd))
    if not np.isfinite(qn) or not np.isfinite(a_mag) or not np.isfinite(float(np.dot(a_cmd, q))):
        return None, "nonfinite", q
    if qn < q_min:
        return None, "no_geom", q  # почти нулевая ω_LOS — знаменатель meaningless
    if a_mag >= sat_frac * n_max * 9.81:
        return None, "saturation", q
    return float(np.dot(a_cmd, q) / float(np.dot(q, q))), None, q


def scheduled_pn_accel(
    r: Vec,
    v_m: Vec,
    v_t: Vec,
    rho: float,
    n0: float,
    k_rho: float,
    n_sched_min: float,
    n_sched_max: float,
    n_lim: float,
) -> Vec:
    """Scheduled PN: N_sched = clip(N0 + k_rho·rho, N_min, N_max), команда —
    классическая форма N·(ω×v) с переменным коэффициентом и пределом перегрузки
    n_lim. rho имеет единицы 1/с (θ̇/θ); k_rho — единицы секунд (с).
    Oracle передаёт истинное Vc/R, сенсорный — измеренное."""
    n_sched = float(np.clip(n0 + k_rho * float(rho), n_sched_min, n_sched_max))
    return ppn_accel(r, v_m, v_t, n_sched, n_lim)


def scheduled_pn_seeker_accel(
    az_dot: float,
    el_dot: float,
    rho: float,
    v_m: Vec,
    n0: float,
    k_rho: float,
    n_sched_min: float,
    n_sched_max: float,
    n_lim: float,
) -> Vec:
    """Сенсорная scheduled PN: команда только из декодированных угловых скоростей
    (конвенции осей — см. pn_seeker_accel), коэффициент scheduled — из
    ИЗМЕРЕННОГО rho (после фовеи, шумов, отказов и задержки)."""
    n_sched = float(np.clip(n0 + k_rho * float(rho), n_sched_min, n_sched_max))
    return pn_seeker_accel(az_dot, el_dot, v_m, n_sched, n_lim)


def apn_accel(r: Vec, v_m: Vec, v_t: Vec, a_t: Vec, n_const: float, n_max: float) -> Vec:
    """Augmented PN: a = N·(ω×v) + (N/2)·a_t,⊥ (a_t,⊥ ⊥ скорости ракеты).

    При a_t = 0 совпадает с PPN с машинной точностью. Добавка имеет ту же
    размерность ускорения, что и базовый член; насыщение по n_max включается
    только когда сумма действительно превышает предел."""
    omega = los_omega(r, v_t - v_m)
    a_ppn = n_const * np.cross(omega, v_m)
    a = a_ppn + 0.5 * n_const * _perp(a_t, v_m)
    return clip_accel(_perp(a, v_m), n_max)


def pn_seeker_accel(az_dot: float, el_dot: float, v_m: Vec, n_const: float, n_max: float) -> Vec:
    """«ПН по измерениям ГСН»: команда только из ИЗМЕРЕННЫХ углов и их скоростей.

    В осях скоростной (траекторной) системы — x по вектору скорости, y вправо,
    z вверх (ориентация корпуса моделью не задаётся, см. seeker.body_axes):
    ω ≈ ω_az·ẑ − ω_el·ŷ, откуда a = N·(ω × v) даёт компоненты вправо N·V·ω_az
    и ВВЕРХ N·V·ω_el (прямой знак по месту). Использует декодированные из
    изображения угловые скорости — без истинной геометрии, как сенсорный закон."""
    _x, y, z = _body_axes(v_m)
    speed = float(np.linalg.norm(v_m)) + 1e-9
    a = speed * n_const * (float(az_dot) * y + float(el_dot) * z)
    return clip_accel(_perp(a, v_m), n_max)


def _body_axes(v_m: Vec) -> tuple[Vec, Vec, Vec]:
    from navedenie.seeker import body_axes

    return body_axes(v_m)


def pure_pursuit_accel(r: Vec, v_m: Vec, n_max: float) -> Vec:
    """Метод погони: вектор скорости ракеты постоянно разворачивается на цель.

    Геометрическое условие совпадает с [Гусев] §5.4, ур. (5.16), с. 104: ψ = φ
    (угол наклона вектора скорости равен углу линии визирования), т.е. предельный
    случай ПН (5.15) при навигационном коэффициенте N=1 (у Гусева K≡N=1), C=0.
    Реализация — насыщенный доворот скорости к текущей линии визирования
    (`a = (r̂ − V̂к)·n_max·g`); в модели нет угла скольжения, поэтому погоня и
    «прямое наведение» (с. 106) здесь совпадают.
    Недостаток по книге (с. 105): на терминальном участке требуемые перегрузки
    растут и обращаются в бесконечность при φ→180° (цель убегает вдогон) — закон
    годится лишь против скоростных целей и служит наглядным антиэталоном ПН."""
    desired = _unit(r)
    current = _unit(v_m)
    a = (desired - current) * n_max * 9.81
    return clip_accel(_perp(a, v_m), n_max)


def clos_accel(missile_p: Vec, r: Vec, v_m: Vec, target_p: Vec, launch_p: Vec, n_max: float, gain: float = 1.2) -> Vec:
    """Метод трёх точек: ракета удерживается на линии «пуск — цель».

    Условие трёх точек (ракета и цель на одном луче от ПУ) — [Гусев] §5.1,
    ур. (5.1) ψ_ц = ψ_р, с. 95; отклонение от луча h = r·sin Δψ — ур. (5.6), с. 98.
    Здесь ошибка — перпендикулярное смещение ракеты от прямой «пуск—цель», а
    команда пропорциональна ей (усиление `k = gain·|Vк|/max(|r|,300)·2`, растёт
    при сближении). Это регулятор удержания луча по отклонению h, а не интегрирование
    книжной системы (5.2)–(5.4); недостаток метода по книге — перерасход перегрузки."""
    to_t = target_p - launch_p
    dist = float(np.linalg.norm(to_t)) + 1e-9
    frac = float(np.clip(1.0 - np.linalg.norm(target_p - missile_p) / dist, 0.0, 1.0))
    line_p = launch_p + to_t * frac
    err = line_p - missile_p
    k = gain * float(np.linalg.norm(v_m)) / max(float(np.linalg.norm(r)), 300.0) * 2.0
    a = err * k
    return clip_accel(_perp(a, v_m), n_max)
