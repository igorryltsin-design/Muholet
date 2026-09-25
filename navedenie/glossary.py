"""Канонический словарь терминов и законов стенда «МУХОЛЁТ» — единый источник
истины (§4 технического задания).

Из этого модуля формируются / согласуются: подписи интерфейса, подсказки, справка,
паспорта законов, таблицы сравнения, метаданные API. Не допускается появление
независимых таблиц названий в Python и TypeScript без автоматической проверки —
см. `tests/test_glossary.py` и `tools/gen_glossary_ts.py`.

Обозначения:
  - навигационный коэффициент метода пропорциональной навигации во всём проекте
    есть безразмерная величина `N`. Буква `K` допускается ТОЛЬКО при дословном
    цитировании Гусева и всегда с переходом `N ≡ K` (см. docs/notation.md);
  - каждое поле несёт: идентификатор, русское полное/краткое имя, символ,
    единицу, определение, формулу, источник, статус, legacy-имена, инфо-группу.

Статусы законов: «книжный» | «книжный по геометрическому условию» |
«реализация стенда» | «экспериментальный» | «эвристический» |
«диагностический показатель, не закон наведения».
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class Term:
    id: str
    rus_full: str                       # русское полное название
    rus_short: str = ""                 # русское краткое название
    symbol: str = ""                    # математический символ
    unit: str = "безразм."              # единица измерения
    definition: str = ""                # точное определение
    formula: str = ""                   # формула (в обозначениях стенда)
    source: str = ""                    # источник; иначе «Реализация стенда…»
    status: str = "реализация стенда"   # статус (см. таблицу выше)
    legacy: tuple[str, ...] = ()        # старые совместимые машинные имена
    info_group: str = ""                # информационная группа (oracle|sensor|geometry|…)


# ── Законы наведения (§5) ───────────────────────────────────────────────────
# Ключ — внутренний (legacy) id, сохранён для совместимости API.
LAWS: dict[str, Term] = {
    "pn": Term(
        id="pn", rus_full="Метод пропорциональной навигации", rus_short="ПН",
        symbol="N", unit="безразм.",
        definition="Скорость поворота вектора скорости пропорциональна угловой скорости линии визирования.",
        formula="ψ̇_к = N·φ̇  →  a_н = N·(ω_ЛВ × V_к)",
        source="[Гусев] §5.3, ур. (5.15), с. 102–103", status="книжный",
        legacy=("pn", "PPN", "proportional navigation"), info_group="oracle",
    ),
    "tpn": Term(
        id="tpn", rus_full="Истинная пропорциональная навигация (ТПН)", rus_short="истинная ПН",
        symbol="N", unit="безразм.",
        definition="Нормальное ускорение перпендикулярно линии визирования, пропорциональное угловой скорости ЛВ и закрывающей скорости.",
        formula="a = N·V_c·(ω_ЛВ × r̂), V_c = −(r̂ · V_отн)",
        source="Зархан, Tactical Missiles Guidance, гл. 2 (true proportional navigation); Siourinas et al. 2021",
        status="книжный", legacy=("tpn", "TPN", "true PN"), info_group="oracle",
    ),
    "apn": Term(
        id="apn", rus_full="ПН с компенсацией нормального ускорения цели", rus_short="",
        symbol="N", unit="безразм.",
        definition="ПН с добавкой, пропорциональной нормальной составляющей ускорения цели.",
        formula="a_н = N·(ω_ЛВ × V_к) + (N/2)·a_ц,⊥",
        source="Реализация стенда; книжным источником не нормируется (коэффициент N/2 и знак добавки русским источником не подтверждены).",
        status="экспериментальный", legacy=("apn", "APN", "augmented PN"), info_group="oracle",
    ),
    "pure": Term(
        id="pure", rus_full="Метод погони", rus_short="",
        symbol="ψ = φ", unit="—",
        definition="Вектор скорости ракеты постоянно разворачивается на цель.",
        formula="ψ = φ (предельный случай ПН при N=1)",
        source="[Гусев] §5.4, ур. (5.16), с. 104", status="книжный по геометрическому условию",
        legacy=("pure", "pure pursuit"), info_group="oracle",
    ),
    "clos": Term(
        id="clos", rus_full="Метод трёх точек", rus_short="",
        symbol="Δy", unit="м",
        definition="Удержание ракеты на линии «точка пуска — цель» регулятором по поперечному отклонению.",
        formula="a_н = регулятор(Δy); книжное условие h = r·sinΔψ",
        source="[Гусев] §5.1, ур. (5.1), (5.6), с. 95, 98; регулятор — реализация стенда",
        status="книжный по геометрическому условию", legacy=("clos", "CLOS"), info_group="oracle",
    ),
    "pn_gsn": Term(
        id="pn_gsn", rus_full="ПН по измерениям сенсорного канала ГСН", rus_short="",
        symbol="N", unit="безразм.",
        definition="ПН, где угловая скорость линии визирования восстанавливается из измерений сенсора.",
        formula="a = |V_к|·N·(ω_az·ŷ + ω_el·ẑ)",
        source="[Гусев] §5.3 — геометрическая основа; сенсорная реализация — стенд",
        status="книжный метод в сенсорной реализации", legacy=("pn_gsn",), info_group="sensor",
    ),
    "pn_sched_oracle": Term(
        id="pn_sched_oracle", rus_full="Экспериментальная ПН с переменным навигационным коэффициентом N по точному состоянию",
        rus_short="", symbol="N_sched", unit="безразм.",
        definition="ПН, где N планируются по относительной скорости сближения ρ из точной геометрии.",
        formula="N_sched = clamp(N₀ + k_ρ·ρ, N_min, N_max)",
        source="Реализация стенда; книжным источником не нормируется",
        status="экспериментальный", legacy=("pn_sched_oracle",), info_group="oracle",
    ),
    "pn_sched_sensor": Term(
        id="pn_sched_sensor", rus_full="Экспериментальная ПН с переменным навигационным коэффициентом N по сенсорным измерениям",
        rus_short="", symbol="N_sched", unit="безразм.",
        definition="Как pn_sched_oracle, но ρ и угловые скорости — из измерений сенсора.",
        formula="N_sched = clamp(N₀ + k_ρ·ρ_изм, N_min, N_max)",
        source="Реализация стенда; книжным источником не нормируется",
        status="экспериментальный", legacy=("pn_sched_sensor",), info_group="sensor",
    ),
}

# Отображение legacy id → пользовательская подпись (канон §5, единый источник).
# Русские названия без машинных тегов группы: info_group — отдельное поле API/UI.
LAW_LABEL: dict[str, str] = {
    "pn": "Метод пропорциональной навигации (ПН)",
    "tpn": "Истинная пропорциональная навигация (команда по нормали к ЛВ)",
    "apn": "ПН с компенсацией нормального ускорения цели (APN)",
    "pure": "Метод погони",
    "clos": "Метод трёх точек (CLOS)",
    "pn_gsn": "ПН по измерениям сенсорного канала ГСН",
    "pn_sched_oracle": "Экспериментальная ПН с переменным навигационным коэффициентом N по точному состоянию",
    "pn_sched_sensor": "Экспериментальная ПН с переменным навигационным коэффициентом N по сенсорным измерениям",
}

# Законы реактивного уклонения цели в дуэли «муха-ракета против мухи-самолёта»
# (navedenie/evader.py). Это НЕ законы наведения — отдельный реестр.
EVADER_LABEL: dict[str, str] = {
    "away": "Уклон от точки встречи",
    "negpn": "ПН наоборот (отрицательный коэффициент по ЛВ ракеты)",
    "cpa_max": "Градиент наименьшего сближения",
    "brain": "Мозг-уклонист (обучаемая схема цели)",
}


# ── Метрики и величины (§6) ───────────────────────────────────────────────────
METRICS: dict[str, Term] = {
    "los": Term(
        id="los", rus_full="Линия визирования", rus_short="ЛВ", symbol="ЛВ", unit="—",
        definition="Мгновенное направление «ракета — цель».",
        source="[Гусев] §5.1", status="книжный", legacy=("LOS", "los"), info_group="geometry",
    ),
    "los_rate": Term(
        id="los_rate", rus_full="Угловая скорость линии визирования", rus_short="",
        symbol="ω_ЛВ", unit="рад/с", definition="Скорость поворота линии визирования.",
        formula="ω_ЛВ = (r × V_отн)/|r|²", status="книжный",
        legacy=("LOS rate", "theta_dot", "omega_los"), info_group="geometry",
    ),
    "h0": Term(
        id="h0", rus_full="Прогнозируемый промах без дальнейшего управления", rus_short="Прогнозируемый промах",
        symbol="h₀", unit="м",
        definition="Минимум расстояния при нулевой дальнейшей команде, найденный прогнозным интегрированием используемой модели движения.",
        formula="h₀ = min_t |r(t)| при a_cmd = 0", source="Реализация стенда",
        status="диагностический показатель, не закон наведения",
        legacy=("ZEM", "terminal_zem_m"), info_group="geometry",
    ),
    "h_cv": Term(
        id="h_cv", rus_full="Прогноз минимального расстояния при неизменных скоростях", rus_short="",
        symbol="h_cv", unit="м",
        definition="Геометрический прогноз: текущие векторы скорости ракеты и цели считаются постоянными.",
        formula="h_cv = |r(t_cpa)|, t_cpa = −(r·V_отн)/|V_отн|²", source="Реализация стенда",
        status="диагностический показатель, не закон наведения",
        legacy=("ZEM",), info_group="geometry",
    ),
    "cpa": Term(
        id="cpa", rus_full="Минимальное расстояние сближения", rus_short="", symbol="R_min", unit="м",
        definition="Фактический минимум расстояния «ракета — цель» за прогон.",
        source="Реализация стенда", status="диагностический показатель, не закон наведения",
        legacy=("CPA", "cpa_m", "miss_m"), info_group="geometry",
    ),
    "t_cpa": Term(
        id="t_cpa", rus_full="Прогнозируемое время до минимального сближения", rus_short="",
        symbol="t_cpa", unit="с", definition="Время до ближайшего сближения при неизменных скоростях.",
        formula="t_cpa = −(r·V_отн)/|V_отн|²", source="Реализация стенда",
        status="диагностический показатель, не закон наведения", legacy=("t_cpa",), info_group="geometry",
    ),
    "t_radial": Term(
        id="t_radial", rus_full="Радиальная оценка оставшегося времени", rus_short="",
        symbol="t_radial", unit="с", definition="Дальность, делённая на скорость сближения; НЕ есть время до встречи.",
        formula="t_radial = R / V_c", source="Реализация стенда",
        status="диагностический показатель, не закон наведения", legacy=("tgo", "t_go"), info_group="geometry",
    ),
    "r_end": Term(
        id="r_end", rus_full="Конечная дистанция при завершении моделирования", rus_short="",
        symbol="R_end", unit="м", definition="Расстояние «ракета — цель» в последний момент прогона.",
        source="Реализация стенда", status="диагностический показатель, не закон наведения",
        legacy=("end_range_m",), info_group="geometry",
    ),
    "r_trigger": Term(
        id="r_trigger", rus_full="Расстояние срабатывания", rus_short="", symbol="R_trigger", unit="м",
        definition="Радиус (дальность) срабатывания: порог события «перехват». Поражающее действие не моделируется.",
        source="Реализация стенда", status="диагностический показатель, не закон наведения",
        legacy=("kill_radius_m", "trigger_range_m"), info_group="geometry",
    ),
    "hit": Term(
        id="hit", rus_full="Перехват", rus_short="", symbol="—", unit="событие",
        definition="Пересечение сферы срабатывания. Вероятность поражения и работа боевой части НЕ моделируются.",
        source="Реализация стенда", status="диагностический показатель, не закон наведения",
        legacy=("hit",), info_group="geometry",
    ),
    "control_effort": Term(
        id="control_effort", rus_full="Интеграл модуля заданной нормальной перегрузки", rus_short="",
        symbol="J_n", unit="g·с", definition="Накопленная интенсивность команды; НЕ энергия, НЕ расход топлива.",
        formula="J_n = ∫ |a_cmd|/g dt", source="Реализация стенда",
        status="диагностический показатель, не закон наведения", legacy=("n_int", "effort_gs", "control effort"),
        info_group="control",
    ),
    "n_eff": Term(
        id="n_eff", rus_full="Эквивалентный навигационный коэффициент", rus_short="", symbol="N_экв",
        unit="безразм.",
        definition="Какому постоянному N эквивалентна проекция команды в текущем состоянии; не определён при вырожденной геометрии (ω_ЛВ≈0) и на насыщении.",
        formula="N_экв = ⟨a_cmd, q⟩/⟨q, q⟩, q = ω_ЛВ × V_к", source="Реализация стенда",
        status="диагностический показатель, не закон наведения",
        legacy=("n_eff", "K_eff"), info_group="geometry",
    ),
    "nrms": Term(
        id="nrms", rus_full="Нормированное среднеквадратическое отклонение", rus_short="НСКО", symbol="NRMS",
        unit="безразм.", definition="СКО рассогласования с призраком-эталоном, нормированное на начальную дальность.",
        formula="NRMS = RMS(|r_ракета − r_призрак|) / R₀", source="Реализация стенда",
        status="диагностический показатель, не закон наведения", legacy=("ref_nrms",), info_group="geometry",
    ),
}


# ── Сенсорика и биологический контур (§7) ──────────────────────────────────────
SENSOR: dict[str, Term] = {
    "seeker": Term(
        id="seeker", rus_full="Сенсорный канал ГСН", rus_short="", symbol="—", unit="—",
        definition="Модель оптического сенсора: изображение цели и декодирование углов. Полноценная реальная ГСН не моделируется.",
        source="Реализация стенда", status="реализация стенда", legacy=("seeker",), info_group="sensor",
    ),
    "lock": Term(
        id="lock", rus_full="Сопровождение цели", rus_short="", symbol="—", unit="событие",
        definition="Цель в поле зрения и декодирована. События: «захват цели» / «потеря сопровождения».",
        source="Реализация стенда", status="реализация стенда", legacy=("lock",), info_group="sensor",
    ),
    "fov": Term(
        id="fov", rus_full="Поле зрения", rus_short="", symbol="ПЗ", unit="°",
        definition="Угол поля зрения сенсора; задаётся ПОЛНЫМ углом (fov_deg), не полуразмером.",
        source="Реализация стенда", status="реализация стенда", legacy=("FOV", "fov_deg"), info_group="sensor",
    ),
    "readout": Term(
        id="readout", rus_full="Выходной декодер", rus_short="", symbol="—", unit="—",
        definition="Декодирование углов цели из изображения сенсора.", source="Реализация стенда",
        status="реализация стенда", legacy=("readout",), info_group="sensor",
    ),
    "rho": Term(
        id="rho", rus_full="Относительная скорость оптического расширения", rus_short="", symbol="ρ", unit="с⁻¹",
        definition="ρ = θ̇/θ — относительный рост углового размера цели.", source="Реализация стенда",
        status="реализация стенда", legacy=("rho",), info_group="sensor",
    ),
    "tau_contact": Term(
        id="tau_contact", rus_full="Определение времени до контакта", rus_short="", symbol="τ", unit="с",
        definition="τ = −θ/θ̇ — оптическая оценка времени до контакта при θ̇ > 0; вне определения при θ̇ ≤ 0.",
        source="Реализация стенда", status="диагностический показатель, не закон наведения",
        legacy=("tau_contact",), info_group="sensor",
    ),
    "cmd_pitch": Term(
        id="cmd_pitch", rus_full="Вертикальная команда нормального ускорения", rus_short="",
        symbol="a_z", unit="м/с²",
        definition="Выход декодера/регулятора по вертикали; НЕ угол тангажа корпуса.", source="Реализация стенда",
        status="реализация стенда", legacy=("pitch", "dn_pitch"), info_group="sensor",
    ),
    "cmd_yaw": Term(
        id="cmd_yaw", rus_full="Боковая команда нормального ускорения", rus_short="",
        symbol="a_y", unit="м/с²",
        definition="Выход декодера/регулятора по горизонтали; НЕ угол рыскания корпуса.", source="Реализация стенда",
        status="реализация стенда", legacy=("yaw", "dn_yaw"), info_group="sensor",
    ),
    "oracle": Term(
        id="oracle", rus_full="Эталон по точному состоянию", rus_short="", symbol="—", unit="—",
        definition="Расчёт по истинной геометрии (полная информация о состоянии).", source="Реализация стенда",
        status="реализация стенда", legacy=("oracle",), info_group="oracle",
    ),
    "sensor": Term(
        id="sensor", rus_full="Закон по сенсорным измерениям", rus_short="", symbol="—", unit="—",
        definition="Закон, использующий только измерения после поля зрения, шумов, отказов и задержки.",
        source="Реализация стенда", status="реализация стенда", legacy=("sensor",), info_group="sensor",
    ),
    "saturation": Term(
        id="saturation", rus_full="Доля времени ограничения команды", rus_short="", symbol="—", unit="доля",
        definition="Доля кадров, где команда достигла предела располагаемой перегрузки.", source="Реализация стенда",
        status="диагностический показатель, не закон наведения", legacy=("sat_frac", "saturation fraction"),
        info_group="control",
    ),
}

TERMS: dict[str, Term] = {t.id: t for t in [*LAWS.values(), *METRICS.values(), *SENSOR.values()]}


def all_terms() -> dict[str, Term]:
    return TERMS


def as_json() -> dict[str, Any]:
    """Полный словарь в машиночитаемом виде — для метаданных API и генерации TS."""
    return {
        "version": 1,
        "laws": {k: asdict(v) for k, v in LAWS.items()},
        "metrics": {k: asdict(v) for k, v in METRICS.items()},
        "sensor": {k: asdict(v) for k, v in SENSOR.items()},
    }


def law_label(law_id: str) -> str:
    t = LAWS.get(law_id)
    return LAW_LABEL.get(law_id, t.rus_full if t else law_id)
