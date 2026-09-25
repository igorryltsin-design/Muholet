"""Научные прогоны уровня 2: матрица переносимости, scaling-кривая, матрица
скоростей, диагностика N_eff и абляции фазовых признаков.

Оба классических эксперимента отвечают на вопрос «мозг обобщает или заучивает /
сколько нужно мозга»: свежий (необученный) экземпляр сети обучается на узком
наборе манёвров и испытывается на всех, либо экземпляры разного размера
обучаются одинаково и сравниваются по каноническому трио. Живые мозги стенда и
кеш «лучших весов» не затрагиваются: обучение идёт на отдельных экземплярах,
кеш на время прогона изолируется.

Скоростная серия (speed_matrix) — held-out матрица «v_m × v_t × аспект ×
профиль скорости × манёвр» с per-cell метриками (hit rate, CPA-квартили,
усилие, медианный N_eff, доля насыщения, захват). Отчёт N_eff (neff_report)
отвечает на вопрос «постоянный ли коэффициент выучен или он адаптивен».
Абляции (feature_ablation) выключают theta/rho на уровне признаков контура.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Callable

import numpy as np

from navedenie.circuit import ConnectomeCircuit, FlyCircuit
from navedenie.engine import collect
from navedenie.train import (
    PROTOCOL,
    TEST_SPEED_MODES,
    TEST_VM_GRID,
    TEST_VT_GRID,
    _best_cache,
    _episode_scenario,
    _eval_canonical,
    _rollout_update,
)
from navedenie.sim import Scenario

Progress = Callable[[dict[str, Any]], None]

# манёвры обучения и испытаний: обучаем на одном — проверяем на всех (+ прямолинейно)
TRANSFER_MANEUVERS = ("turn", "weave", "scissors", "dive")
TRANSFER_STRAIGHT = "straight"
# испытания: три аспекта (head-on/beam/tail-chase) — геометрическое разнообразие
# вместо фейковых «повторов» с разными seed при нулевой стохастике
TEST_ASPECTS = ("head-on", "beam", "tail-chase")
# манёвры скоростной матрицы (минимальная проверочная матрица)
SPEED_MATRIX_MANEUVERS = ("straight", "weave")


@contextmanager
def _isolated_best_cache():
    """Кеш «лучших весов» общий на вид мозга; научные прогоны на свежих экземплярах
    не должны подменять снапшоты живых мозгов стенда."""
    saved = {k: _best_cache.pop(k) for k in list(_best_cache)}
    try:
        yield
    finally:
        _best_cache.update(saved)


def _fresh(kind: str, size: int | None = None):
    """Свежий необученный экземпляр мозга заданного вида (и размера, если просили)."""
    if kind == "connectome":
        c = ConnectomeCircuit(channels=size) if size else ConnectomeCircuit()
    elif kind == "full":
        c = FlyCircuit(kind="full", pool_size=size or 32)
    else:
        c = FlyCircuit(kind="stub")
    c.trained = False
    return c


def _n_params(circuit) -> int:
    """Число ОБУЧАЕМЫХ параметров: W_dn всегда, скрытые — где есть пластичность."""
    n = int(circuit.W_dn.size)
    if getattr(circuit, "kind", "") in ("full", "connectome"):
        if getattr(circuit, "W_pool", None) is not None:
            n += int(circuit.W_pool.size)
        if getattr(circuit, "W_fx", None) is not None:
            n += int(circuit.W_fx.size)
    return n


def _train_scenario(ep: int, gain: float, maneuver: str) -> Scenario:
    """Учебная геометрия матрицы: та же семья, что и испытания, но с разбросом
    дальности/сближения. Используется как БАТЧ обучения по результату."""
    rng = np.random.default_rng(11 + ep * 7)
    return Scenario(
        aspect="head-on",
        v_m=780.0,
        v_t=float(rng.uniform(200, 300)),
        range_m=float(rng.uniform(6000, 9500)),
        off_axis_m=float(rng.uniform(200, 650)),
        n_max=30.0,
        n_target=0.0 if maneuver == TRANSFER_STRAIGHT else 3.0,
        maneuver=maneuver,  # type: ignore[arg-type]
        pn_n=4.0,
        dt=0.02,
        t_max=12.0,
        fov_deg=14.0,
        kill_radius_m=45.0,
        mode="bio",
        circuit_gain=gain,
        seed=100 + ep,
    )


def _test_scenario(maneuver: str, gain: float, aspect: str) -> Scenario:
    """Испытательная геометрия — как карта преимуществ: 8 км, вираж 3 g, аспект из тройки."""
    return Scenario(
        aspect=aspect,  # type: ignore[arg-type]
        v_m=780.0,
        v_t=260.0,
        range_m=8000.0,
        off_axis_m=420.0,
        n_max=30.0,
        n_target=0.0 if maneuver == TRANSFER_STRAIGHT else 3.0,
        maneuver=maneuver,  # type: ignore[arg-type]
        pn_n=4.0,
        dt=0.02,
        t_max=14.0,
        fov_deg=14.0,
        kill_radius_m=45.0,
        mode="bio",
        circuit_gain=gain,
        seed=7,  # фиксирован: испытание детерминировано, разнообразие — аспектами
    )


def _test_maneuver(circuit, maneuver: str) -> float:
    misses = [
        _rollout_update(circuit, _test_scenario(maneuver, circuit.gain, aspect), lr=0.0)["miss"]
        for aspect in TEST_ASPECTS
    ]
    return float(np.median(misses))


def _cem_polish(circuit, maneuver: str, generations: int = 6, pop: int = 4, batch: int = 2, sigma: float = 0.35) -> float:
    """Обучение «по результату» на ОДНОМ манёвре: CEM по W_dn (замкнутый контур,
    фитнес — фактический промах). Имитация учителя даёт хрупкий в замкнутом
    контуре закон; здесь результат оптимизируется напрямую, как в режиме
    «по результату» UI. Возвращает лучший средний промах."""
    rng = np.random.default_rng(500 + len(maneuver))
    mean_w = circuit.W_dn.copy()
    best_w = mean_w.copy()
    best_fit = float("inf")
    for gen in range(generations):
        cands = [best_w if gen > 0 else mean_w]
        cands += [np.clip(mean_w + rng.normal(0, sigma, mean_w.shape), -4.0, 4.0) for _ in range(pop - len(cands))]
        scored = []
        for w in cands:
            circuit.W_dn = w
            circuit.reset()
            rs = [
                _rollout_update(circuit, _train_scenario(gen * batch + k, circuit.gain, maneuver), lr=0.0)
                for k in range(batch)
            ]
            fit = float(np.median([r["miss"] for r in rs])) + 0.02 * float(np.mean([r["control_effort_gs"] for r in rs]))
            scored.append((fit, w))
        scored.sort(key=lambda x: x[0])
        best_fit, best_w = scored[0]
        mean_w = np.stack([w for _f, w in scored[:2]]).mean(axis=0)
        sigma = float(np.clip(sigma * (0.9 if best_fit < scored[-1][0] else 1.15), 0.08, 0.6))
    circuit.W_dn = best_w
    circuit.reset()
    return best_fit


def transfer_matrix(
    kind: str = "connectome",
    episodes: int = 6,
    maneuvers: tuple[str, ...] = TRANSFER_MANEUVERS,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Матрица переносимости: строка — манёвр обучения, столбец — манёвр испытания.

    На каждый манёвр создаётся свежий мозг и обучается ТОЛЬКО на этом манёвре —
    методом «по результату» (CEM по весам выхода, `episodes` поколений), затем
    испытывается на всех манёврах по трём аспектам. Диагональ «обучался —
    испытан», вне диагонали — перенос навыка. Ответ на «обобщает или заучивает»."""
    t0 = time.time()
    test_list = (TRANSFER_STRAIGHT, *maneuvers)
    rows: list[dict[str, Any]] = []
    with _isolated_best_cache():
        for i, m in enumerate(maneuvers):
            circuit = _fresh(kind)
            train_fit = _cem_polish(circuit, m, generations=max(2, episodes))
            tests = {tm: round(_test_maneuver(circuit, tm), 1) for tm in test_list}
            rows.append(
                {
                    "train": m,
                    "train_miss_med": round(float(train_fit), 1),
                    "tests": tests,
                }
            )
            del circuit
            if progress:
                progress({"row": i + 1, "total": len(maneuvers), "train": m})
    return {
        "kind": kind,
        "episodes": int(episodes),
        "maneuvers": list(maneuvers),
        "test_maneuvers": list(test_list),
        "rows": rows,
        "seconds": round(time.time() - t0, 1),
    }


def scaling_curve(
    kind: str = "connectome",
    sizes: tuple[int, ...] = (32, 64, 128),
    episodes: int = 10,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Scaling-кривая: мозги разного размера обучаются ОДИНАКОВО (одни эпизоды,
    один шаг) и сравниваются по каноническому трио. Прямая проверка «сколько
    мозга окупается»: параметры ↔ промах."""
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    with _isolated_best_cache():
        for i, n in enumerate(sizes):
            circuit = _fresh(kind, size=n)
            before = _eval_canonical(circuit)
            train_miss: list[float] = []
            for ep in range(episodes):
                sc = _episode_scenario(ep, circuit.gain)
                train_miss.append(float(_rollout_update(circuit, sc, lr=0.04)["miss"]))
            after = _eval_canonical(circuit)
            rows.append(
                {
                    "size": int(n),
                    "params": _n_params(circuit),
                    "n_cells": int(circuit.n_cells),
                    "miss_before": round(float(before["miss"]), 1),
                    "miss_after": round(float(after["miss"]), 1),
                    "hit_rate_after": round(float(after["hit_rate"]), 2),
                    "ref_dev_after": round(float(after["ref_dev"]), 1),
                    "train_miss_med": round(float(np.median(train_miss[-5:])), 1),
                }
            )
            del circuit
            if progress:
                progress({"step": i + 1, "total": len(sizes), "size": int(n)})
    return {
        "kind": kind,
        "episodes": int(episodes),
        "rows": rows,
        "seconds": round(time.time() - t0, 1),
    }


# ── скоростная серия: held-out матрица v_m × v_t × аспект × профиль скорости ──


def _speed_cell_scenario(vm: float, vt: float, aspect: str, mode: str, maneuver: str, cell_i: int) -> Scenario:
    """Ячейка held-out матрицы: seed из PROTOCOL['seeds']['test_base'] — test-зёрна
    не пересекаются с train (11+ep*7) и validation (каноническое трио)."""
    period = 6.0
    return Scenario(
        aspect=aspect,  # type: ignore[arg-type]
        v_m=vm,
        v_t=vt,
        range_m=8000.0,
        off_axis_m=420.0,
        n_max=30.0,
        n_target=0.0 if maneuver == TRANSFER_STRAIGHT else 3.0,
        maneuver=maneuver,  # type: ignore[arg-type]
        pn_n=4.0,
        dt=0.02,
        t_max=16.0,
        fov_deg=14.0,
        kill_radius_m=45.0,
        mode="bio",
        target_speed_mode=mode,  # type: ignore[arg-type]
        target_longitudinal_g=1.0,
        target_speed_min=float(max(PROTOCOL["target_speed_bounds"]["abs_min"], vt * 0.55)),
        target_speed_max=float(min(PROTOCOL["target_speed_bounds"]["abs_max"], vt * 1.5)),
        target_speed_period_s=period,
        target_speed_phase=0.25 * period,
        circuit_gain=1.0,
        seed=PROTOCOL["seeds"]["test_base"] + cell_i,
    )


def _run_cell(circuit, sc: Scenario) -> dict[str, Any]:
    """Прогон ячейки: перехват, CPA, усилие, N_eff, насыщение, захват.
    CPA — единственное значение res.cpa_m (непрерывный минимум за полёт);
    квартили CPA не вычисляются внутри ячейки (при stride=100_000 это
    бессмысленно — 1–2 кадра). Распределение CPA строится по эпизодам."""
    res = collect(sc, stride=100_000, circuit=circuit)
    return {
        "hit": bool(res.hit),
        "cpa_m": round(float(res.cpa_m), 1),
        "impact_angle_deg": round(float(res.impact_angle_deg), 1) if res.impact_angle_deg is not None else None,  # η — угол встречи
        "t_hit": round(float(res.t_hit), 2) if res.t_hit is not None else None,
        "effort_gs": round(float(res.n_int), 1),
        "n_eff_median": round(float(res.n_eff_median), 3) if res.n_eff_median is not None else None,
        "n_eff_iqr": (
            round(float(res.n_eff_q75) - float(res.n_eff_q25), 3)
            if res.n_eff_q25 is not None and res.n_eff_q75 is not None
            else None
        ),
        "corr_n_eff_rho": round(float(res.corr_n_eff_rho), 3) if res.corr_n_eff_rho is not None else None,
        "sat_frac": round(float(res.sat_frac), 3),
        "lock_frac": round(float(res.lock_fraction), 3),
    }


def speed_matrix(
    kinds: tuple[str, ...] = ("connectome",),
    aspects: tuple[str, ...] = TEST_ASPECTS,
    speed_modes: tuple[str, ...] = TEST_SPEED_MODES,
    maneuvers: tuple[str, ...] = SPEED_MATRIX_MANEUVERS,
    vm_grid: tuple[float, ...] = TEST_VM_GRID,
    vt_grid: tuple[float, ...] = TEST_VT_GRID,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Held-out проверочная матрица: для каждого мозга и каждой ячейки
    «v_m × v_t × аспект × профиль скорости × манёвр» — метрики прогона.
    По умолчанию 3×3×3×4×2 = 216 ячеек на мозг; для быстрых оценок сужайте сетки."""
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    cell_i = 0
    total = len(kinds) * len(vm_grid) * len(vt_grid) * len(aspects) * len(speed_modes) * len(maneuvers)
    done = 0
    for kind in kinds:
        circuit = get_live_circuit(kind)
        for vm in vm_grid:
            for vt in vt_grid:
                for aspect in aspects:
                    for mode in speed_modes:
                        for maneuver in maneuvers:
                            cell_i += 1
                            sc = _speed_cell_scenario(vm, vt, aspect, mode, maneuver, cell_i)
                            rows.append(
                                {
                                    "kind": kind,
                                    "v_m": vm,
                                    "v_t": vt,
                                    "aspect": aspect,
                                    "speed_mode": mode,
                                    "maneuver": maneuver,
                                    **_run_cell(circuit, sc),
                                }
                            )
                            done += 1
                            if progress:
                                progress({"cell": done, "total": total, "kind": kind, "v_m": vm, "v_t": vt, "mode": mode})
    return {
        "feature_schema_version": PROTOCOL["feature_schema_version"],
        "protocol_seeds": PROTOCOL["seeds"],
        "rows": rows,
        "seconds": round(time.time() - t0, 1),
    }


def get_live_circuit(kind: str):
    """Экземпляр мозга с текущими сохранёнными весами (без подмены живого стенда
    там, где это важно, — научные прогоны сбрасывают только состояние)."""
    if kind == "connectome":
        circuit = ConnectomeCircuit()
        circuit.load()
    elif kind == "full":
        circuit = FlyCircuit(kind="full")
        circuit.load()
    else:
        circuit = FlyCircuit(kind="stub")
        circuit.load()
    circuit.reset()
    return circuit


# ── отчёт «что выучил мозг»: адаптивность N_eff ───────────────────────────────


def neff_report(
    kind: str = "connectome",
    aspects: tuple[str, ...] = TEST_ASPECTS,
    vt_values: tuple[float, ...] = (150.0, 260.0, 360.0),
    vm_values: tuple[float, ...] = (650.0, 950.0),
    speed_modes: tuple[str, ...] = ("constant", "accelerate", "sine"),
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Анализ выученного закона по N_eff:
    1) постоянен ли N_eff (IQR/медиана);
    2) зависит ли от rho (корреляция);
    3) растёт ли усиление при уменьшении t_go (наклон N_eff по 1/t_go);
    4) снижается ли N_eff при насыщении;
    5) меняется ли по v_m и v_t;
    6) обобщается ли на невиданные профили скорости (sine — не в проверочной матрице).
    Все прогоны — с истинной геометрией только в диагностике (не в управлении)."""
    t0 = time.time()
    per_run: list[dict[str, Any]] = []
    total = len(aspects) * len(vt_values) * len(vm_values) * len(speed_modes)
    done = 0
    circuit = get_live_circuit(kind)
    for vm in vm_values:
        for vt in vt_values:
            for aspect in aspects:
                for mode in speed_modes:
                    sc = _speed_cell_scenario(vm, vt, aspect, mode, TRANSFER_STRAIGHT, 500 + done)
                    res = collect(sc, stride=100_000, circuit=circuit)
                    per_run.append(
                        {
                            "v_m": vm,
                            "v_t": vt,
                            "aspect": aspect,
                            "speed_mode": mode,
                            "hit": bool(res.hit),
                            "cpa_m": round(float(res.cpa_m), 1),
                            "n_eff_median": round(float(res.n_eff_median), 3) if res.n_eff_median is not None else None,
                            "n_eff_q25": round(float(res.n_eff_q25), 3) if res.n_eff_q25 is not None else None,
                            "n_eff_q75": round(float(res.n_eff_q75), 3) if res.n_eff_q75 is not None else None,
                            "n_eff_valid_frac": round(float(res.n_eff_valid_frac), 3),
                            "sat_frac": round(float(res.sat_frac), 3),
                            "corr_n_eff_rho": round(float(res.corr_n_eff_rho), 3) if res.corr_n_eff_rho is not None else None,
                        }
                    )
                    done += 1
                    if progress:
                        progress({"run": done, "total": total, "vm": vm, "vt": vt, "mode": mode})

    # сводный анализ по валидным отсчётам всех прогонов
    valid = [r for r in per_run if r["n_eff_median"] is not None]
    meds = [float(r["n_eff_median"]) for r in valid]
    median_of_med = float(np.median(meds)) if meds else None
    iqr_of_med = (
        float(np.percentile(meds, 75) - np.percentile(meds, 25)) if len(meds) >= 4 else None
    )
    corrs = [float(r["corr_n_eff_rho"]) for r in valid if r["corr_n_eff_rho"] is not None]

    def _group_medians(key: str) -> dict[str, float | None]:
        groups: dict[str, list[float]] = {}
        for r in valid:
            groups.setdefault(str(r[key]), []).append(float(r["n_eff_median"]))  # type: ignore[arg-type]
        return {k: round(float(np.median(v)), 3) for k, v in groups.items()}

    sat_rows = [r for r in valid if r["sat_frac"] > 0.05]
    nosat_rows = [r for r in valid if r["sat_frac"] <= 0.05]
    med_sat = float(np.median([r["n_eff_median"] for r in sat_rows])) if sat_rows else None
    med_nosat = float(np.median([r["n_eff_median"] for r in nosat_rows])) if nosat_rows else None

    by_mode = _group_medians("speed_mode")
    unseen = [m for m in by_mode if m not in TEST_SPEED_MODES]
    return {
        "kind": kind,
        "feature_schema_version": PROTOCOL["feature_schema_version"],
        "n_runs": len(per_run),
        "per_run": per_run,
        "n_eff_median_of_medians": round(median_of_med, 3) if median_of_med is not None else None,
        "n_eff_iqr_of_medians": round(iqr_of_med, 3) if iqr_of_med is not None else None,
        "constancy_note": (
            "N_eff считается постоянным, если IQR медиан ≲ 5% медианы"
        ),
        "is_constant": (
            bool(median_of_med is not None and iqr_of_med is not None and median_of_med > 0 and iqr_of_med / max(abs(median_of_med), 1e-9) <= 0.05)
        ),
        "corr_n_eff_rho_median": round(float(np.median(corrs)), 3) if corrs else None,
        "rho_dependence": (
            "есть" if corrs and abs(float(np.median(corrs))) >= 0.3 else "слабая/нет"
        ),
        "n_eff_median_saturated": round(med_sat, 3) if med_sat is not None else None,
        "n_eff_median_unsaturated": round(med_nosat, 3) if med_nosat is not None else None,
        "by_v_m": _group_medians("v_m"),
        "by_v_t": _group_medians("v_t"),
        "by_speed_mode": by_mode,
        "unseen_modes": unseen,
        "generalizes_to_unseen": (
            bool(unseen) and all(abs(by_mode[m] - median_of_med) <= 0.2 * max(abs(median_of_med), 1e-9) for m in unseen if by_mode[m] is not None)
            if median_of_med is not None
            else None
        ),
        "seconds": round(time.time() - t0, 1),
    }


# ── абляции фазовых признаков theta/rho ───────────────────────────────────────

# индексы признаков v2: 4 = theta, 6 = rho
FEAT_THETA = 4
FEAT_RHO = 6


ABLATION_MODES = (
    "none", "no_theta", "no_rho", "no_both",
    "rho_fixed",
    "rho_episode_perm",   # rho из ДРУГОГО эпизода (интерполяция по нормированной фазе)
    "rho_time_shuffle",   # случайная перестановка rho внутри эпизода
    "rho_phase_shift",    # циклический сдвиг на половину длины
    "rho_reverse",        # обратный порядок
)


def _collect_full_rho(circuit, sc: Scenario) -> tuple[dict[str, Any], list[float]]:
    """Полный прогон с записью rho на каждом шаге (для честных абляций).
    Возвращает (результат эпизода, rho-последовательность)."""
    rho_trace: list[float] = []
    out = _rollout_update(circuit, sc, lr=0.0, rho_trace=rho_trace)
    return out, rho_trace


def _resample_by_phase(seq: list[float], n: int) -> np.ndarray:
    """Пересэмплировать временной ряд на n отсчётов по НОРМИРОВАННОЙ ФАЗЕ эпизода.

    donor-эпизоды разной длины сравнимы только по фазе (t/t_end), не по номеру
    кадра: без интерполяции хвост donor-ряда повторялся бы почти весь полёт —
    это не временная перестановка, а подмена признака константой.
    Используется ТОЛЬКО офлайн при подготовке абляционной последовательности;
    истинное время в BIO не поступает."""
    src = np.asarray(seq, dtype=np.float64).reshape(-1)
    if src.size == 0:
        return np.zeros(n)
    if src.size == 1:
        return np.full(n, float(src[0]))
    x_src = np.linspace(0.0, 1.0, src.size)
    x_dst = np.linspace(0.0, 1.0, max(int(n), 2))
    return np.interp(x_dst, x_src, src)


def _max_steps(sc: Scenario, t_cap: float = 16.0) -> int:
    """Верхняя оценка числа шагов эпизода (для длины абляционной последовательности):
    эпизод может кончиться раньше (перехват/проход), но не позже."""
    return int(min(sc.t_max, t_cap) / max(sc.dt, 1e-6)) + 2


def _apply_rho_ablation(
    circuit, mode: str, rho_sequences: list[list[float]],
    cell_i: int, episode_seed: int, n_steps: int,
) -> None:
    """Установить feat_patch для заданного режима абляции rho.

    Все последовательности пересэмплированы по нормированной фазе эпизода на
    n_steps отсчётов: донор другой длины больше НЕ «размазывает» последнее
    значение на весь полёт."""
    if mode == "none":
        circuit.feat_patch = None
    elif mode == "no_theta":
        circuit.feat_patch = {FEAT_THETA: 0.0}
    elif mode == "no_rho":
        circuit.feat_patch = {FEAT_RHO: 0.0}
    elif mode == "no_both":
        circuit.feat_patch = {FEAT_THETA: 0.0, FEAT_RHO: 0.0}
    elif mode == "rho_fixed":
        circuit.feat_patch = {FEAT_RHO: 0.8}
    elif mode == "rho_episode_perm":
        # rho из ДРУГОГО эпизода: интерполяция по нормированной фазе
        if len(rho_sequences) >= 2:
            donor_idx = (cell_i + 1) % len(rho_sequences)
            donor = _resample_by_phase(rho_sequences[donor_idx], n_steps)
            circuit.feat_patch = {FEAT_RHO: donor}
        else:
            circuit.feat_patch = {FEAT_RHO: 0.8}  # fallback
    elif mode == "rho_time_shuffle":
        if len(rho_sequences) >= 1:
            donor = _resample_by_phase(rho_sequences[min(cell_i, len(rho_sequences) - 1)], n_steps)
            rng_ab = np.random.default_rng(episode_seed + 999)
            rng_ab.shuffle(donor)
            circuit.feat_patch = {FEAT_RHO: donor}
        else:
            circuit.feat_patch = {FEAT_RHO: 0.8}
    elif mode == "rho_phase_shift":
        if len(rho_sequences) >= 1:
            src = _resample_by_phase(rho_sequences[min(cell_i, len(rho_sequences) - 1)], n_steps).tolist()
            shift = len(src) // 2
            shifted = src[shift:] + src[:shift]
            circuit.feat_patch = {FEAT_RHO: shifted}
        else:
            circuit.feat_patch = {FEAT_RHO: 0.8}
    elif mode == "rho_reverse":
        if len(rho_sequences) >= 1:
            src = _resample_by_phase(rho_sequences[min(cell_i, len(rho_sequences) - 1)], n_steps)
            circuit.feat_patch = {FEAT_RHO: list(reversed(src))}
        else:
            circuit.feat_patch = {FEAT_RHO: 0.8}


def feature_ablation(
    kind: str = "connectome",
    aspects: tuple[str, ...] = TEST_ASPECTS,
    vt_values: tuple[float, ...] = (150.0, 260.0, 360.0),
    modes: tuple[str, ...] | None = None,
    speed_modes: tuple[str, ...] = ("constant", "sine"),
    circuit=None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Абляции theta/rho на уровне признаков контура (feat_patch: столбец
    обнуляется, фиксируется или переставляется между эпизодами). Сравниваются
    hit rate, CPA, усилие и медианный N_eff по скоростям. Сравнение с baseline
    показывает, что именно фазовые признаки добавили к пеленговой навигации.
    circuit — внешний экземпляр (например, v2-обученная копия); None — живой
    мозг вида kind с сохранёнными весами.

    Rho-последовательности собираются на каждом шаге через _rollout_update
    (исправлено: ранее использовался collect(stride=100_000), дававший 1–2 кадра)."""
    if modes is None:
        modes = ABLATION_MODES
    t0 = time.time()
    circuit = circuit if circuit is not None else get_live_circuit(kind)
    saved_patch = circuit.feat_patch
    rho_sequences: list[list[float]] = []  # полные rho-последовательности baseline
    results: dict[str, Any] = {"kind": kind, "modes": {}, "cells": []}

    # Фаза 1: baseline — собрать rho-последовательности
    if "none" in modes:
        cell_i = 0
        rows_out: list[dict[str, Any]] = []
        for vt in vt_values:
            for aspect in aspects:
                for sp_mode in speed_modes:
                    cell_i += 1
                    sc = _speed_cell_scenario(780.0, vt, aspect, sp_mode, TRANSFER_STRAIGHT, 700 + cell_i)
                    circuit.feat_patch = None
                    circuit.reset()
                    out, rho_seq = _collect_full_rho(circuit, sc)
                    rho_sequences.append(rho_seq)
                    rows_out.append({
                        "v_t": vt, "aspect": aspect, "speed_mode": sp_mode,
                        "hit": out["hit"],
                        "cpa_m": round(float(out["cpa_m"]), 1),
                        "effort_gs": round(float(out.get("control_effort_gs", out.get("n_avg", 0.0))), 1),
                        "n_eff_median": None,  # N_eff не считается в _rollout_update
                    })
        hits = [r["hit"] for r in rows_out]
        results["modes"]["none"] = {
            "hit_rate": round(sum(hits) / max(len(hits), 1), 3),
            "cpa_median": round(float(np.median([r["cpa_m"] for r in rows_out])), 1),
            "effort_median": round(float(np.median([r["effort_gs"] for r in rows_out])), 1),
            "n_eff_median": None,
            "cells": rows_out,
        }
        results["cells"].extend(rows_out)
        if progress:
            progress({"mode": "none", "cells": len(rows_out)})

    # Фаза 2: абляции (кроме "none")
    for mode in modes:
        if mode == "none":
            continue
        rows_out = []
        cell_i = 0
        for vt in vt_values:
            for aspect in aspects:
                for sp_mode in speed_modes:
                    cell_i += 1
                    sc = _speed_cell_scenario(780.0, vt, aspect, sp_mode, TRANSFER_STRAIGHT, 700 + cell_i)
                    _apply_rho_ablation(circuit, mode, rho_sequences, cell_i - 1, sc.seed, _max_steps(sc))
                    circuit.reset()
                    res = collect(sc, stride=100_000, circuit=circuit)
                    rows_out.append({
                        "v_t": vt, "aspect": aspect, "speed_mode": sp_mode,
                        "hit": bool(res.hit),
                        "cpa_m": round(float(res.cpa_m), 1),
                        "effort_gs": round(float(res.n_int), 1),
                        "n_eff_median": round(float(res.n_eff_median), 3) if res.n_eff_median is not None else None,
                    })
        hits = [r["hit"] for r in rows_out]
        results["modes"][mode] = {
            "hit_rate": round(sum(hits) / max(len(hits), 1), 3),
            "cpa_median": round(float(np.median([r["cpa_m"] for r in rows_out])), 1),
            "effort_median": round(float(np.median([r["effort_gs"] for r in rows_out])), 1),
            "n_eff_median": (
                round(float(np.median([r["n_eff_median"] for r in rows_out if r["n_eff_median"] is not None])), 3)
                if any(r["n_eff_median"] is not None for r in rows_out)
                else None
            ),
            "cells": rows_out,
        }
        results["cells"].extend(rows_out)
        if progress:
            progress({"mode": mode, "cells": len(rows_out)})
    circuit.feat_patch = saved_patch
    circuit.reset()
    return {
        "kind": kind,
        "feature_schema_version": PROTOCOL["feature_schema_version"],
        "modes": results["modes"],
        "rho_sequence_lengths": [len(s) for s in rho_sequences[:5]],  # diagnostic
        "seconds": round(time.time() - t0, 1),
    }



def _wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Точный биномиальный 95 % ДИ (Уилсон) для доли k/n."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    den = 1.0 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * float(np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den
    return (max(0.0, ctr - half), min(1.0, ctr + half))


def monte_carlo(sc: Scenario, n_runs: int = 200, seed_start: int = 1000) -> dict[str, Any]:
    """Monte-Carlo рассеивание: n_runs независимых прогонов одного сценария с
    seed = seed_start, seed_start+1, … (каждый прогон тянет свои реализации шума
    измерений, срыва сопровождения и отказов сетчатки).

    Ответ Weapons Handbook: статистика точки прицела при возмущающих факторах.
    Возвращает: вероятность перехвата p_hit (доля попаданий в сферу срабатывания)
    с точным биномиальным 95 % ДИ (Уилсон), разброс R_min (среднее, СКО, медиана,
    P90), R_95 и CEP_50 по всем выстрелам и отдельно по промахам.
    При нулевой стохастике все прогоны детерминированно совпадают — СКО = 0."""
    from dataclasses import replace

    t0 = time.time()
    n_runs = max(1, int(n_runs))
    cpa: list[float] = []
    hits: list[bool] = []
    h0s: list[float] = []
    per_run: list[dict[str, Any]] = []
    for i in range(n_runs):
        seed = int(seed_start) + i
        r = collect(replace(sc, seed=seed), stride=8)
        cpa.append(float(r.cpa_m))
        hits.append(bool(r.hit))
        if r.h0_m is not None:
            h0s.append(float(r.h0_m))
        per_run.append({
            "seed": seed,
            "r_min_m": round(float(r.cpa_m), 3),
            "hit": bool(r.hit),
            "reason": r.reason,
            "h0_m": round(float(r.h0_m), 3) if r.h0_m is not None else None,
            "impact_angle_deg": round(float(r.impact_angle_deg), 1) if r.impact_angle_deg is not None else None,
        })
    arr = np.asarray(cpa, dtype=float)
    miss_arr = arr[np.asarray([not h for h in hits], dtype=bool)]
    k = int(sum(hits))
    lo, hi = _wilson_ci(k, n_runs)

    def _q(a: np.ndarray, p: float) -> float | None:
        return round(float(np.percentile(a, p)), 3) if a.size else None

    return {
        "n_runs": n_runs,
        "seed_start": int(seed_start),
        "model": sc.model,
        "trigger_range_m": float(sc.kill_radius_m),
        "p_hit": round(k / n_runs, 4),
        "p_hit_ci95": [round(lo, 4), round(hi, 4)],
        "r_min_mean_m": round(float(arr.mean()), 3),
        "r_min_std_m": round(float(arr.std(ddof=1)), 3) if n_runs >= 2 else None,
        "r_min_median_m": _q(arr, 50),
        "r_min_p90_m": _q(arr, 90),
        "r_95_m": _q(arr, 95),  # радиус, накрывающий 95 % выстрелов
        "cep50_m": _q(arr, 50),  # CEP_50 — медианный промах (50 % выстрелов)
        "r_min_miss_mean_m": round(float(miss_arr.mean()), 3) if miss_arr.size else None,
        "r_min_miss_std_m": round(float(miss_arr.std(ddof=1)), 3) if miss_arr.size >= 2 else None,
        "h0_mean_m": round(float(np.mean(h0s)), 3) if h0s else None,
        # «детерминированно» = разброс не виден даже на допуске 1 мкм (сравнение == 0.0
        # ломается на шуме младших бит: 44.986000000000006 vs 44.98599999999999)
        "deterministic": bool(n_runs >= 2 and float(arr.std(ddof=1)) < 1e-6 * max(1.0, abs(float(arr.mean())))),
        "per_run": per_run,
        "seconds": round(time.time() - t0, 1),
    }


# ── зона неубегаемого перехвата (ЗНП / no-escape zone) ────────────────────────

# сетки по умолчанию: дальность пуска 2…12 км, перегрузка цели 0…20 g
CZ_RANGES_M: tuple[float, ...] = (2000.0, 4000.0, 6000.0, 8000.0, 10000.0, 12000.0)
CZ_TARGET_GS: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 15.0, 20.0)


def capture_zone(
    sc: Scenario,
    ranges_m: tuple[float, ...] = CZ_RANGES_M,
    target_gs: tuple[float, ...] = CZ_TARGET_GS,
    refine: bool = True,
    refine_tol_m: float = 50.0,
    refine_iters: int = 14,
    progress: Progress | None = None,
    n_runs: int = 1,
    seed_start: int = 1000,
) -> dict[str, Any]:
    """Зона неубегаемого перехвата: граница «успеет / не успеет» в плоскости
    «дальность пуска × нормальная перегрузка цели».

    Классическая задача теории перехвата (Дмитрий, Ровинский, Юрьев «Теория
    оптимального управления полётом летательных аппаратов»: задача уклонения;
    в учебниках по прицельности — зона гарантированного поражения): цель
    уходит от перехватчика с ПОСТОЯННОЙ нормальной перегрузкой, и при каждой
    перегрузке есть предельная дальность пуска, за которой перехват невозможен
    уже физически. Здесь границы находятся численно: сетка прогонов + бисекция
    скобок на обеих сторонах зоны — дальней «попал → промах» и ближней
    «промах → попал» (на малых дальностях ракета не успевает развернуться).

    Для каждой строки с n_target > 0 скан использует манёвр «turn» (постоянная
    перегрузка — именно она задаёт классическую задачу уклонения); при
    n_target = 0 манёвр не важен (цель прямолинейна). Закон/модель/шумы —
    из переданного сценария; при mode='pn' без шума прогон детерминирован,
    и сетка — чистая геометрия перехвата, а не статистика.

    Асимптотика для проверки (учебный результат по ПН, Зархан; [Гусев 1996]
    гл. 4): при постоянной перегрузке цели установившаяся команда ракете
    a_уст = N/(N−1)·a_цели — если она превышает располагаемую, цель НЕУБЕЖАЕМА
    на любой дальности; если меньше — перехват возможен на всей глубине зоны.
    Граница зоны по n_target лежит около n_цели* = (N−1)/N·n_max.

    Честно: зона захвата цели с ПОСТОЯННОЙ перегрузкой — классически НЕ отрезок
    (Дмитрий, Ровинский, Юрьев: область «успеет/не успеет» при маневре цели
    имеет долепестную структуру — зависит от фазы разворота в момент прохода).
    Поэтому монотонность «дальше = хуже» не предполагается: для строки считается
    число смен исхода по сетке (monotonic_violations) и число связных блоков
    перехвата (hit_blocks). Границы бисекцией строятся только там, где зона —
    один связный отрезок (hit_blocks == 1); у него бывает и ближняя, и дальняя
    граница: ближняя — ракета не успевает развернуться на упреждение (время
    разворота ~√(2·y_смещ/a_расп), классическая минимальная дальность применения;
    [Гусев 1996] гл. 4, Зархан), дальняя — цель уходит. Блоков перехвата несколько
    — структура долепестная, строка получает статус lobed и численных границ не
    получает. Статусы строк: all_hit (все узлы сетки — перехват), no_hit (ни
    одного), bracketed (один связный блок: fields boundary_m / inner_boundary_m
    по факту границы внутри сетки), lobed (несколько блоков). Промах по истечении
    окна t_max помечается в ячейке time_limited=true — это не физический предел
    зоны, а граница окна прогона: увеличивайте t_max, если тайм-ауты левее
    физической границы; у внешней границы это отдельно честно помечено
    флагом boundary_time_limited.

    Вероятностный режим (n_runs > 1): каждая ячейка сетки — серия из n_runs
    независимых прогонов со seed = seed_start + i (та же стохастика движка,
    что в monte_carlo: шум измерений, срывы сопровождения, отказы сетчатки).
    Ячейка несёт p_hit с биномиальным 95 % ДИ (Уилсон) и медиану R_min;
    бинарное поле hit — большинство серии (p_hit ≥ 0.5), им же определяются
    блоки и статусы строк. Бисекция границ в этом режиме не запускается
    (флаг refine_skipped): скобка «попал→промах» на majority-исходах была бы
    статистикой из одного выстрела. Если все прогоны каждой ячейки совпали
    побитно (СКО R_min в ячейке ниже 1e-6·среднего) — series_deterministic=true:
    в сценарии нет стохастики либо закон не читает измерительный канал."""
    t0 = time.time()
    n_runs = max(1, int(n_runs))
    ranges_m = tuple(float(r) for r in ranges_m)
    target_gs = tuple(float(g) for g in target_gs)
    t_cap = float(sc.t_max)
    if n_runs > 1:
        refine = False  # бисекция majority-скобки = статистика из одного выстрела

    def _run(n_t: float, r_m: float, seed: int | None = None) -> tuple[bool, float, float, bool]:
        s = replace(sc, range_m=r_m, n_target=n_t,
                    maneuver="turn" if n_t > 0.0 else sc.maneuver)
        if seed is not None:
            s = replace(s, seed=seed)
        res = collect(s, stride=100_000)
        return (bool(res.hit), float(res.cpa_m), float(res.t_end),
                (not res.hit) and res.t_end >= t_cap - 1e-9)

    def _edge(n_t: float, r_miss: float, r_hit: float) -> tuple[float, float, float]:
        """Граница перехвата между узлом-промахом и соседним узлом-перехватом
        бисекцией (тот же допуск refine_tol_m и то же число итераций для обеих
        сторон зоны). Возвращает (граница, левый край скобки, правый край)."""
        lo, hi = sorted((float(r_miss), float(r_hit)))
        hit_is_low = r_hit < r_miss  # перехват снизу => скобка внешняя (дальше = хуже)
        for _ in range(int(refine_iters)):
            if hi - lo <= refine_tol_m:
                break
            mid = 0.5 * (lo + hi)
            h, _, _, _ = _run(n_t, mid)
            if hit_is_low:
                lo, hi = (mid, hi) if h else (lo, mid)
            else:
                lo, hi = (lo, mid) if h else (mid, hi)
        return round(0.5 * (lo + hi), 1), round(lo, 1), round(hi, 1)

    rows: list[dict[str, Any]] = []
    degenerate_cells = 0
    done = 0
    total = len(target_gs)
    for n_t in target_gs:
        cells: list[dict[str, Any]] = []
        for r_m in ranges_m:
            if n_runs == 1:
                hit, cpa, t_end, tlim = _run(n_t, r_m)
                cells.append({"range_m": round(r_m, 1), "hit": hit,
                              "r_min_m": round(cpa, 1),
                              "t_end_s": round(t_end, 2), "time_limited": tlim})
                continue
            series = [_run(n_t, r_m, seed=int(seed_start) + i) for i in range(n_runs)]
            cpas = [s[1] for s in series]
            k_hit = sum(1 for s in series if s[0])
            miss_tlim = [s[3] for s in series if not s[0]]
            arr = np.asarray(cpas, dtype=float)
            lo, hi = _wilson_ci(k_hit, n_runs)
            med = float(np.median(arr))
            deg = float(arr.std(ddof=1)) < 1e-6 * max(1.0, abs(float(arr.mean())))
            degenerate_cells += int(deg)
            cells.append({
                "range_m": round(r_m, 1),
                "hit": k_hit * 2 >= n_runs,  # большинство серии
                "p_hit": round(k_hit / n_runs, 4),
                "p_hit_ci95": [round(lo, 4), round(hi, 4)],
                "r_min_m": round(med, 1),
                "median_cpa_m": round(med, 3),
                "t_end_s": round(max(s[2] for s in series), 2),
                "time_limited": bool(miss_tlim) and sum(miss_tlim) * 2 >= len(miss_tlim),
                "deterministic_cell": deg,
            })
        hits_seq = [c["hit"] for c in cells]
        # смены исхода по возрастанию дальности (честь: монотонность не гарантирована)
        flips = sum(1 for i in range(1, len(hits_seq)) if hits_seq[i] != hits_seq[i - 1])
        hit_ranges = [c["range_m"] for c in cells if c["hit"]]
        miss_ranges = [c["range_m"] for c in cells if not c["hit"]]
        # связные блоки перехватов: зона как ОТРЕЗОК дальностей — это один блок
        # (у него ближняя и/или дальняя граница внутри сетки); блоков несколько —
        # долепестная структура, границ не строим
        blocks: list[tuple[int, int]] = []
        start: int | None = None
        for i, h in enumerate(hits_seq):
            if h and start is None:
                start = i
            elif not h and start is not None:
                blocks.append((start, i - 1))
                start = None
        if start is not None:
            blocks.append((start, len(hits_seq) - 1))
        row: dict[str, Any] = {
            "n_target_g": n_t,
            "cells": cells,
            "hit_frac": round(sum(hits_seq) / max(len(hits_seq), 1), 3),
            "monotonic_violations": int(max(flips - 1, 0)) if hits_seq else 0,
            "hit_blocks": len(blocks),
        }
        if not hit_ranges:
            row["status"] = "no_hit"
        elif not miss_ranges:
            row["status"] = "all_hit"
            row["r_min_range_m"] = min(hit_ranges)
        elif len(blocks) == 1:
            (b0, b1) = blocks[0]
            row["status"] = "bracketed"
            row["r_last_hit_m"] = max(hit_ranges)
            row["r_first_miss_m"] = min(miss_ranges)
            if refine and b1 + 1 < len(cells):  # дальняя граница: попал → промах
                mid, lo, hi = _edge(n_t, cells[b1 + 1]["range_m"], cells[b1]["range_m"])
                row["boundary_m"] = mid
                row["boundary_bracket_m"] = [lo, hi]
                # честно: промах у внешней границы может быть отсечкой окна прогона
                row["boundary_time_limited"] = bool(cells[b1 + 1]["time_limited"])
            if refine and b0 - 1 >= 0:  # ближняя граница: промах → попал
                mid, lo, hi = _edge(n_t, cells[b0 - 1]["range_m"], cells[b0]["range_m"])
                row["inner_boundary_m"] = mid
                row["inner_boundary_bracket_m"] = [lo, hi]
        else:
            row["status"] = "lobed"
            row["r_last_hit_m"] = max(hit_ranges)
            row["r_first_miss_m"] = min(miss_ranges)
        rows.append(row)
        done += 1
        if progress:
            progress({"row": done, "total": total, "n_target_g": n_t})

    out = {
        "law": sc.law,
        "mode": sc.mode,
        "model": sc.model,
        "aspect": sc.aspect,
        "v_m": float(sc.v_m),
        "v_t": float(sc.v_t),
        "t_max_s": float(sc.t_max),
        "pn_n": float(sc.pn_n),
        "n_max_g": float(sc.n_max),
        "trigger_range_m": float(sc.kill_radius_m),
        "ranges_m": list(ranges_m),
        "target_gs": list(target_gs),
        "rows": rows,
        "note": ("перегрузка цели работает только при манёвре с постоянной "
                 "нормальной составляющей; для строк n_target > 0 скан "
                 "использует «turn», иначе (maneuver='straight', n_target=0) "
                 "цель прямолинейна"),
        "seconds": round(time.time() - t0, 1),
    }
    if n_runs > 1:
        out["n_runs"] = n_runs
        out["seed_start"] = int(seed_start)
        out["refine_skipped"] = True
        out["series_deterministic"] = bool(
            ranges_m and target_gs and degenerate_cells == len(ranges_m) * len(target_gs))
        out["note"] += ("; вероятность: p_hit — доля серии из n_runs прогонов с seed "
                        "от seed_start, hit строки — большинство серии (p_hit ≥ 0.5); "
                        "бисекция границ в этом режиме не выполняется")
    return out
