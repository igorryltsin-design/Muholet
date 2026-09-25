"""HTTP + WebSocket стенд."""

from __future__ import annotations

import asyncio
import base64
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from navedenie.brain_store import get_circuit, replace as brain_replace, set_circuit, status as brain_status
from navedenie.circuit import FEAT_DIM, FEATURE_SCHEMA_VERSION, ConnectomeCircuit, FlyCircuit
from navedenie.engine import collect, frame_to_dict
from navedenie.formula import brain_formula
from navedenie.sim import Scenario
from navedenie.swarm import evolve, evaluate_population, fly_from_json, init_population
from navedenie.train import train as train_brain
from navedenie.train import evolve_finish as train_evolve_finish_brain
from navedenie.train import evolve_start as train_evolve_start_brain
from navedenie.train import evolve_step as train_evolve_step_brain
from navedenie.train import train_finish as train_finish_brain
from navedenie.train import train_start as train_start_brain
from navedenie.train import train_step as train_step_brain

import numpy as np

app = FastAPI(title="Наведение")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_VERSION = "2.0.0"  # версия стенда: схема признаков v2 + N_eff + профили скорости


class ScenarioIn(BaseModel):
    aspect: str = "head-on"  # head-on | beam | tail-chase | free (свободная расстановка, см. free_* поля)
    v_m: float = 780
    v_t: float = 260
    range_m: float = 8000
    off_axis_m: float = 400
    free_tx: float = 8000.0
    free_ty: float = 0.0
    free_talt: float = 4000.0
    free_mhdg: float = 0.0
    free_mclimb: float = 0.0
    free_thdg: float = 180.0
    free_tclimb: float = 0.0
    alt_m: float = 4000  # высота пуска над уровнем моря, м (в физ. режиме с atmos задаёт ρ(H))
    n_max: float = 30
    n_target: float = 0
    maneuver: str = "straight"
    pn_n: float = 4
    pn_sched_n0: float = 3.0
    pn_sched_k_rho: float = 0.8
    pn_sched_n_min: float = 2.0
    pn_sched_n_max: float = 6.0
    target_speed_mode: str = "constant"
    target_longitudinal_g: float = 1.0
    target_speed_min: float = 120.0
    target_speed_max: float = 520.0
    target_speed_period_s: float = 6.0
    target_speed_phase: float = 0.0
    dt: float = 0.005
    t_max: float = 35
    fov_deg: float = 12
    bio_fov_deg: float = 165
    seeker_delay_s: float = 0.02
    noise_az_deg: float = 0.0
    noise_range_m: float = 0.0
    lock_drop_p: float = 0.0
    seeker_jitter_s: float = 0.0
    retina_death_p: float = 0.0  # постоянная доля «умерших» омматидиев (маска на прогон)
    retina_dropout_p: float = 0.0  # независимый дропаут рецепторов на кадр
    target_brightness: float = 1.0  # фотометрия цели (0.7/1.4 — проверка скрытого дальномера)
    law: str = "pn"
    mode: str = "pn"
    circuit_gain: float = 1.0
    tau_s: float = 0.025
    tau_act_s: float = 0.0  # лаг рулевого привода (1-е звено) в кинематическом контуре, с; 0 — мгновенно
    kill_radius_m: float = 45
    brain: str = "stub"
    # дуэль «муха-ракета против мухи-самолёта»: реактивный уклонист вместо слепого
    # манёвра + боевая жизнь ракеты (законы — navedenie/evader.py)
    duel: bool = False
    evader_law: str = "away"  # away | negpn | cpa_max
    fuse_life_s: float = 30.0
    # модель движения ракеты: kinematic_legacy | point_mass_3dof (цель/призрак всегда кинематические)
    model: str = "kinematic_legacy"
    phys_mass_kg: float = 150.0
    phys_ref_area_m2: float = 0.05
    phys_drag_cx: float = 0.30
    phys_cx_wave: float = 0.0  # прирост ΔCx волнового кризиса на сверхзвуковом плато (0 — Cx постоянен)
    phys_mach_kr: float = 1.0  # число Маха перегиба волнового роста M_кр
    phys_mach_band: float = 0.10  # ширина околозвукового перехода δM
    phys_rho_air: float = 0.736
    phys_atmos: bool = False  # True: ρ(H) по стандартной атмосфере ICAO/US1976 (phys_rho_air игнорируется)
    phys_thrust_n: float = 0.0
    phys_burn_time_s: float = 3.0
    phys_tau_a_s: float = 0.05
    phys_wn_act: float = 0.0  # звено 2-го порядка привода: ω_n, рад/с (0 — прежнее 1-е звено)
    phys_zeta_act: float = 1.0  # звено 2-го порядка привода: ζ
    phys_n_avail_max: float = 30.0
    phys_cn_max: float = 0.0  # граница динамического полёта: Cn_max (0 — константный предел n_avail_max)
    phys_da_dt_max: float = 0.0
    phys_gravity: bool = True


def _sc(body: ScenarioIn) -> Scenario:
    return Scenario(**body.model_dump())


def _data_dir() -> Path:
    """Каталог данных по текущему состоянию модуля circuit (тесты подменяют его
    на временный каталог — файлы не должны уходить в прод-данные)."""
    from navedenie import circuit as _circuit

    return _circuit.DATA_DIR


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"ok": "navedenie", "model_version": MODEL_VERSION, "feature_schema_version": str(FEATURE_SCHEMA_VERSION)}


@app.get("/api/glossary")
def glossary_endpoint() -> dict:
    """Единый канонический словарь терминов и законов (§4) — метаданные для UI,
    подсказок, справки и экспорта. Навигационный коэффициент — N; K только в
    цитатах Гусева с переходом N ≡ K."""
    from navedenie.glossary import as_json

    return as_json()


def _run_answer(result, frames_json: list[dict]) -> dict[str, Any]:
    """Форма ответа прогона: /api/run целиком и финальное сообщение 'done'
    стрима /api/ws/run (там frames_json — пустой список, кадры ушли потоком)."""
    return {
        "model_version": MODEL_VERSION,
        "metrics_version": result.metrics_version,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "miss_m": result.miss_m,  # deprecated alias для cpa_m
        "cpa_m": result.cpa_m,
        "trigger_range_m": result.trigger_range_m,
        "hit": result.hit,
        "t_hit": result.t_hit,
        "t_end": result.t_end,
        "lock_time_s": result.lock_time_s,
        "lock_fraction": result.lock_fraction,
        "n_mean_g": result.n_mean_g,
        "n_int": result.n_int,
        "terminal_zem_m": result.terminal_zem_m,  # LEGACY alias == h_cv_m
        "h_cv_m": result.h_cv_m,        # h_cv — прогноз при неизменных скоростях
        "h0_m": result.h0_m,            # h₀ — промах при нулевой дальнейшей команде
        "end_range_m": result.end_range_m,  # R_end — конечная дистанция
        "impact_angle_deg": result.impact_angle_deg,  # η — угол встречи в момент наибольшего сближения
        "model": result.model,          # kinematic_legacy | point_mass_3dof
        "metrics_version": result.metrics_version,
        "t_guide": result.t_guide,
        "ref_dev_m": result.ref_dev_m,
        "ref_rms_m": result.ref_rms_m,
        "ref_nrms": result.ref_nrms,
        "t_ref": result.t_ref,
        "fov_lock_frac": result.fov_lock_frac,
        "n_peak": result.n_peak,
        "reason": result.reason,
        "n_eff_median": result.n_eff_median,
        "n_eff_q25": result.n_eff_q25,
        "n_eff_q75": result.n_eff_q75,
        "n_eff_min": result.n_eff_min,
        "n_eff_max": result.n_eff_max,
        "n_eff_valid_frac": result.n_eff_valid_frac,
        "sat_frac": result.sat_frac,
        "corr_n_eff_rho": result.corr_n_eff_rho,
        "corr_n_eff_tgo": result.corr_n_eff_tgo,
        "n_eff_count": result.n_eff_count,
        "duel": result.duel,
        "duel_result": result.duel_result,  # missile | evader | None (не дуэль)
        "fuse_expired": result.fuse_expired,  # ракета выдохлась без перехвата
        "t_survived": result.t_survived,  # сколько продержалась цель, с
        "frames": frames_json,
    }


@app.post("/api/run")
def run_once(body: ScenarioIn) -> dict[str, Any]:
    sc = _sc(body)
    result = collect(sc, stride=max(1, int(0.04 / sc.dt)))
    return _run_answer(result, [frame_to_dict(fr) for fr in result.frames])


class DuelIn(BaseModel):
    """Тело «Ринга»: сценарий вложенным объектом (как /api/brain/compare),
    стороны — списки id; пусто — базовые законы против всех уклонистов."""

    scenario: ScenarioIn = ScenarioIn()
    missiles: list[str] = []
    evaders: list[str] = []
    repeats: int = 1
    matrix_dt: float = 0.02  # шаг интегрирования боя: ринг не обязан наследовать точность проигрывания


@app.post("/api/duel")
def duel_endpoint(body: DuelIn) -> dict[str, Any]:
    """Матрица дуэлей «Ринг»: строки — закон/мозг ракеты (`bio:<kind>` для мозга),
    столбцы — закон уклонения цели ('straight' — неманёвренная базовая линия).
    Ячейка: вердикт win, доля перехватов, время жизни цели, R_min, усилие n_int."""
    from dataclasses import replace

    from navedenie.duel import EVADER_COLUMNS, duel_matrix
    from navedenie.pn import LAWS

    sc = replace(_sc(body.scenario), dt=max(float(body.matrix_dt), 0.01))
    missiles = body.missiles or [law for law in LAWS if law in ("pn", "tpn", "apn", "pn_gsn")]
    evaders = body.evaders or list(EVADER_COLUMNS)
    return duel_matrix(
        sc,
        missiles,
        evaders,
        repeats=max(1, min(int(body.repeats), 15)),
        stride=max(1, int(0.04 / sc.dt)),
    )


@app.get("/api/brain")
def brain() -> dict[str, Any]:
    return brain_status()


class BrainImportIn(BaseModel):
    kind: str
    w: list[list[float]]
    trained: bool = True
    gain: float | None = None
    tau: float | None = None
    wiring: dict[str, Any] | None = None
    real_wiring: dict[str, Any] | None = None  # {"format": "circuit_v1.npz (base64)", "data": …}


@app.get("/api/brain/formula")
def brain_formula_endpoint(kind: str = "stub", degree: int | None = None) -> dict[str, Any]:
    """Формула закона наведения из обученного мозга: точный вид + полиномиальный суррогат.

    degree — явная степень полинома (выбор в UI «Формула»); по умолчанию авто по виду мозга.
    """
    return brain_formula(kind, degree)


@app.get("/api/brain/formula/python")
def brain_formula_python_endpoint(kind: str = "stub", degree: int | None = None) -> Any:
    """Исполняемый Python-файл с точной формулой мозга (для скачивания)."""
    from fastapi import Response

    from navedenie.formula import brain_formula_python
    code = brain_formula_python(kind, degree)
    return Response(
        content=code,
        media_type="text/x-python; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="muholet-mozg-{kind}.py"'},
    )


@app.get("/api/brain/formula/c")
def brain_formula_c_endpoint(kind: str = "stub", degree: int | None = None) -> Any:
    """Чистый C99 с точной формулой мозга dn() и полиномом dn_approx() —
    для внешнего симулятора или железа: без numpy, без API стенда."""
    from fastapi import Response

    from navedenie.formula import brain_formula_c
    code = brain_formula_c(kind, degree)
    return Response(
        content=code,
        media_type="text/x-c; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="muholet-mozg-{kind}.c"'},
    )


@app.get("/api/brain/export")
def brain_export(kind: str = "stub") -> dict[str, Any]:
    """Мозг в JSON: веса выхода, усиление, тау и рецепт проводки.

    Для коннектома с реальной проводкой в файл вшивается и сама проводка
    (circuit_v1.npz в base64) — выгружается ПОЛНЫЙ мозг, а не только выход.
    """
    circuit = get_circuit(kind)
    out: dict[str, Any] = {
        "kind": circuit.kind,
        "trained": bool(circuit.trained),
        "n_cells": circuit.n_cells,
        "w": [[float(x) for x in row] for row in circuit.W_dn],
        "gain": float(circuit.gain),
        "tau": float(circuit.tau),
        "seed": int(getattr(circuit, "seed", 0)),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wiring": circuit.wiring() if hasattr(circuit, "wiring") else None,
    }
    if getattr(circuit, "real_wiring", False):
        path = Path(__file__).resolve().parent.parent / "data" / "circuit_v1.npz"
        if path.exists():
            out["real_wiring"] = {
                "format": "circuit_v1.npz (base64)",
                "data": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
    return out


@app.post("/api/brain/import")
def brain_import(body: BrainImportIn) -> dict[str, Any]:
    """Загрузить мозг из файла. Проводка сверяется по хешу; реальная — устанавливается целиком."""
    circuit = get_circuit(body.kind)
    w = np.asarray(body.w, dtype=np.float64)
    if w.shape != circuit.W_dn.shape:
        return {
            "ok": False,
            "error": f"ожидалась матрица {circuit.W_dn.shape[0]}×{circuit.W_dn.shape[1]}, пришло {w.shape[0]}×{w.shape[1]}",
        }
    if body.wiring is not None and hasattr(circuit, "restore_wiring"):
        if not circuit.restore_wiring(body.wiring):
            return {
                "ok": False,
                "error": "проводка файла не совпадает с версией стенда — экспортируйте мозг заново",
            }
    # реальная проводка FlyWire: вшитые блоки устанавливаются и сохраняются в data/circuit_v1.npz
    real_installed = False
    if body.real_wiring is not None and hasattr(circuit, "install_real_wiring"):
        data = str(body.real_wiring.get("data") or "")
        if data and circuit.install_real_wiring(base64.b64decode(data)):
            real_installed = True
            out_path = Path(__file__).resolve().parent.parent / "data" / "circuit_v1.npz"
            out_path.write_bytes(base64.b64decode(data))
    circuit.W_dn = w
    circuit.trained = bool(body.trained)
    if body.gain is not None:
        circuit.gain = float(np.clip(body.gain, 0.05, 5.0))
    if body.tau is not None:
        circuit.tau = float(np.clip(body.tau, 0.002, 0.5))
    circuit.save()
    return {
        "ok": True,
        "kind": circuit.kind,
        "n_cells": circuit.n_cells,
        "trained": circuit.trained,
        "gain": circuit.gain,
        "tau": circuit.tau,
        "real_wiring_installed": real_installed,
    }


class BrainCompareIn(BaseModel):
    scenario: ScenarioIn
    kinds: list[str] = ["stub", "full", "connectome"]
    laws: list[str] = []  # законы наведения для сравнения: apn, pure, clos
    with_traj: bool = False  # добавить traj_m/traj_t — для наложения траекторий на сцене


def _metrics(res, sc: Scenario) -> dict[str, Any]:
    t_end = res.t_end if res.t_end is not None else (res.frames[-1].t if res.frames else None)
    return {
        "miss_m": round(float(res.miss_m), 1),
        "cpa_m": round(float(res.cpa_m), 1),
        "trigger_range_m": round(float(res.trigger_range_m), 1),
        "hit": bool(res.hit),
        "n_peak": round(float(res.n_peak), 1),
        "n_mean_g": round(float(res.n_mean_g), 2),
        "n_int": round(float(res.n_int), 1),
        "lock_frac": round(float(res.lock_fraction), 3),
        "t_end": round(float(t_end), 2) if t_end is not None else None,
        "t_guide": round(float(res.t_guide), 2) if res.t_guide is not None else None,
        "ref_dev_m": round(float(res.ref_dev_m), 1) if res.ref_dev_m is not None else None,
        "ref_rms_m": round(float(res.ref_rms_m), 1) if getattr(res, "ref_rms_m", None) is not None else None,
        "ref_nrms": round(float(res.ref_nrms), 4) if getattr(res, "ref_nrms", None) is not None else None,
        "t_ref": round(float(res.t_ref), 2) if res.t_ref is not None else None,
        "terminal_zem_m": round(float(res.terminal_zem_m), 1) if getattr(res, "terminal_zem_m", None) is not None else None,
        "h_cv_m": round(float(res.h_cv_m), 1) if getattr(res, "h_cv_m", None) is not None else None,
        "h0_m": round(float(res.h0_m), 1) if getattr(res, "h0_m", None) is not None else None,
        "impact_angle_deg": round(float(res.impact_angle_deg), 1) if getattr(res, "impact_angle_deg", None) is not None else None,  # η — угол встречи
        "t_max": float(sc.t_max),
        "metrics_version": 4,
        "model_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "n_eff_median": round(float(res.n_eff_median), 3) if res.n_eff_median is not None else None,
        "n_eff_q25": round(float(res.n_eff_q25), 3) if res.n_eff_q25 is not None else None,
        "n_eff_q75": round(float(res.n_eff_q75), 3) if res.n_eff_q75 is not None else None,
        "n_eff_valid_frac": round(float(res.n_eff_valid_frac), 3),
        "sat_frac": round(float(res.sat_frac), 3),
        "corr_n_eff_rho": round(float(res.corr_n_eff_rho), 3) if res.corr_n_eff_rho is not None else None,
    }


@app.post("/api/brain/compare")
def brain_compare(body: BrainCompareIn) -> dict[str, Any]:
    """Сравнение на ОДИНАКОВЫХ условиях: те же начальные условия, t_max, dt,
    радиус срабатывания и динамика для всех строк. Каждая строка помечена по
    информационному бюджету (поле `info_group`):
    - «oracle» — законы с точной геометрией (эталоны с подсказкой, НЕ равноправные
      сенсорным участникам);
    - «sensor» — МПС по измерениям ГСН, экспериментальное МПС с переменным
      коэффициентом (сенсорная) и био-мозга (только измерения после поля зрения,
      шумов, отказов и задержки). Профиль сенсора (поле зрения) — в `sensor_fov_deg`.
    Единый рейтинг oracle и сенсорных законов не строится (см. docs/model_assumptions.md)."""
    from dataclasses import replace

    from navedenie.glossary import LAW_LABEL as law_label  # единый источник (§4)
    base = _sc(body.scenario)
    # для наложения траекторий нужен плотный список точек, для метрик — нет
    stride = max(1, int(0.04 / max(base.dt, 1e-4))) if body.with_traj else 100_000

    def _finalize(row: dict[str, Any], res: Any, with_traj: bool, sensory: bool = False, sensor_fov_deg: float | None = None) -> dict[str, Any]:
        row["sensory"] = sensory
        # честная группа: oracle (точная геометрия) ≠ sensor (только измерения)
        row["info_group"] = "sensor" if sensory else "oracle"
        # профиль сенсора: поле зрения, реально доступное этому участнику
        row["sensor_fov_deg"] = sensor_fov_deg
        if with_traj:
            row["traj_m"] = [fr.missile.tolist() for fr in res.frames]
            row["traj_t"] = [fr.target.tolist() for fr in res.frames]
        return row

    out: list[dict[str, Any]] = []
    # строки законов наведения (по запросу клиента)
    for law in body.laws:
        if law not in ("pn", "tpn", "apn", "pure", "clos", "pn_gsn", "pn_sched_oracle", "pn_sched_sensor"):
            continue
        sc_l = replace(base, mode="pn", law=law)
        res = collect(sc_l, stride=stride)
        is_sensor = law in ("pn_gsn", "pn_sched_sensor")
        out.append(_finalize({"kind": law, "label": law_label.get(law, law), "n_cells": 0, "trained": None, **_metrics(res, sc_l)}, res, body.with_traj, sensory=is_sensor, sensor_fov_deg=sc_l.fov_deg if is_sensor else None))
    # эталонный МПС (oracle)
    ref_res = collect(replace(base, mode="pn", law="pn"), stride=stride)
    out.append(_finalize({"kind": "pn", "label": law_label["pn"], "n_cells": 0, "trained": None, **_metrics(ref_res, base)}, res=ref_res, with_traj=body.with_traj))
    for kind in body.kinds:
        if kind not in ("stub", "full", "connectome", "ensemble"):
            continue
        if kind == "ensemble":
            circuit = _make_ensemble()
            circuit.reset()
            sc_k = replace(base, mode="bio", brain="connectome")
            res = collect(sc_k, stride=stride, circuit=circuit)
            row = {"kind": "ensemble", "label": "ансамбль (медиана трёх)", "n_cells": circuit.n_cells, "trained": circuit.trained, **_metrics(res, sc_k)}
        else:
            circuit = get_circuit(kind)
            sc_k = replace(base, mode="bio", brain=kind)
            res = collect(sc_k, stride=stride)
            row = {
                "kind": kind,
                "label": {"stub": "схема", "full": "полный", "connectome": "коннектом-модель"}.get(kind, kind),
                "n_cells": circuit.n_cells,
                "trained": bool(circuit.trained),
                **_metrics(res, sc_k),
            }
        out.append(_finalize(row, res, body.with_traj, sensory=True, sensor_fov_deg=sc_k.bio_fov_deg))
    return {"results": out, "kill_radius_m": base.kill_radius_m, "metrics_version": 4, "model_version": MODEL_VERSION, "feature_schema_version": FEATURE_SCHEMA_VERSION}


def _make_ensemble():
    """Ансамбль «медиана трёх»: схема + полный + коннектом на текущих весах."""
    from navedenie.circuit import EnsembleBrain

    ens = EnsembleBrain([get_circuit("stub"), get_circuit("full"), get_circuit("connectome")])
    ens.reset()
    return ens


@app.get("/api/brain/ablation")
def brain_ablation() -> dict[str, Any]:
    """Абляция зон коннектома: поочерёдно выключаем зоны синаптических каналов
    (обнуляем их строки W_fx, НЕ сохраняя веса) и меряем каноническое трио.
    Показывает вклад каждой зоны в точность наведения."""
    import numpy as np

    from navedenie.circuit import CHANNEL_ZONES, ConnectomeCircuit
    from navedenie.train import _eval_canonical

    # отдельный экземпляр сети с весами с диска: не мешаем живому мозгу стенда
    # (параллельные пуск/сравнение иначе искажают друг друга) и не зависим от него
    circuit = ConnectomeCircuit()
    circuit.load()
    zones_list = getattr(circuit, "channel_zones", [])
    snap = circuit.W_fx.copy()

    def _row(r: dict[str, Any]) -> dict[str, Any]:
        return {
            "miss": round(float(r["miss"]), 1),
            "hit_rate": round(float(r["hit_rate"]), 3),
            "ref_dev": round(float(r["ref_dev"]), 1) if r.get("ref_dev") is not None else None,
            "t_guide": round(float(r["t_guide"]), 2) if r.get("t_guide") is not None else None,
        }

    circuit.reset()
    base = _row(_eval_canonical(circuit))
    rows: list[dict[str, Any]] = []
    for zone, _count in CHANNEL_ZONES:
        if zone == "прямой":
            continue  # прямые каналы — врождённый каркас, не выключаем
        mask = np.array([z == zone for z in zones_list], dtype=bool)
        if not mask.any():
            continue
        circuit.W_fx[mask, :] = 0.0
        circuit.reset()
        r = _row(_eval_canonical(circuit))
        circuit.W_fx = snap.copy()
        circuit.reset()
        rows.append({"zone": zone, "channels": int(mask.sum()), **r})
    circuit.W_fx = snap
    circuit.reset()
    return {"kind": "connectome", "base": base, "rows": rows}

@app.get("/api/brain/faults")
def brain_faults() -> dict[str, Any]:
    """Карта отказов: деградация промаха при «умирании» доли омматидиев сетчатки
    (0…80%). Каждая точка — медиана канонических аспектов с фиксированным seed."""
    import numpy as np

    from navedenie.train import _rollout_update

    circuit = get_circuit("connectome")
    rows: list[dict[str, Any]] = []
    for frac in (0.0, 0.25, 0.5, 0.75):
        misses: list[float] = []
        hits = 0
        for a in ("head-on", "beam", "tail-chase"):
            sc = Scenario(aspect=a, mode="bio", t_max=12.0, dt=0.02, circuit_gain=circuit.gain,
                          retina_death_p=frac, seed=100 + int(frac * 50))
            r = _rollout_update(circuit, sc, lr=0.0)
            misses.append(float(r["miss"]))
            hits += 1 if r["hit"] else 0
        rows.append({
            "fraction": frac,
            "miss": round(float(np.median(misses)), 1),
            "worst": round(max(misses), 1),
            "hit_rate": round(hits / 3, 2),
        })
    return {"kind": "connectome", "rows": rows}


class TransferIn(BaseModel):
    kind: str = "connectome"
    episodes: int = 10


@app.post("/api/transfer")
def transfer_endpoint(body: TransferIn) -> dict[str, Any]:
    """Матрица переносимости: свежий мозг обучается на ОДНОМ манёвре — испытывается
    на всех (включая прямолинейную). Диагональ — «обучался здесь», вне диагонали —
    перенос навыка. Ответ на вопрос «обобщает или заучивает». На коннектом считается
    несколько минут (экземпляры тяжёлые), на схеме/полном — секунды."""
    from navedenie.science import transfer_matrix

    kind = body.kind if body.kind in ("stub", "full", "connectome") else "connectome"
    return transfer_matrix(kind, episodes=max(2, min(int(body.episodes), 24)))


class ScalingIn(BaseModel):
    kind: str = "connectome"
    sizes: list[int] = [32, 64, 128]
    episodes: int = 10


@app.post("/api/scaling")
def scaling_endpoint(body: ScalingIn) -> dict[str, Any]:
    """Scaling-кривая: мозги разного размера (каналы коннектома / пул полного)
    обучаются ОДИНАКОВО и сравниваются по каноническому трио. График
    «число обучаемых параметров ↔ промах»: сколько мозга окупается."""
    from navedenie.science import scaling_curve

    kind = body.kind if body.kind in ("full", "connectome") else "connectome"
    sizes = sorted({max(8, min(int(s), 512)) for s in body.sizes})[:6] or [32, 64, 128]
    return scaling_curve(kind, sizes=tuple(sizes), episodes=max(2, min(int(body.episodes), 40)))


class SpeedMatrixIn(BaseModel):
    kinds: list[str] = ["connectome"]
    vm_grid: list[float] = [650, 780, 950]
    vt_grid: list[float] = [150, 260, 360]
    speed_modes: list[str] = ["constant", "accelerate", "decelerate", "pulse"]
    aspects: list[str] = ["head-on", "beam", "tail-chase"]
    maneuvers: list[str] = ["straight", "weave"]


@app.post("/api/science/speed-matrix")
def speed_matrix_endpoint(body: SpeedMatrixIn) -> dict[str, Any]:
    """Held-out матрица «v_m × v_t × аспект × профиль скорости × манёвр»:
    hit rate, CPA-квартили, усилие, медианный N_eff, доля насыщения и захвата
    на каждую ячейку. Test-зёрна не пересекаются с train/validation."""
    from navedenie.science import speed_matrix

    kinds = tuple(k for k in body.kinds if k in ("stub", "full", "connectome")) or ("connectome",)
    modes = tuple(m for m in body.speed_modes if m in ("constant", "accelerate", "decelerate", "pulse", "sine"))
    aspects = tuple(a for a in body.aspects if a in ("head-on", "beam", "tail-chase"))
    return speed_matrix(
        kinds=kinds,
        vm_grid=tuple(float(v) for v in body.vm_grid),
        vt_grid=tuple(float(v) for v in body.vt_grid),
        speed_modes=modes,
        aspects=aspects,
        maneuvers=tuple(str(m) for m in body.maneuvers),
    )


class NeffReportIn(BaseModel):
    kind: str = "connectome"


@app.post("/api/science/neff-report")
def neff_report_endpoint(body: NeffReportIn) -> dict[str, Any]:
    """Анализ выученного закона: постоянен ли N_eff, зависит ли от rho/t_go,
    меняется ли при насыщении и скоростях, обобщается ли на невиданные профили.
    Истинная геометрия используется только в диагностике — не в управлении."""
    from navedenie.science import neff_report

    kind = body.kind if body.kind in ("stub", "full", "connectome") else "connectome"
    return neff_report(kind)


class FeatureAblationIn(BaseModel):
    kind: str = "connectome"


@app.post("/api/science/ablation-features")
def feature_ablation_endpoint(body: FeatureAblationIn) -> dict[str, Any]:
    """Абляции фазовых признаков: без theta, без rho, без обоих, с фиксированным
    rho и с перестановкой rho между эпизодами. Сравнение hit rate / CPA /
    усилия / N_eff показывает вклад theta/rho в политику."""
    from navedenie.science import feature_ablation

    kind = body.kind if body.kind in ("stub", "full", "connectome") else "connectome"
    return feature_ablation(kind)


# ── конвейер «закон наведения» (P6–P10): диагностика, гипотезы, closed-loop ──


class LawKindIn(BaseModel):
    kind: str = "stub"


def _law_circuit(kind: str):
    from navedenie.science import get_live_circuit

    return get_live_circuit(kind if kind in ("stub", "full", "connectome") else "stub")


@app.post("/api/science/neff-decomposition")
def neff_decomposition_endpoint(body: LawKindIn) -> dict[str, Any]:
    """P6: разложение команды a_bio = N·q + residual на каноническом трио +
    скоростных ячейках: оконная N_wls, alignment, explained, residual, bootstrap CI
    по эпизодам и вердикт. Истинная геометрия — только постфактум."""
    from navedenie.diagnostics import neff_decomposition
    from navedenie.law import canonical_trio

    circuit = _law_circuit(body.kind)
    scens = canonical_trio(circuit.gain)
    from navedenie.train import _episode_scenario

    scens += [_episode_scenario(40 + i, circuit.gain) for i in range(3)]
    return neff_decomposition(circuit, scens)


@app.post("/api/science/law-hypotheses")
def law_hypotheses_endpoint(body: LawKindIn) -> dict[str, Any]:
    """P7: сравнение H1–H5 на полночастотных трассах (разбиение по эпизодам).
    Напоминание: аппроксимация команды ≠ закон — см. /api/science/closed-loop."""
    from navedenie.diagnostics import episode_trace, fit_hypotheses
    from navedenie.law import canonical_trio

    circuit = _law_circuit(body.kind)
    scens = canonical_trio(circuit.gain)
    from navedenie.train import _episode_scenario

    scens += [_episode_scenario(60 + i, circuit.gain) for i in range(4)]
    traces = [episode_trace(circuit, sc) for sc in scens]
    report = fit_hypotheses(traces)
    # Сырые коэффициенты NARX нужны соседнему closed-loop расчёту, но это
    # внутренняя ndarray-матрица и она не является частью публичного отчёта.
    # Если вернуть её напрямую, FastAPI/Pydantic не сможет сериализовать ответ.
    report.pop("coef_narx", None)
    return report


@app.post("/api/science/closed-loop")
def closed_loop_endpoint(body: LawKindIn) -> dict[str, Any]:
    """P8: deployable-суррогаты ставятся в контур вместо BIO и летят одно и то же
    held-out множество; сравнение по физическим метрикам (hit rate, geometric CPA)."""
    from navedenie.diagnostics import NarxLaw, PolyReadout, SensorConstantPN, SensorScheduledPN, closed_loop_check, episode_trace, fit_hypotheses
    from navedenie.law import canonical_trio

    circuit = _law_circuit(body.kind)
    scens = canonical_trio(circuit.gain)
    # параметры deployable-форм берём из фитинга H1/H2/H5 на отдельных эпизодах
    from navedenie.train import _episode_scenario

    fit_traces = [episode_trace(circuit, _episode_scenario(80 + i, circuit.gain)) for i in range(4)]
    fits = fit_hypotheses(fit_traces)
    hyp = fits["hypotheses"]
    laws: list[Any] = []
    try:
        laws.append(SensorConstantPN(float(hyp["H1_const_pn"]["coef_summary"]["N"])))
    except Exception:  # noqa: BLE001
        pass
    try:
        h2 = hyp["H2_sched_pn"]["coef_summary"]
        laws.append(SensorScheduledPN(float(h2["N0"]), float(h2["k_rho_s"])))
    except Exception:  # noqa: BLE001
        pass
    from navedenie.formula import _flight_fit

    flight = _flight_fit(body.kind if body.kind in ("stub", "full", "connectome") else "stub", circuit, 2)
    if flight is not None:
        laws.append(PolyReadout(flight["fits"]["pitch"]["coef"], flight["fits"]["yaw"]["coef"], 2))
    coef_narx = fits.get("coef_narx")
    if coef_narx is not None:
        laws.append(NarxLaw(coef_narx))
    return closed_loop_check(circuit, laws, scens)


@app.post("/api/science/law-pipeline")
def law_pipeline_endpoint(body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Полный двухэтапный конвейер (P1–P3) на быстром пресете: этап A — warm start
    подражанием, этап B — outcome evolution (teacher выключен, лексикографический
    fitness), чемпион по validation. Тяжёлый запрос: stub — секунды, connectome — минуты.
    body: {kind, init, preset, seed}."""
    body = body or {}
    from navedenie.law import discover_law

    kind = str(body.get("kind", "stub"))
    if kind not in ("stub", "full", "connectome"):
        kind = "stub"
    init = str(body.get("init", "pn_coarse_warmstart"))
    preset = str(body.get("preset", "quick"))
    if preset not in ("quick", "research"):
        preset = "quick"
    seed = int(body.get("seed", 0))
    out = discover_law(kind, init=init, preset=preset, seed=seed)
    out.pop("weights", None)  # веса не гоняем по HTTP — они в артефактах
    return out


@app.get("/api/science/law-runs")
def law_runs_endpoint() -> dict[str, Any]:
    """Список checkpoint'ов экспериментов artifacts/law_discovery/<run_id>/."""
    from navedenie.law import ARTIFACTS_DIR

    items = []
    if ARTIFACTS_DIR.exists():
        for d in sorted(ARTIFACTS_DIR.iterdir()):
            if d.is_dir():
                items.append(
                    {
                        "run_id": d.name,
                        "files": sorted(p.name for p in d.iterdir() if p.is_file()),
                        "saved_at": datetime.fromtimestamp(d.stat().st_mtime).isoformat(timespec="seconds"),
                    }
                )
    return {"runs": items, "dir": str(ARTIFACTS_DIR)}


@app.post("/api/science/photometric")
def photometric_endpoint(body: LawKindIn) -> dict[str, Any]:
    """P9: фотометрическая устойчивость — яркость пятна 0.7/1.0/1.4 при той же
    геометрии. Просадка = политика читала яркость как скрытый датчик дальности."""
    from navedenie.diagnostics import photometric_robustness
    from navedenie.law import canonical_trio

    circuit = _law_circuit(body.kind)
    return photometric_robustness(circuit, canonical_trio(circuit.gain))


class NightReportIn(BaseModel):
    title: str = "Ночная смена"
    map_rows: list[dict[str, Any]] = []
    ablation: dict[str, Any] | None = None
    faults: list[dict[str, Any]] = []
    coev_history: list[dict[str, Any]] = []
    transfer: dict[str, Any] | None = None
    scaling: dict[str, Any] | None = None


@app.post("/api/report/night")
def report_night(body: NightReportIn) -> dict[str, Any]:
    """Автоотчёт ночной смены: markdown + PNG-графики (без внешних зависимостей)
    в data/experiments/night_<дата>/; текст отчёта возвращается в ответе."""
    from datetime import datetime as _dt

    from navedenie import png

    stamp = _dt.now().strftime("%Y%m%d_%H%M")
    folder = _EXP_DIR / f"night_{stamp}"
    folder.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    def _png(name: str) -> str:
        files.append(name)
        return name

    lines = [f"# МУХОЛЁТ — {body.title}", "", f"Дата: {_dt.now().strftime('%Y-%m-%d %H:%M')}", ""]

    if body.ablation:
        vals = [float(body.ablation["base"]["miss"])] + [float(r["miss"]) for r in body.ablation["rows"]]
        png.bars_png(folder / "ablation.png", vals)
        files.append("ablation.png")
        lines += ["## Абляция зон коннектома", "",
                  f"База: {body.ablation['base']['miss']} м (перехваты {round(body.ablation['base']['hit_rate'] * 100)}%)",
                  "", "![абляция](ablation.png)", "",
                  "| Зона | Каналов | Промах, м | Перехваты |", "|---|---|---|---|"]
        for r in body.ablation["rows"]:
            lines.append(f"| {r['zone']} | {r['channels']} | {r['miss']} | {round(r['hit_rate'] * 100)}% |")
        lines.append("")
    if body.map_rows:
        maneuvers: list[str] = []
        laws: list[str] = []
        for r in body.map_rows:
            if r["maneuver"] not in maneuvers:
                maneuvers.append(r["maneuver"])
            if r["law"] not in laws:
                laws.append(r["law"])
        grid = [
            [next((float(x["miss_fly"]) for x in body.map_rows if x["maneuver"] == m and x["law"] == l), 0.0) for l in laws]
            for m in maneuvers
        ]
        _png("map.png")
        png.heatmap_png(folder / "map.png", grid)
        lines += ["## Карта преимуществ (муха vs ПН)", "", "![карта](map.png)", "",
                  "| Манёвр | Закон | ПН, м | Муха, м | Преимущество, м |", "|---|---|---|---|---|"]
        for r in body.map_rows:
            lines.append(f"| {r['maneuver']} | {r['law']} | {r['miss_pn']} | {r['miss_fly']} | {r['advantage']:+} |")
        lines.append("")
    if body.faults:
        _png("faults.png")
        png.lines_png(folder / "faults.png", [{"values": [float(r["miss"]) for r in body.faults], "color": png.PHOS}])
        lines += ["## Карта отказов сетчатки", "", "![отказы](faults.png)", "",
                  "| Доля отказавших | Промах, м | Худший, м | Перехваты |", "|---|---|---|---|"]
        for r in body.faults:
            lines.append(f"| {round(r['fraction'] * 100)}% | {r['miss']} | {r['worst']} | {round(r['hit_rate'] * 100)}% |")
        lines.append("")
    if body.coev_history:
        _png("coev.png")
        png.lines_png(folder / "coev.png", [{"values": [float(h["miss_fly"]) for h in body.coev_history], "color": png.AMBER}])
        lines += ["## Коэволюция: опаснейшие цели", "", "![коэволюция](coev.png)", "",
                  "| Поколение | Цель | Промах мухи, м |", "|---|---|---|"]
        for h in body.coev_history:
            lines.append(f"| {h['gen']} | {h['maneuver']} n={h.get('n_target', '')} | {h['miss_fly']} |")
        lines.append("")
    if body.transfer:
        t = body.transfer
        _png("transfer.png")
        png.heatmap_png(folder / "transfer.png", [[float(v) for v in row["tests"].values()] for row in t["rows"]])
        cols = list(t["rows"][0]["tests"].keys()) if t["rows"] else []
        lines += ["## Матрица переносимости", "",
                  f"Мозг «{t.get('kind')}», {t.get('episodes')} эпизодов обучения на манёвр, {t.get('seconds')} с.", "",
                  "![переносимость](transfer.png)", "",
                  "| обучался ↓ / испытан → | " + " | ".join(cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for row in t["rows"]:
            lines.append("| " + row["train"] + " | " + " | ".join(f"{row['tests'][c]:.0f}" for c in cols) + " |")
        lines.append("")
    if body.scaling:
        s = body.scaling
        _png("scaling.png")
        png.lines_png(folder / "scaling.png", [{"values": [float(r["miss_after"]) for r in s["rows"]], "color": png.PHOS}])
        lines += ["## Scaling-кривая: параметры ↔ промах", "",
                  f"Мозг «{s.get('kind')}», {s.get('episodes')} одинаковых эпизодов на каждый размер, {s.get('seconds')} с.", "",
                  "![scaling](scaling.png)", "",
                  "| Размер | Обучаемых параметров | Промах до, м | Промах после, м | Перехваты | Откл. от ПН, м |",
                  "|---|---|---|---|---|---|"]
        for r in s["rows"]:
            lines.append(
                f"| {r['size']} | {r['params']} | {r['miss_before']} | {r['miss_after']} | "
                f"{round(r['hit_rate_after'] * 100)}% | {r['ref_dev_after']} |"
            )
        lines.append("")

    md = chr(10).join(lines)
    name = f"night_report_{stamp}.md"
    path = folder / name
    path.write_text(md, encoding="utf-8")
    return {"ok": True, "name": str(folder.relative_to(_EXP_DIR)) + "/" + name, "path": str(path), "files": files, "markdown": md}


class BrainResetIn(BaseModel):
    kind: str


@app.post("/api/brain/reset")
def brain_reset(body: BrainResetIn) -> dict[str, Any]:
    """Сбросить мозг к заводскому состоянию: файл обученных весов удаляется,
    создаётся свежий необученный мозг с размерами по умолчанию (врождённый рефлекс)."""
    kind = body.kind if body.kind in ("stub", "full", "connectome") else "stub"
    wpath = _data_dir() / f"weights_{kind}.npz"
    if wpath.exists():
        wpath.unlink()
    old_c = get_circuit(kind)
    if kind == "connectome":
        new_c = ConnectomeCircuit(tau_s=old_c.tau, gain=old_c.gain)
    else:
        new_c = FlyCircuit(kind=kind, tau_s=old_c.tau, gain=old_c.gain)
    new_c.trained = False
    new_c.save()
    brain_replace(kind, new_c)
    return {"ok": True, "kind": kind, "n_cells": new_c.n_cells, "trained": False}


class MapIn(BaseModel):
    brain: str = "connectome"
    v_m: float = 780.0
    v_t: float = 260.0
    range_m: float = 8000.0
    off_axis_m: float = 420.0
    n_max: float = 30.0
    pn_n: float = 4.0
    dt: float = 0.02
    fov_deg: float = 14.0
    kill_radius_m: float = 45.0
    repeats: int = 1  # повторов био-прогона на ячейку: медиана + разброс (шумы меняют seed)
    retina_death_p: float = 0.0  # доля умерших омматидиев (карта отказов)


# ─── Дистилляция «большой мозг → схема»: конвейер учитель → дистиллят → файл ──

_distill_state: dict[str, Any] = {}  # W_dn дистиллята + учитель, в памяти процесса


class DistillIn(BaseModel):
    teacher: str = "connectome"  # коннектом | полный | схема


@app.post("/api/brain/distill")
def brain_distill(body: DistillIn | None = None) -> dict[str, Any]:
    """Дистилляция: маленькая схема (выход 2×8) учится повторять команды учителя
    на сетке признаков. Выбор учителя — коннектом, полный или схема. Показывает,
    что теряется при сжатии большого мозга до 16 весов. Основную схему стенда
    не трогаем — до кнопки «Применить»."""
    import numpy as np

    from navedenie.circuit import FlyCircuit
    from navedenie.formula import brain_dn
    from navedenie.train import _eval_canonical

    body = body or DistillIn()
    teacher_kind = body.teacher if body.teacher in ("connectome", "full", "stub") else "connectome"
    teacher_circuit = get_circuit(teacher_kind)
    rng = np.random.default_rng(7)
    X = rng.uniform(-1.0, 1.0, size=(6000, FEAT_DIM))
    T = brain_dn(teacher_kind, teacher_circuit, X)

    w = np.zeros((2, FEAT_DIM))
    lr0, epochs = 0.8, 400
    for ep in range(epochs):
        lr = lr0 * (1.0 - ep / epochs) + 0.02
        pred = np.tanh(X @ w.T)
        err = T - pred
        w += lr * (err.T @ X) / len(X)
        np.clip(w, -4.0, 4.0, out=w)
    _distill_state["w"] = w.copy()
    _distill_state["teacher"] = teacher_kind

    def _eval_with(w_dn: np.ndarray | None) -> dict[str, Any]:
        c = FlyCircuit(kind="stub")
        if w_dn is not None:
            c.W_dn = w_dn.copy()
            c.trained = True
        c.reset()
        r = _eval_canonical(c)
        return {"miss": round(float(r["miss"]), 1), "hit_rate": round(float(r["hit_rate"]), 2), "ref_dev": round(float(r["ref_dev"]), 1)}

    teacher_eval = _eval_canonical(teacher_circuit)
    return {
        "epochs": epochs,
        "teacher_kind": teacher_kind,
        "teacher": {"miss": round(float(teacher_eval["miss"]), 1), "hit_rate": round(float(teacher_eval["hit_rate"]), 2)},
        "distilled": _eval_with(w),
        "scratch": _eval_with(None),
    }


@app.post("/api/brain/distill/save")
def brain_distill_save() -> dict[str, Any]:
    """Сохранить дистиллят в data/weights_distilled.npz — переживает перезапуск стенда."""
    if "w" not in _distill_state:
        raise HTTPException(status_code=409, detail="сначала выполните дистилляцию")
    import numpy as np

    from navedenie.circuit import FlyCircuit

    c = FlyCircuit(kind="stub")
    c.W_dn = np.asarray(_distill_state["w"], dtype=np.float64).copy()
    c.trained = True
    path = c.save(_data_dir() / "weights_distilled.npz")
    return {"ok": True, "path": str(path), "teacher": _distill_state.get("teacher")}


@app.post("/api/brain/distill/load")
def brain_distill_load() -> dict[str, Any]:
    """Загрузить сохранённый дистиллят обратно в рабочий буфер (можно «Применить»)."""
    import numpy as np

    from navedenie.circuit import FlyCircuit
    from navedenie.train import _eval_canonical

    path = _data_dir() / "weights_distilled.npz"
    if not path.exists():
        raise HTTPException(status_code=404, detail="сохранённого дистиллята нет: сначала дистилляция + «Сохранить»")
    c = FlyCircuit(kind="stub")
    if not c.load(path):
        raise HTTPException(status_code=500, detail="файл дистиллята не читается")
    _distill_state["w"] = c.W_dn.copy()
    _distill_state.setdefault("teacher", "файл")
    c.reset()
    r = _eval_canonical(c)
    return {
        "ok": True,
        "teacher": _distill_state.get("teacher"),
        "distilled": {
            "miss": round(float(r["miss"]), 1),
            "hit_rate": round(float(r["hit_rate"]), 2),
            "ref_dev": round(float(r["ref_dev"]), 1),
        },
    }


@app.post("/api/brain/distill/apply")
def brain_distill_apply() -> dict[str, Any]:
    """Применить дистиллят к схеме стенда: схема получает выученные веса и сохраняется."""
    if "w" not in _distill_state:
        raise HTTPException(status_code=409, detail="сначала выполните дистилляцию")
    circuit = get_circuit("stub")
    circuit.W_dn = _distill_state["w"].copy()
    circuit.trained = True
    circuit.reset()
    set_circuit(circuit)
    return {"ok": True, "n_cells": circuit.n_cells}


# ─── Коэволюция «цель против мухи» ───────────────────────────────────────────

_coev_state: dict[str, Any] = {}
COEV_MANEUVERS = ("turn", "weave", "weave_var", "scissors", "dive")
COEV_POP = 6


@app.post("/api/coevolve/start")
def coevolve_start() -> dict[str, Any]:
    """Гонка вооружений: популяция целей эволюционирует против текущего мозга.
    Фитнес цели = промах мухи (чем больше, тем «опаснее» цель)."""
    import numpy as np

    rng = np.random.default_rng()
    pop = [
        {
            "maneuver": COEV_MANEUVERS[int(rng.integers(0, len(COEV_MANEUVERS)))],
            "n_target": float(rng.choice([1.0, 2.0, 3.0, 4.0, 5.0])),
            "range_m": float(rng.uniform(4000, 10000)),
            "off_axis_m": float(rng.uniform(100, 1200)),
        }
        for _ in range(COEV_POP)
    ]
    _coev_state.clear()
    _coev_state.update({"pop": pop, "gen": 0, "history": [], "rng": rng})
    return {"ok": True, "gen": 0, "pop_size": COEV_POP}


@app.post("/api/coevolve/step")
def coevolve_step() -> dict[str, Any]:
    import numpy as np

    from dataclasses import replace

    st = _coev_state
    if not st.get("pop"):
        raise HTTPException(status_code=409, detail="коэволюция не запущена: /api/coevolve/start")
    rng: np.random.Generator = st["rng"]

    def _eval_target(g: dict) -> float:
        sc = Scenario(
            aspect="head-on", v_m=780.0, v_t=260.0, range_m=g["range_m"], off_axis_m=g["off_axis_m"],
            n_max=30.0, n_target=g["n_target"], maneuver=g["maneuver"], pn_n=4.0, dt=0.02, t_max=12.0,
            fov_deg=14.0, kill_radius_m=45.0, law="pn", mode="bio", brain="connectome", circuit_gain=1.15,
        )
        r = collect(sc, stride=100_000)
        return float(r.miss_m)

    scored = sorted(((_eval_target(g), g) for g in st["pop"]), key=lambda p: -p[0])
    best_miss, best_g = scored[0]
    misses = sorted(s for s, _g in scored)
    n = len(misses)
    median_miss = misses[n // 2] if n % 2 else (misses[n // 2 - 1] + misses[n // 2]) / 2
    st["gen"] += 1
    st["history"].append({
        "gen": st["gen"], "miss_fly": round(best_miss, 1),
        "median_miss": round(median_miss, 1), "retrained": False, **best_g,
    })

    # элита 2 без изменений + турнирный отбор из топ-3 с мутациями
    elite = [dict(g) for _s, g in scored[:2]]
    kids: list[dict] = []
    while len(elite) + len(kids) < COEV_POP:
        i, j = (int(rng.integers(0, 3)) for _ in (0, 1))
        a, b = scored[min(i, len(scored) - 1)][1], scored[min(j, len(scored) - 1)][1]
        kid = {
            "maneuver": a["maneuver"] if rng.random() < 0.5 else b["maneuver"],
            "n_target": float(np.clip((a["n_target"] + b["n_target"]) / 2 + rng.normal(0, 0.8), 0.0, 5.0)),
            "range_m": float(np.clip(rng.choice([a["range_m"], b["range_m"]]) + rng.normal(0, 700), 3500, 11000)),
            "off_axis_m": float(np.clip(rng.choice([a["off_axis_m"], b["off_axis_m"]]) + rng.normal(0, 200), 50, 1400)),
        }
        if rng.random() < 0.3:
            kid["maneuver"] = COEV_MANEUVERS[int(rng.integers(0, len(COEV_MANEUVERS)))]
        kids.append(kid)
    st["pop"] = elite + kids

    return {
        "gen": st["gen"],
        "best": {"miss_fly": round(best_miss, 1), **best_g},
        "history": st["history"],
        "pop": st["pop"],
    }


@app.post("/api/coevolve/reset")
def coevolve_reset() -> dict[str, Any]:
    _coev_state.clear()
    return {"ok": True}


def _coev_champion_scenarios(count: int = 3) -> list[Scenario]:
    """Сценарии опаснейших целей коэволюции (элита популяции) — батч дообучения мухи."""
    top = _coev_state.get("pop", [])[:max(1, count)]
    return [
        Scenario(
            aspect="head-on", v_m=780.0, v_t=260.0, range_m=g["range_m"], off_axis_m=g["off_axis_m"],
            n_max=30.0, n_target=g["n_target"], maneuver=g["maneuver"], pn_n=4.0, dt=0.02, t_max=12.0,
            fov_deg=14.0, kill_radius_m=45.0, law="pn", mode="bio", brain="connectome", circuit_gain=1.15,
        )
        for g in top
    ]


@app.post("/api/coevolve/train_fly")
def coevolve_train_fly(body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Дообучить муху по результату против опаснейших целей коэволюции —
    замыкание петли «цель учится → муха учится». body: {generations, rounds}."""
    from dataclasses import replace

    from navedenie.engine import collect
    from navedenie.train import evolve_finish, evolve_start, evolve_step

    body = body or {}
    generations = max(1, min(int(body.get("generations", 3)), 10))
    count = max(1, min(int(body.get("rounds", 3)), 6))
    if not _coev_state.get("pop"):
        raise HTTPException(status_code=409, detail="коэволюция не запущена: /api/coevolve/start")

    def _fly_miss(g: dict) -> float:
        sc = Scenario(
            aspect="head-on", v_m=780.0, v_t=260.0, range_m=g["range_m"], off_axis_m=g["off_axis_m"],
            n_max=30.0, n_target=g["n_target"], maneuver=g["maneuver"], pn_n=4.0, dt=0.02, t_max=12.0,
            fov_deg=14.0, kill_radius_m=45.0, law="pn", mode="bio", brain="connectome", circuit_gain=1.15,
        )
        return float(collect(sc, stride=100_000).miss_m)

    # лестница меряется против ЧЕМПИОНА (самой опасной цели), а не лучшей для мухи
    before = max(_fly_miss(g) for g in _coev_state["pop"])
    champs = _coev_champion_scenarios(count)
    ev = evolve_start("connectome", generations=generations, sigma=0.3, batch_scenarios=champs)
    for _ in range(generations):
        evolve_step("connectome")
    fin = evolve_finish("connectome")
    after = max(_fly_miss(g) for g in _coev_state["pop"])
    ladder = {
        "fly_miss_before": round(before, 1),
        "fly_miss_after": round(after, 1),
        "generations": generations,
        "restored": fin.get("restored_best"),
    }
    _coev_state.setdefault("ladder", []).append(ladder)
    hist = _coev_state.get("history") or []
    if hist:  # поколение, на котором замыкалась петля, помечаем для графика истории
        hist[-1].update({"retrained": True, "fly_before": round(before, 1), "fly_after": round(after, 1)})
    return {"ok": True, **ladder, "history": hist, "canonical": {"miss": round(float(fin["miss_after"]), 1), "hit_rate": round(float(fin["hit_rate_after"]), 2)}}


@app.post("/api/map")
def map_advantage(body: MapIn) -> dict[str, Any]:
    """Карта преимуществ: сетка «манёвр × закон наведения» — промах эталонного МПС
    (mode=pn, заданный закон) против промаха био-мозга (mode=bio, текущий сохранённый)."""
    from dataclasses import replace

    maneuvers = [
        ("прямолинейно", "straight", 0),
        ("вираж", "turn", 3),
        ("змейка", "weave", 3),
        ("ножницы", "scissors", 3),
        ("горка/пике", "dive", 3),
    ]
    laws = [("pn", "МПС"), ("tpn", "истинный МПС"), ("apn", "МПС+компенсация"), ("pure", "погоня"), ("clos", "три точки")]
    rows: list[dict[str, Any]] = []
    for label_m, maneuver, n_target in maneuvers:
        for law_id, label_l in laws:
            base = Scenario(
                aspect="head-on", v_m=body.v_m, v_t=body.v_t, range_m=body.range_m,
                off_axis_m=body.off_axis_m, n_max=body.n_max, n_target=n_target,
                maneuver=maneuver, pn_n=body.pn_n, dt=body.dt, t_max=20.0,
                fov_deg=body.fov_deg, kill_radius_m=body.kill_radius_m,
                law=law_id, mode="pn",
            )
            res_law = collect(base, stride=100_000)
            # одинаковый t_max для обеих сторон — честное сравнение
            sc_bio = replace(base, mode="bio", brain=body.brain, t_max=20.0,
                             retina_death_p=body.retina_death_p)
            misses: list[float] = []
            energies: list[float] = []
            hits = 0
            for rep in range(max(1, body.repeats)):
                sc_rep = replace(sc_bio, seed=7919 + rep * 131)
                ens = _make_ensemble() if body.brain == "ensemble" else None
                res_bio = collect(sc_rep, stride=100_000, circuit=ens) if ens is not None else collect(sc_rep, stride=100_000)
                misses.append(float(res_bio.miss_m))
                energies.append(float(res_bio.n_int))
                hits += 1 if res_bio.hit else 0
            misses.sort()
            median_fly = misses[len(misses) // 2] if len(misses) % 2 else (misses[len(misses) // 2 - 1] + misses[len(misses) // 2]) / 2
            rows.append({
                "maneuver": label_m,
                "law": label_l,
                "law_id": law_id,
                "miss_pn": round(float(res_law.miss_m), 1),
                "hit_pn": bool(res_law.hit),
                "n_int_pn": round(float(res_law.n_int), 1),
                "miss_fly": round(median_fly, 1),
                "miss_fly_min": round(min(misses), 1),
                "miss_fly_max": round(max(misses), 1),
                "hit_fly_share": round(hits / len(misses), 2),
                "n_int_fly": round(sum(energies) / len(energies), 1),
                "advantage": round(float(res_law.miss_m) - median_fly, 1),
                "energy_advantage": round(float(res_law.n_int) - sum(energies) / len(energies), 1),
                "repeats": max(1, body.repeats),
            })
    if body.brain == "ensemble":
        ens = _make_ensemble()
        n_cells, brain_name = ens.n_cells, "ансамбль"
    else:
        circuit = get_circuit(body.brain if body.brain in ("stub", "full", "connectome") else "connectome")
        n_cells, brain_name = circuit.n_cells, body.brain
    return {"rows": rows, "brain": brain_name, "n_cells": n_cells, "kill_radius_m": body.kill_radius_m}


class BrainRebuildIn(BaseModel):
    kind: str
    pool_size: int = 32
    channels: int = 64


@app.post("/api/brain/rebuild")
def brain_rebuild(body: BrainRebuildIn) -> dict[str, Any]:
    """Пересоздать мозг с новыми размерами скрытых слоёв (сброс обучения).

    «Полный»: пул N → читающий слой 2×N; «коннектом»: N каналов 8→N → 2×N.
    Исследовательская ось: как число скрытых параметров влияет на точность наведения.
    """
    from navedenie.train import clear_learned_cache

    kind = body.kind if body.kind in ("stub", "full", "connectome") else "stub"
    if body.pool_size < 8 or body.pool_size > 128:
        raise HTTPException(status_code=422, detail="пул: 8…128")
    if body.channels < 16 or body.channels > 512:
        raise HTTPException(status_code=422, detail="каналы: 16…512")
    old_c = get_circuit(kind)
    if kind == "connectome":
        new_c = ConnectomeCircuit(tau_s=old_c.tau, gain=old_c.gain, channels=body.channels)
    else:
        new_c = FlyCircuit(kind=kind, tau_s=old_c.tau, gain=old_c.gain, pool_size=max(8, body.pool_size))
    new_c.trained = False
    new_c.save()
    brain_replace(kind, new_c)
    # старые снапшоты «лучших весов» несовместимы с новой формой — забываем их
    clear_learned_cache(kind)
    return {
        "ok": True,
        "kind": kind,
        "n_cells": new_c.n_cells,
        "pool_size": getattr(new_c, "pool_size", None),
        "channels": getattr(new_c, "channels", None),
    }


class RobustIn(BaseModel):
    scenario: ScenarioIn
    kinds: list[str] = ["stub", "full", "connectome"]
    noise_levels: list[float] = [0.0, 1.0, 2.0, 3.0, 4.0]


@app.post("/api/robustness")
def robustness(body: RobustIn) -> dict[str, Any]:
    """Робастность: серия прогонов с нарастающим шумом пеленга по каждому мозгу.

    Возвращает уровни шума и ряды метрик (промах, норм. СКО) на каждый уровень —
    для графика «метрика vs шум» и сравнения деградации мозгов.
    """
    from dataclasses import replace

    base = _sc(body.scenario)
    levels = [float(x) for x in body.noise_levels] or [0.0]
    labels = {"stub": "схема", "full": "полный", "connectome": "коннектом"}
    series = []
    for kind in body.kinds:
        if kind not in ("stub", "full", "connectome"):
            continue
        circuit = get_circuit(kind)
        miss_row: list[float] = []
        nrms_row: list[float | None] = []
        for lvl in levels:
            sc_k = replace(base, mode="bio", brain=kind, noise_az_deg=lvl, t_max=min(base.t_max, 20.0))
            res = collect(sc_k, stride=100_000)
            miss_row.append(round(float(res.miss_m), 1))
            nrms_row.append(round(float(res.ref_nrms), 4) if res.ref_nrms is not None else None)
        series.append({"kind": kind, "label": labels.get(kind, kind), "miss": miss_row, "nrms": nrms_row})
    return {"levels": levels, "series": series}


class MonteCarloIn(ScenarioIn):
    n_runs: int = 200      # число независимых траекторий (прогонов)
    seed_start: int = 1000  # первый seed; i-й прогон идёт с seed = seed_start + i


@app.post("/api/science/monte-carlo")
def monte_carlo_endpoint(body: MonteCarloIn) -> dict[str, Any]:
    """Monte-Carlo рассеивание: n_runs прогонов одного сценария с разными seed
    (шум измерений / срыв сопровождения / отказы сетчатки). Сводка: p_hit с
    точным биномиальным 95 % ДИ (Уилсон), разброс R_min, R_95, CEP_50."""
    from navedenie.science import monte_carlo

    sc = Scenario(**{k: v for k, v in body.model_dump().items()
                     if k in Scenario.__dataclass_fields__})
    return monte_carlo(sc, n_runs=int(body.n_runs), seed_start=int(body.seed_start))


class CaptureZoneIn(ScenarioIn):
    ranges_m: list[float] | None = None   # сетка дальностей пуска, м (None — 2…12 км)
    target_gs: list[float] | None = None  # сетка перегрузок цели, g (None — 0…20)
    refine: bool = True                   # бисекция границы в скобке «попал→промах»
    refine_tol_m: float = 50.0            # допуск границы по дальности, м
    n_runs: int = 1                       # прогонов на ячейку (>1 — p_hit по Monte-Carlo)
    seed_start: int = 1000                # первый seed серии (i-й прогон — seed_start + i)


@app.post("/api/science/capture-zone")
def capture_zone_endpoint(body: CaptureZoneIn) -> dict[str, Any]:
    """Зона неубегаемого перехвата: отрезок «успеет / не успеет» в плоскости
    «дальность пуска × постоянная нормальная перегрузка цели». Сетка прогонов
    выбранного закона + бисекция обеих скобок связного блока перехвата:
    дальней «попал → промах» и ближней «промах → попал». При n_runs > 1 —
    вероятностный режим: каждая ячейка серия Monte-Carlo, p_hit с ДИ Уилсона,
    бисекция пропускается."""
    from navedenie.science import capture_zone, CZ_RANGES_M, CZ_TARGET_GS

    sc = Scenario(**{k: v for k, v in body.model_dump().items()
                     if k in Scenario.__dataclass_fields__})
    return capture_zone(
        sc,
        ranges_m=tuple(body.ranges_m) if body.ranges_m else CZ_RANGES_M,
        target_gs=tuple(body.target_gs) if body.target_gs else CZ_TARGET_GS,
        refine=bool(body.refine),
        refine_tol_m=float(body.refine_tol_m),
        n_runs=max(1, min(int(body.n_runs), 50)),
        seed_start=int(body.seed_start),
    )


class TrainIn(BaseModel):
    kind: str = "stub"
    episodes: int = 36


@app.post("/api/train")
def train_endpoint(body: TrainIn) -> dict[str, Any]:
    kind = "full" if body.kind == "full" else "stub"
    episodes = max(6, min(int(body.episodes), 80))
    return train_brain(kind=kind, episodes=episodes)


class TrainStartIn(BaseModel):
    kind: str = "stub"
    episodes: int = 36
    lr: float = 0.04


class TrainStepIn(BaseModel):
    kind: str
    ep: int
    lr: float = 0.04


class TrainFinishIn(BaseModel):
    kind: str
    miss_before: float | None = None


@app.post("/api/train/start")
def train_start_endpoint(body: TrainStartIn) -> dict[str, Any]:
    kind = body.kind if body.kind in ("stub", "full", "connectome") else "stub"
    episodes = max(6, min(int(body.episodes), 80))
    return train_start_brain(kind=kind, episodes=episodes, lr=body.lr)


@app.post("/api/train/step")
def train_step_endpoint(body: TrainStepIn) -> dict[str, Any]:
    kind = body.kind if body.kind in ("stub", "full", "connectome") else "stub"
    return train_step_brain(kind=kind, ep=max(0, int(body.ep)), lr=body.lr)


@app.post("/api/train/finish")
def train_finish_endpoint(body: TrainFinishIn) -> dict[str, Any]:
    kind = body.kind if body.kind in ("stub", "full", "connectome") else "stub"
    return train_finish_brain(kind=kind, miss_before=body.miss_before)


class EvolveStartIn(BaseModel):
    kind: str = "stub"
    generations: int = 6
    sigma: float = 0.3
    pop: int = 4
    gain: float = 1.0
    tau_s: float = 0.025


@app.post("/api/train/evolve/start")
def train_evolve_start_endpoint(body: EvolveStartIn) -> dict[str, Any]:
    kind = body.kind if body.kind in ("stub", "full", "connectome") else "stub"
    return train_evolve_start_brain(kind=kind, generations=body.generations, sigma=body.sigma, pop=body.pop, gain=body.gain, tau_s=body.tau_s)


@app.post("/api/train/evolve/step")
def train_evolve_step_endpoint(body: TrainFinishIn) -> dict[str, Any]:
    kind = body.kind if body.kind in ("stub", "full", "connectome") else "stub"
    return train_evolve_step_brain(kind=kind)


@app.post("/api/train/evolve/finish")
def train_evolve_finish_endpoint(body: TrainFinishIn) -> dict[str, Any]:
    kind = body.kind if body.kind in ("stub", "full", "connectome") else "stub"
    return train_evolve_finish_brain(kind=kind)


class SwarmFlyIn(BaseModel):
    kind: str = "bio"
    w: list[float] = []
    gain: float = 1.0
    pn_n: float = 4.0


# ── эксперименты: персистентность в data/experiments (docker-volume) ──────────
_EXP_DIR = Path(__file__).resolve().parent.parent / "data" / "experiments"
_NAME_RE = re.compile(r"^[A-Za-zА-Яа-я0-9 _\-]{1,64}$")


class ExperimentIn(BaseModel):
    name: str
    payload: dict[str, Any]


@app.get("/api/experiments")
def experiments_list() -> dict[str, Any]:
    if not _EXP_DIR.exists():
        return {"experiments": []}
    items = []
    for f in sorted(_EXP_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        items.append(
            {
                "name": f.stem,
                "saved_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds"),
                "size": f.stat().st_size,
            }
        )
    return {"experiments": items}


@app.post("/api/experiments")
def experiment_save(body: ExperimentIn) -> dict[str, Any]:
    if not _NAME_RE.match(body.name):
        raise HTTPException(status_code=422, detail="имя: буквы, цифры, пробел, - _ (до 64)")
    _EXP_DIR.mkdir(parents=True, exist_ok=True)
    path = _EXP_DIR / f"{body.name}.json"
    path.write_text(json.dumps(body.payload, ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "name": body.name}


@app.get("/api/experiments/{name}")
def experiment_get(name: str) -> dict[str, Any]:
    if not _NAME_RE.match(name):
        raise HTTPException(status_code=422, detail="плохое имя эксперимента")
    path = _EXP_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="эксперимент не найден")
    return json.loads(path.read_text(encoding="utf-8"))


@app.delete("/api/experiments/{name}")
def experiment_delete(name: str) -> dict[str, Any]:
    """Удалить снимок эксперимента из архива data/experiments/<name>.json."""
    if not _NAME_RE.match(name):
        raise HTTPException(status_code=422, detail="плохое имя эксперимента")
    path = _EXP_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="эксперимент не найден")
    path.unlink()
    return {"ok": True, "name": name}


# ── библиотека сценариев: персистентность в data/scenarios (docker-volume) ────
_SCN_DIR = Path(__file__).resolve().parent.parent / "data" / "scenarios"


class ScenarioSaveIn(BaseModel):
    name: str
    scenario: ScenarioIn


@app.get("/api/scenarios")
def scenarios_list() -> dict[str, Any]:
    if not _SCN_DIR.exists():
        return {"scenarios": []}
    items = []
    for f in sorted(_SCN_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        items.append(
            {
                "name": f.stem,
                "saved_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )
    return {"scenarios": items}


@app.post("/api/scenarios")
def scenario_save(body: ScenarioSaveIn) -> dict[str, Any]:
    """Сохранить сценарий в серверную библиотеку (переживает перезапуск стенда)."""
    if not _NAME_RE.match(body.name):
        raise HTTPException(status_code=422, detail="имя: буквы, цифры, пробел, - _ (до 64)")
    _SCN_DIR.mkdir(parents=True, exist_ok=True)
    path = _SCN_DIR / f"{body.name}.json"
    path.write_text(json.dumps(body.scenario.model_dump(), ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "name": body.name}


@app.get("/api/scenarios/{name}")
def scenario_get(name: str) -> dict[str, Any]:
    if not _NAME_RE.match(name):
        raise HTTPException(status_code=422, detail="плохое имя сценария")
    path = _SCN_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="сценарий не найден")
    return {"name": name, "scenario": json.loads(path.read_text(encoding="utf-8"))}


@app.delete("/api/scenarios/{name}")
def scenario_delete(name: str) -> dict[str, Any]:
    if not _NAME_RE.match(name):
        raise HTTPException(status_code=422, detail="плохое имя сценария")
    path = _SCN_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="сценарий не найден")
    path.unlink()
    return {"ok": True, "name": name}


class SwarmGenIn(BaseModel):
    scenario: ScenarioIn  # основной сценарий поколения — его траектории идут в повтор
    scenarios: list[ScenarioIn] = []  # вся геометрия поколения; при валидации + эталонное трио
    population: list[SwarmFlyIn] = []
    elite_k: int = 3
    mutation: float = 0.25
    seed: int = 7


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


@app.post("/api/swarm/gen")
def swarm_gen(body: SwarmGenIn) -> dict[str, Any]:
    scen_list = [_sc(s) for s in (body.scenarios or [body.scenario])]
    flies = [fly_from_json(f.model_dump()) for f in body.population]
    if not flies:
        flies = init_population(16, seed=body.seed)
    per_sc = [evaluate_population(sc, flies) for sc in scen_list]
    n = len(flies)
    fits = [_median([per_sc[s][i]["fitness"] for s in range(len(scen_list))]) for i in range(n)]
    misses = [_median([per_sc[s][i]["miss_m"] for s in range(len(scen_list))]) for i in range(n)]
    nxt = evolve(flies, fits, elite_k=max(1, body.elite_k), mutation=body.mutation, seed=body.seed)
    best = min(range(n), key=lambda i: fits[i])
    worst = max(range(n), key=lambda i: fits[i])
    hits = sum(1 for i in range(n) if per_sc[0][i]["hit"])
    bio_ws = [np.asarray(f.w, dtype=float).reshape(-1) for f in flies if f.kind == "bio" and np.size(f.w)]
    diversity = 0.0
    if len(bio_ws) > 1:
        stack = np.stack(bio_ws)
        diversity = float(np.mean(np.std(stack, axis=0)))
    out: dict[str, Any] = {
        "results": [
            {
                "fly": f.to_json(),
                "fitness": fits[i],
                "miss_m": misses[i],
                "hit": per_sc[0][i]["hit"],
                "n_int": per_sc[0][i]["n_int"],
                "traj_m": per_sc[0][i]["traj_m"],
                "traj_t": per_sc[0][i]["traj_t"],
                "tel": per_sc[0][i].get("tel", []),
            }
            for i, f in enumerate(flies)
        ],
        "next_population": [f.to_json() for f in nxt],
        "stats": {
            "best": fits[best],
            "avg": sum(fits) / n,
            "worst": fits[worst],
            "hit_rate": hits / n,
            "diversity": round(diversity, 4),
            "best_idx": best,
        },
    }
    if len(scen_list) > 1:
        canon_miss = [_median([per_sc[s][i]["miss_m"] for s in range(1, len(scen_list))]) for i in range(n)]
        canon_best = min(range(n), key=lambda i: canon_miss[i])
        out["stats"]["canon_best"] = canon_miss[canon_best]
    return out


class EvaderGenIn(BaseModel):
    scenario: ScenarioIn = ScenarioIn()
    laws: list[str] = []  # против каких законов ракеты учиться; пусто — pn/tpn/apn
    population: list[SwarmFlyIn] = []  # геномы учеников; пусто — старт-популяция
    gen: int = 0
    seed: int = 7
    elite_k: int = 3
    mutation: float = 0.25
    exam_every: int = 0  # N>0: на каждом N-м поколении — экзамен чемпиона на эталонной геометрии


@app.post("/api/evader/gen")
def evader_gen_endpoint(body: EvaderGenIn) -> dict[str, Any]:
    """Поколение «школы уклониста»: фронт крутит цикл (популяция туда —
    следующее поколение обратно), как в рое; веса чемпиона применяют
    /api/evader/save."""
    from navedenie.evader_train import EVA_LAWS_DEFAULT, init_evader_population, train_generation

    sc = _sc(body.scenario)
    laws = list(body.laws) or list(EVA_LAWS_DEFAULT)
    pop = (
        [fly_from_json(f.model_dump()) for f in body.population]
        if body.population
        else init_evader_population(12, seed=body.seed)
    )
    try:
        return train_generation(
            sc,
            pop,
            laws=laws,
            gen=body.gen,
            seed=body.seed,
            elite_k=max(1, body.elite_k),
            mutation=float(body.mutation),
            exam_every=max(0, body.exam_every),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


class EvaderSaveIn(BaseModel):
    w: list[float]
    gain: float = 1.0


@app.post("/api/evader/save")
def evader_save_endpoint(body: EvaderSaveIn) -> dict[str, Any]:
    """Применить выученные веса мозга-уклониста: реестр процесса +
    data/weights_evader.npz; дальше duel evader_law='brain' летает на них."""
    from navedenie.brain_store import evader_status, set_evader_circuit

    w = np.asarray(body.w, dtype=np.float64).reshape(-1)
    if w.size != 2 * FEAT_DIM:
        raise HTTPException(status_code=422, detail=f"нужно {2 * FEAT_DIM} весов DN, получено {w.size}")
    c = FlyCircuit(kind="stub", tau_s=0.025, gain=float(np.clip(body.gain, 0.2, 3.0)))
    c.W_dn = np.clip(w.reshape(2, FEAT_DIM), -4.0, 4.0)
    c.trained = True
    set_evader_circuit(c)
    return {"ok": True, "evader": evader_status()}


class QueenGenIn(BaseModel):
    scenario: ScenarioIn = ScenarioIn()
    missile_population: list[SwarmFlyIn] = []  # геномы ракеты; пусто — старт-популяция
    evader_population: list[SwarmFlyIn] = []  # геномы цели; пусто — старт-популяция
    gen: int = 0
    seed: int = 7
    pop: int = 6  # размер стороны при автостарте
    elite_k: int = 2
    mutation: float = 0.25


@app.post("/api/queen/gen")
def queen_gen_endpoint(body: QueenGenIn) -> dict[str, Any]:
    """«Красная королева»: одно поколение совместной эволюции мозг-ракеты против
    мозг-цели. Фронт крутит цикл (обе популяции туда — следующие обратно), как в
    школе уклониста/рое; попусту — обе стороны стартуют врождённым рефлексом+шум."""
    from navedenie.redqueen import init_brain_population, queen_generation

    sc = _sc(body.scenario)
    mpop = (
        [fly_from_json(f.model_dump()) for f in body.missile_population]
        if body.missile_population
        else init_brain_population(max(2, body.pop), seed=body.seed)
    )
    epop = (
        [fly_from_json(f.model_dump()) for f in body.evader_population]
        if body.evader_population
        else init_brain_population(max(2, body.pop), seed=body.seed + 1)
    )
    try:
        return queen_generation(
            sc,
            mpop,
            epop,
            gen=body.gen,
            seed=body.seed,
            elite_k=max(1, body.elite_k),
            mutation=float(body.mutation),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


class QueenApplyIn(BaseModel):
    missile: EvaderSaveIn | None = None
    evader: EvaderSaveIn | None = None


@app.post("/api/queen/apply")
def queen_apply_endpoint(body: QueenApplyIn) -> dict[str, Any]:
    """Посадить чемпионов королевы за штурвал: evader — в weights_evader.npz
    (duel evader_law='brain'), missile — в живой мозг-контейнер схемы 'bio'
    (mode='bio', brain='stub'). Возвращает статусы обеих сторон. Та же ворота,
    которой пользуется фоновое самообучение (redqueen.apply_champions)."""
    from navedenie.redqueen import apply_champions

    def blob(side: EvaderSaveIn | None) -> dict | None:
        return None if side is None else {"w": list(side.w), "gain": float(side.gain)}

    try:
        return apply_champions(missile=blob(body.missile), evader=blob(body.evader))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


class QueenRingSaveIn(BaseModel):
    label: str = ""
    missile: EvaderSaveIn
    evader: EvaderSaveIn


def _fly_blob(side: EvaderSaveIn) -> dict:
    return {"kind": "bio", "w": list(side.w), "gain": float(side.gain)}


@app.get("/api/queen/ring")
def queen_ring_list_endpoint() -> dict[str, Any]:
    """Полка ринга чемпионов: сохранённые мозговые дуэты (ракета+цель)."""
    from navedenie.queen_ring import list_duels

    return {"duels": list_duels()}


@app.post("/api/queen/ring/save")
def queen_ring_save_endpoint(body: QueenRingSaveIn) -> dict[str, Any]:
    """Поставить дуэт чемпионов на полку ринга (data/queen_ring.json)."""
    from navedenie.queen_ring import save_duel

    try:
        return {"ok": True, "duel": save_duel(body.label, _fly_blob(body.missile), _fly_blob(body.evader))}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


class QueenRingDeleteIn(BaseModel):
    id: int


@app.post("/api/queen/ring/delete")
def queen_ring_delete_endpoint(body: QueenRingDeleteIn) -> dict[str, Any]:
    from navedenie.queen_ring import delete_duel

    return {"ok": delete_duel(int(body.id))}


class QueenPairIn(BaseModel):
    """Сборная пара с полки: чью ракету ставим и чьего уклониста."""

    attacker: int
    defender: int


@app.post("/api/queen/ring/pair")
def queen_ring_pair_endpoint(body: QueenPairIn) -> dict[str, Any]:
    """Посадить сборную пару (ракета атакующего дуэта + уклонист обороняющегося)
    в живые веса: дальше «Пуск» летает их между собой. Чужого id на полке — 422."""
    from navedenie.queen_ring import apply_pair

    try:
        return {"ok": True, **apply_pair(int(body.attacker), int(body.defender))}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


class QueenRingBattleIn(BaseModel):
    """Тело сводного боя: сценарий вложенным (как /api/duel), ids — дуэты с
    полки; пусто — все до единого."""

    scenario: ScenarioIn = ScenarioIn()
    ids: list[int] = []


@app.post("/api/queen/ring/battle")
def queen_ring_battle_endpoint(body: QueenRingBattleIn) -> dict[str, Any]:
    """Круговой самобой дуэтов: каждый стреляет по каждому на трёх геометриях,
    таблица рангов — взятия в атаке плюс отражения в защите."""
    from navedenie.queen_ring import list_duels, ring_battle

    shelf = list_duels()
    wanted = {int(i) for i in body.ids}
    duels = [d for d in shelf if int(d.get("id", 0)) in wanted] if wanted else shelf
    try:
        return ring_battle(_sc(body.scenario), duels)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/api/queen/ring/form")
def queen_ring_form_endpoint(limit: int = 2) -> dict[str, Any]:
    """Форма полки: кого называть сильнейшими. Снимок последнего сводного боя
    (ранги после ✕ и нового боя устаревают честно), limit — сколько брать
    родителей следующей кампании. Фронт без этих id наследует «кого последним
    поставили», что не то же самое, что «кто сильнее»."""
    from navedenie.queen_ring import best_duels

    return {"ok": True, **best_duels(max(1, min(4, int(limit))))}


class QueenTrainIn(BaseModel):
    scenario: ScenarioIn = ScenarioIn()
    generations: int = 6
    pop: int = 4
    seed: int = 7
    save_duel: bool = True  # по готовности поставить дуэт чемпионов на полку ринга
    apply: bool = True  # и посадить их за штурвалы живых мозгов
    inherit: list[int] = []  # id дуэтов с полки: начать кампанию с выученных мозгов
    auto_ring: bool = False  # и свести ринг по окончании, не дожидаясь клика

    @field_validator("generations", "pop", "seed")
    @classmethod
    def _in_range(cls, v: int, info) -> int:
        """Границы — в валидаторе, а не в clamp внутри start(): фоновая задача
        по чужому мусорному телу не должна молча писать живые веса."""
        lo, hi = {"generations": (1, 60), "pop": (2, 12), "seed": (-10**9, 10**9)}[info.field_name]
        if not lo <= v <= hi:
            raise ValueError(f"{info.field_name}: надо {lo}…{hi}, получено {v}")
        return v

    @field_validator("inherit")
    @classmethod
    def _heirs(cls, v: list[int]) -> list[int]:
        """Наследников — разумное число: мест для прививки в старте не больше
        pop//2 (популяция ограничена снизу и сверху), а лишний id — это лишний
        поход на полку и молча незадействованный дуэт."""
        if len(v) > 4:
            raise ValueError(f"inherit: не больше 4 дуэтов, получено {len(v)}")
        for i in v:
            if not 1 <= int(i) <= 10**9:
                raise ValueError(f"inherit: id дуэта вне диапазона — {i}")
        return [int(i) for i in v]


@app.post("/api/queen/train")
def queen_train_endpoint(body: QueenTrainIn) -> dict[str, Any]:
    """Запустить самообучение на сервере: цикл поколений Красной королевы в
    фоновом потоке — вкладку можно закрыть. Одна задача за раз: вторая — 409."""
    from navedenie.queen_train import QueenTrainBusy, start

    try:
        return start(
            _sc(body.scenario),
            generations=body.generations,
            pop=body.pop,
            seed=body.seed,
            save_duel=body.save_duel,
            apply=body.apply,
            inherit=body.inherit,
            auto_ring=body.auto_ring,
        )
    except QueenTrainBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:  # наследник не с полки — 422 до того, как поднимется поток
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/api/queen/train")
def queen_train_status_endpoint() -> dict[str, Any]:
    """Снимок фоновой гонки: сколько поколений досчитано, их метрики, чемпионы
    и куда они уже легли. Пустой формы не бывает — задачи может ещё не быть."""
    from navedenie.queen_train import status

    return status()


@app.post("/api/queen/train/stop")
def queen_train_stop_endpoint() -> dict[str, Any]:
    """Просить остановку: текущее поколение досчитывается, дальше — сворачиваемся."""
    from navedenie.queen_train import stop

    return stop()


@app.get("/api/queen/chronicle")
def queen_chronicle_endpoint() -> dict[str, Any]:
    """Хроника войн: доигранные кампании самообучения с кривой взятий по
    поколениям и финалом (что легло в веса и на ринг). Переживает рестарт стенда."""
    from navedenie.queen_chronicle import list_campaigns

    entries = list_campaigns()
    return {"ok": True, "campaigns": entries, "total": len(entries)}


class QueenChronicleDeleteIn(BaseModel):
    id: int


@app.post("/api/queen/chronicle/delete")
def queen_chronicle_delete_endpoint(body: QueenChronicleDeleteIn) -> dict[str, Any]:
    """Вычеркнуть кампанию из хроники; несуществующий id — 404."""
    from navedenie.queen_chronicle import delete_campaign, list_campaigns

    if not delete_campaign(body.id):
        raise HTTPException(status_code=404, detail=f"в хронике нет кампании {body.id}")
    return {"ok": True, "campaigns": list_campaigns()}


@app.websocket("/api/ws/run")
async def ws_run(ws: WebSocket) -> None:
    """Прогон ПОТОКОМ: те же тело и метрики, что у POST /api/run, но кадры
    уходят по мере рождения счёта — сцена трогается через десятки миллисекунд
    после «Пуск», а не после полного проинтегрирования траектории (3–6 с).
    collect крутится в потоке-экзекьюторе; on_frame из него кладёт кадры в
    asyncio-очередь через call_soon_threadsafe; клиент читает, пока не придёт
    'done' с полной метрической пачкой (_run_answer без frames) или 'error'.
    Отвалившийся клиент не убивает расчёт — тот досчитается в пустоту."""
    await ws.accept()
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    try:
        payload = await ws.receive_json()
        if not isinstance(payload, dict):
            raise ValueError("тело стрима — плоский объект сценария, как у /api/run")
        sc = _sc(ScenarioIn(**{k: v for k, v in payload.items() if k in ScenarioIn.model_fields}))
        stride = max(1, int(0.04 / sc.dt))

        def on_frame(i: int, fr, keep: bool) -> None:
            if keep:
                loop.call_soon_threadsafe(q.put_nowait, ("frame", frame_to_dict(fr)))

        def compute() -> None:
            try:
                res = collect(sc, stride=stride, on_frame=on_frame)
                loop.call_soon_threadsafe(q.put_nowait, ("done", res))
            except Exception as exc:  # noqa: BLE01 — из потока в event loop только строкой
                loop.call_soon_threadsafe(q.put_nowait, ("error", str(exc)[:400]))

        loop.run_in_executor(None, compute)
        while True:
            kind, data = await q.get()
            if kind == "frame":
                await ws.send_json({"type": "frame", **data})
            elif kind == "done":
                await ws.send_json({"type": "done", **_run_answer(data, [])})
                break
            else:
                await ws.send_json({"type": "error", "message": data})
                break
    except WebSocketDisconnect:
        return
    except Exception as exc:  # noqa: BLE001
        try:
            await ws.send_json({"type": "error", "message": str(exc)[:400]})
        except Exception:  # noqa: BLE001
            return


# Прод-режим (Docker/офлайн): раздаём собранный фронт web/dist тем же портом.
# Монтируется ПОСЛЕ всех маршрутов, иначе «/» перекроет API.
_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
if (_DIST / "index.html").exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_DIST, html=True), name="static")
