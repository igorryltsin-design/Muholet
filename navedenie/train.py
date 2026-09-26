"""Обучение выхода DN: подражание эталону ПН на признаках контура."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np

from navedenie.brain_store import get_circuit, set_circuit
from navedenie.circuit import FEATURE_SCHEMA_VERSION
from navedenie.engine import collect
from navedenie.pn import ppn_accel
from navedenie.seeker import body_axes, observe
from navedenie.sim import (
    Body,
    G,
    Scenario,
    TARGET_LONG_G_LIMITS,
    clip_accel,
    integrate,
    integrate_target,
    spawn,
    target_accel,
)
from navedenie.swarm import sample_generation_scenario


Progress = Callable[[dict[str, Any]], None]

# ── протокол робастного обучения (документированное распределение сценариев) ──
# Один и тот же протокол используется обучением и матрицей испытаний: train /
# validation / test различаются ТОЛЬКО зёрнами, не процедурой.
PROTOCOL: dict[str, Any] = {
    "feature_schema_version": None,  # заполняется из circuit при импорте протокола
    "v_m": (650.0, 950.0),  # м/с
    "v_t": (150.0, 360.0),  # м/с — фактический начальный модуль скорости цели (все аспекты)
    "range_m": (3500.0, 11000.0),
    "off_axis_m": (50.0, 1400.0),
    "n_target": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    "fov_deg": (9.0, 17.0),
    "seeker_delay_s": (0.0, 0.04),
    "seeker_jitter_s": [0.0, 0.01, 0.02],
    "noise_az_deg": [0.0, 0.0, 1.0, 1.5, 2.5, 3.5],
    "noise_range_m": [0.0, 0.0, 30.0, 60.0],
    "lock_drop_p": [0.0, 0.0, 0.05, 0.1],
    "retina_death_p": [0.0, 0.0, 0.1, 0.2],
    "retina_dropout_p": [0.0, 0.0, 0.05],
    # профили скорости цели внутри эпизода: доли эпизодов
    "speed_modes": {"constant": 0.55, "accelerate": 0.12, "decelerate": 0.12, "pulse": 0.10, "sine": 0.11},
    "target_longitudinal_g": (0.5, 1.5),  # амплитуда продольного ускорения, g
    "target_speed_bounds": {"abs_min": 120.0, "abs_max": 520.0},  # м/с, физика цели
    "target_speed_period_s": (3.0, 8.0),
    "target_speed_phase": (0.0, 1.0),  # доля периода
    "seeds": {
        "train_base": 11,  # эпизоды обучения: 11 + ep*7
        "validation": 1,  # каноническое трио (детерминировано)
        "test_base": 9000,  # held-out матрица скоростей/профилей: 9000 + ячейка
    },
}
PROTOCOL["feature_schema_version"] = FEATURE_SCHEMA_VERSION

# минимальная проверочная матрица (held-out): сетка скоростей × аспекты × профили
TEST_VM_GRID = (650.0, 780.0, 950.0)
TEST_VT_GRID = (150.0, 260.0, 360.0)
TEST_SPEED_MODES = ("constant", "accelerate", "decelerate", "pulse")

# лучшие веса по виду мозга: защита от «забывания» при обучении на сложных эпизодах
_best_cache: dict[str, dict[str, Any]] = {}
# стартовые веса до обучения: обучение не должно ухудшать врождённый рефлекс
_initial_cache: dict[str, dict[str, Any]] = {}


def _weights_snapshot(circuit) -> dict[str, Any]:
    return {
        "w": circuit.W_dn.copy(),
        "pool": circuit.W_pool.copy() if getattr(circuit, "W_pool", None) is not None else None,
        "fx": circuit.W_fx.copy() if getattr(circuit, "W_fx", None) is not None else None,
    }


def _apply_weights(circuit, snap: dict[str, Any]) -> bool:
    """Применить снапшот, если он совместим с мозгом по форме (после пересоздания
    мозга старые снапшоты могли остаться в кеше — их молча пропускаем)."""
    if snap["w"].shape != circuit.W_dn.shape:
        return False
    if snap.get("pool") is not None and getattr(circuit, "W_pool", None) is not None:
        if snap["pool"].shape != circuit.W_pool.shape:
            return False
    if snap.get("fx") is not None and getattr(circuit, "W_fx", None) is not None:
        if snap["fx"].shape != circuit.W_fx.shape:
            return False
    circuit.W_dn = snap["w"].copy()
    if snap.get("pool") is not None and getattr(circuit, "W_pool", None) is not None:
        circuit.W_pool = snap["pool"].copy()
    if snap.get("fx") is not None and getattr(circuit, "W_fx", None) is not None:
        circuit.W_fx = snap["fx"].copy()
    return True


def clear_learned_cache(kind: str) -> None:
    """Забыть снапшоты лучшего/стартового весов: после пересоздания мозга
    они несовместимы по форме и не должны подмешиваться в финал обучения."""
    _best_cache.pop(kind, None)
    _initial_cache.pop(kind, None)


def _remember_best(kind: str, circuit, miss: float, nrms: float) -> None:
    prev = _best_cache.get(kind)
    better = (
        prev is None
        or miss < prev["miss"] - 5.0
        or (abs(miss - prev["miss"]) <= 5.0 and nrms < prev.get("nrms", 1e9))
    )
    if better:
        _best_cache[kind] = {"miss": float(miss), "nrms": float(nrms), **_weights_snapshot(circuit)}


def _restore_best(kind: str, circuit) -> float | None:
    snap = _best_cache.get(kind)
    if snap is None:
        return None
    _apply_weights(circuit, snap)
    return snap["miss"]


def _teacher_py(a_pn: np.ndarray, v_m: np.ndarray, n_max: float) -> np.ndarray:
    _x, y, z = body_axes(v_m)
    scale = n_max * G + 1e-9
    return np.clip(np.array([float(np.dot(a_pn, z)) / scale, float(np.dot(a_pn, y)) / scale]), -1.0, 1.0)


def _median_or_none(xs: list[float | None]) -> float | None:
    vals = sorted(x for x in xs if x is not None)
    if not vals:
        return None
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def _rollout_update(circuit, sc: Scenario, lr: float, telemetry: list | None = None, t_cap: float = 12.0, feats: list | None = None, rho_trace: list | None = None, teacher_mode: str = "full_command") -> dict[str, Any]:
    """Эпизод: муха летит сама (учится на своих ошибках), призрак-ПН летит рядом —
    метрики «похожести на oracle-закон» и его времени наведения.

    Зрение — как в движке: фовеальная сетчатка bio_fov_deg, декодирование из
    изображения, постоянная карта отказов + дропаут, очередь задержки/джиттера.
    Перехват — межшаговое пересечение сферы БЧ; промах — непрерывный CPA.
    feats — если список передан, в него складываются векторы признаков контура
    каждого шага (для flight-domain фитинга полиномиального суррогата).
    rho_trace — если передан, в него складывается rho каждого шага (для абляций).
    teacher_mode — режим teacher-сигнала:
        full_command — точная нормированная команда oracle-ПН (умолчание);
        direction_only — только направление, без навязывания амплитуды;
        clipped_coarse — грубо ограниченная амплитуда."""
    from navedenie.seeker import ObservationBuffer, dead_mask_for, observe
    from navedenie.sim import step_encounter

    missile, target = spawn(sc)
    ghost = Body(missile.p.copy(), missile.v.copy())
    circuit.reset()
    prev: dict | None = None
    cpa = 1e9
    t = 0.0
    dt = sc.dt
    half_fov = np.deg2rad(max(getattr(sc, "bio_fov_deg", 165.0), 2.0)) / 2.0
    steps = 0
    t_guide: float | None = None
    t_ref: float | None = None
    dev_sum = 0.0
    range_sum = 0.0
    dev_n = 0
    effort = 0.0  # ∫|a_cmd|/g dt, g·с
    n_peak = 0.0
    lock_time = 0.0
    noise_rng = np.random.default_rng(sc.seed * 977 + 13)
    dead_mask = dead_mask_for(sc.retina_death_p, noise_rng)
    buffer = ObservationBuffer(sc.seeker_delay_s, sc.seeker_jitter_s, dt, noise_rng)
    while t < min(sc.t_max, t_cap):
        r = target.p - missile.p
        rng_n = float(np.linalg.norm(r))
        obs = observe(
            r,
            missile.v,
            prev,
            half_fov=half_fov,
            dt_s=dt,
            noise_az=np.deg2rad(sc.noise_az_deg),
            noise_range=sc.noise_range_m,
            lock_drop_p=sc.lock_drop_p,
            dead_mask=dead_mask,
            dropout_p=sc.retina_dropout_p,
            rng=noise_rng,
            brightness=getattr(sc, "target_brightness", 1.0),
        )
        obs["retina_alive"] = 1.0 - sc.retina_death_p
        prev = obs
        buffer.push(t, obs)
        delayed = buffer.sample(t)
        if rho_trace is not None:
            rho_trace.append(float(delayed.get('rho', 0.0)))
        circuit.step(delayed, dt)
        a_pn = ppn_accel(r, missile.v, target.v, sc.pn_n, sc.n_max)
        a_ghost = ppn_accel(target.p - ghost.p, ghost.v, target.v, sc.pn_n, sc.n_max)
        teacher = _teacher_py(a_pn, missile.v, sc.n_max)
        # teacher_mode: full_command (unchanged), direction_only, clipped_coarse
        if teacher_mode == "direction_only":
            tn = float(np.linalg.norm(teacher))
            if tn > 1e-6:
                teacher = teacher / tn
        elif teacher_mode == "clipped_coarse":
            teacher = np.clip(teacher, -0.5, 0.5)
        if feats is not None:
            # у коннектома вектор признаков в circuit.feat, у схемы/полного — в state.feat
            feat_vec = getattr(circuit, "feat", None)
            if feat_vec is None:
                feat_vec = getattr(getattr(circuit, "state", None), "feat", None)
            if feat_vec is not None and np.size(feat_vec):
                feats.append(np.asarray(feat_vec, dtype=np.float64).copy())
        # «полный» и «коннектом» дообучают скрытые слои (пластичность проводки), схема — только выход
        lr_hidden = lr * 0.3 if getattr(circuit, "kind", "") in ("full", "connectome") else 0.0
        circuit.learn_step(teacher, lr, lr_hidden)
        a_bio = circuit.accel_cmd(missile.v, sc.n_max)
        a_cmd = a_bio if delayed["lock"] else np.zeros(3)
        a_cmd = clip_accel(a_cmd, sc.n_max)
        # метрики перегрузки и захвата — для «цены» манёвра при обучении
        gn = float(np.linalg.norm(a_cmd)) / G
        effort += gn * dt
        n_peak = max(n_peak, gn)
        if delayed["lock"]:
            lock_time += dt
        # интеграция (|v| ракеты сохраняется; цель — с профилем скорости) и встреча
        p_m0, p_t0 = missile.p.copy(), target.p.copy()
        integrate_target(target, sc, t, dt)
        integrate(missile, a_cmd, dt)
        integrate(ghost, a_ghost, dt)
        enc = step_encounter(p_m0, missile.p, p_t0, target.p, sc.kill_radius_m)
        cpa = min(cpa, enc["cpa"])
        dev_diff = p_m0 - ghost.p
        dev_sum += float(np.dot(dev_diff, dev_diff))
        range_sum += float(np.linalg.norm(target.p - missile.p))
        dev_n += 1
        nrms_now = float(np.sqrt(dev_sum / dev_n) / (range_sum / dev_n)) if dev_n and range_sum > 0 else 0.0
        _remember_best(getattr(circuit, "kind", "stub"), circuit, float(cpa), nrms_now)
        if t_ref is None and float(np.linalg.norm(target.p - ghost.p)) < sc.kill_radius_m:
            t_ref = t
        if telemetry is not None and steps % 5 == 0:
            # у коннектома состояние в своих векторах, у схемы/полного — state.dn
            dn_vec = getattr(circuit.state, "dn", getattr(circuit, "_dn", np.zeros(2)))
            telemetry.append(
                {
                    "t": float(t),
                    "az": float(delayed["az"]),
                    "el": float(delayed["el"]),
                    "lock": bool(delayed["lock"]),
                    "size": float(delayed["size"]),
                    "size_dot": float(delayed["size_dot"]),
                    "az_dot": float(delayed["az_dot"]),
                    "el_dot": float(delayed["el_dot"]),
                    "theta": float(delayed["theta"]),
                    "rho": float(delayed["rho"]),
                    "tau_contact": float(delayed["tau_contact"]),
                    "pitch": float(dn_vec[0]),
                    "yaw": float(dn_vec[1]),
                    "n_req": float(np.linalg.norm(a_cmd) / G),
                    "miss": float(cpa),
                    "rng": float(rng_n),
                }
            )
        if enc["hit"]:
            t_guide = t + float(enc["alpha_in"]) * dt
            break
        if t > 0.5 and float(-np.dot(r, target.v - missile.v) / (rng_n + 1e-9)) < 0 and rng_n > cpa + 80:
            break
        t += dt
        steps += 1
    duration = max(steps * dt, 1e-9)
    return {
        "miss": float(cpa),
        "cpa_m": float(cpa),
        "hit": bool(t_guide is not None),
        "t_guide": t_guide,
        "ref_dev": float(np.sqrt(dev_sum / max(dev_n, 1))),
        "nrms": float(np.sqrt(dev_sum / max(dev_n, 1)) / (range_sum / max(dev_n, 1))) if dev_n and range_sum > 0 else 0.0,
        "t_ref": t_ref,
        "control_effort_gs": float(effort),
        "n_avg": float(effort / duration),  # среднее нормальное ускорение, g (на ВРЕМЯ, не на шаги)
        "n_peak": float(n_peak),
        "lock_time_s": float(lock_time),
        "lock_frac": float(lock_time / duration),
    }


def evaluate(kind: str, gain: float = 1.15) -> dict[str, float]:
    circuit = get_circuit(kind, gain=gain)
    aspects = ("head-on", "beam", "tail-chase")
    misses: list[float] = []
    hits = 0
    for aspect in aspects:
        sc = Scenario(aspect=aspect, mode="bio", t_max=14, dt=0.02, circuit_gain=gain, seed=3)
        # collect создаёт свой контур — считаем через _rollout без обновления
        sc.mode = "bio"
        result = collect(sc, stride=10_000)
        misses.append(result.miss_m)
        hits += int(result.hit)
    return {"miss": float(np.median(misses)), "hits": hits, "n": len(aspects)}


def _eval_canonical(circuit) -> dict[str, float | None]:
    """Медианные метрики на эталонном трио: промах, время наведения, отклонение от ПН."""
    rows = [
        _rollout_update(circuit, Scenario(aspect=a, mode="bio", t_max=16, dt=0.02, circuit_gain=circuit.gain), lr=0.0, t_cap=16.0)
        for a in ("head-on", "beam", "tail-chase")
    ]
    return {
        "miss": float(np.median([r["miss"] for r in rows])),
        "hit_rate": sum(1 for r in rows if r["hit"]) / len(rows),
        "t_guide": _median_or_none([r["t_guide"] for r in rows]),
        "ref_dev": float(np.median([r["ref_dev"] for r in rows])),
    }


def _episode_scenario(ep: int, gain: float, maneuver: str | None = None, *, rng: np.random.Generator | None = None) -> Scenario:
    """Богатый учебный сценарий по протоколу PROTOCOL: все манёвры, широкие вилки
    дальностей/скоростей, случайные шумы измерителей и отказы сетчатки, профили
    скорости цели (55% constant + разгон/торможение/импульс/синус).
    maneuver — фиксированный манёвр (матрица переносимости: обучение на одном);
    жребий всё равно разыгрывается, чтобы случайная геометрия не съехала.
    rng — внешний генератор (воспроизводимость батчей); None — детерминизм по ep."""
    g = rng if rng is not None else np.random.default_rng(PROTOCOL["seeds"]["train_base"] + ep * 7)
    maneuvers = ("straight", "turn", "weave", "weave_var", "break", "scissors", "dive", "combo")
    v_t = float(g.uniform(*PROTOCOL["v_t"]))
    sc = Scenario(
        aspect=("head-on", "beam", "tail-chase")[ep % 3],
        mode="bio",
        t_max=12,
        dt=0.02,
        off_axis_m=float(g.uniform(*PROTOCOL["off_axis_m"])),
        range_m=float(g.uniform(*PROTOCOL["range_m"])),
        v_m=float(g.uniform(*PROTOCOL["v_m"])),
        v_t=v_t,
        n_target=float(g.choice(PROTOCOL["n_target"])),
        maneuver=maneuver or maneuvers[int(g.integers(0, len(maneuvers)))],  # type: ignore[arg-type]
        fov_deg=float(g.uniform(*PROTOCOL["fov_deg"])),
        seeker_delay_s=float(g.uniform(*PROTOCOL["seeker_delay_s"])),
        seeker_jitter_s=float(g.choice(PROTOCOL["seeker_jitter_s"])),
        noise_az_deg=float(g.choice(PROTOCOL["noise_az_deg"])),
        noise_range_m=float(g.choice(PROTOCOL["noise_range_m"])),
        lock_drop_p=float(g.choice(PROTOCOL["lock_drop_p"])),
        retina_death_p=float(g.choice(PROTOCOL["retina_death_p"])),
        retina_dropout_p=float(g.choice(PROTOCOL["retina_dropout_p"])),
        circuit_gain=gain,
    )
    return _apply_speed_profile(sc, g)


def _apply_speed_profile(sc: Scenario, g: np.random.Generator) -> Scenario:
    """Назначить профиль скорости цели по распределению PROTOCOL['speed_modes']."""
    from dataclasses import replace

    modes, weights = zip(*PROTOCOL["speed_modes"].items())
    mode = modes[int(g.choice(len(modes), p=np.asarray(weights) / sum(weights)))]
    if mode == "constant":
        return replace(sc, target_speed_mode="constant")
    period = float(g.uniform(*PROTOCOL["target_speed_period_s"]))
    lo = max(PROTOCOL["target_speed_bounds"]["abs_min"], sc.v_t * 0.55)
    hi = min(PROTOCOL["target_speed_bounds"]["abs_max"], sc.v_t * 1.5)
    return replace(
        sc,
        target_speed_mode=mode,  # type: ignore[arg-type]
        target_longitudinal_g=float(g.uniform(*PROTOCOL["target_longitudinal_g"])),
        target_speed_min=float(lo),
        target_speed_max=float(hi),
        target_speed_period_s=period,
        target_speed_phase=float(g.uniform(*PROTOCOL["target_speed_phase"])) * period,
    )


def train_start(kind: str = "stub", episodes: int = 36, lr: float = 0.04) -> dict[str, Any]:
    """Начало обучения: оценка мозга на эталоне до тренировки.

    Стартовые веса запоминаются как кандидат финала: обучение не должно
    ухудшать врождённый рефлекс."""
    circuit = get_circuit(kind)
    miss_before = _eval_canonical(circuit)
    _initial_cache[kind] = {"miss": float(miss_before["miss"]), **_weights_snapshot(circuit)}
    return {
        "kind": circuit.kind,
        "n_cells": circuit.n_cells,
        "episodes": int(episodes),
        "miss_before": miss_before["miss"],
        "metrics_before": miss_before,
        "lr": lr,
        # телеметрия этапа (P1): имитация учителя — этап warm_start
        "training_stage": "warm_start",
        "teacher_active": True,
        "teacher_weight": 1.0,
        "scenario_split": "train",
    }


def train_step(kind: str = "stub", ep: int = 0, lr: float = 0.04) -> dict[str, Any]:
    """Один эпизод обучения: подражание ПН + телеметрия для анимации."""
    circuit = get_circuit(kind)
    sc = _episode_scenario(ep, circuit.gain)
    tel: list[dict] = []
    out = _rollout_update(circuit, sc, lr=lr, telemetry=tel)
    return {
        "ep": int(ep),
        "miss": float(out["miss"]),
        "hit": bool(out["hit"]),
        "t_guide": out["t_guide"],
        "ref_dev": out["ref_dev"],
        "nrms": out["nrms"],
        "t_ref": out["t_ref"],
        "n_avg": out["n_avg"],
        "n_peak": out["n_peak"],
        "lock_frac": out["lock_frac"],
        "tel": tel,
        "w": circuit.W_dn.tolist(),
        # телеметрия этапа: веса на этом шаге менял teacher (если lr > 0)
        "training_stage": "warm_start",
        "teacher_active": lr > 0,
        "teacher_weight": float(lr),
        "weights_updated_by": "teacher" if lr > 0 else "none",
        "scenario_split": "train",
    }


def train_finish(kind: str = "stub", miss_before: float | None = None) -> dict[str, Any]:
    """Финал: выбираем лучшие веса по эталонному трио из трёх кандидатов —
    текущие после обучения, лучший снапшот эпизодов и стартовые (врождённый рефлекс)."""
    circuit = get_circuit(kind)
    after = _eval_canonical(circuit)
    best_miss = after["miss"]
    current = _weights_snapshot(circuit)
    best_snap = current
    restored = False
    candidates: list[tuple[str, dict[str, Any]]] = [("current", current)]
    if kind in _best_cache:
        candidates.append(("best_episode", _best_cache[kind]))
    if kind in _initial_cache:
        candidates.append(("initial", _initial_cache[kind]))
    for name, cand in candidates:
        _apply_weights(circuit, cand)
        cand_metrics = _eval_canonical(circuit)
        if cand_metrics["miss"] < best_miss:
            best_miss = cand_metrics["miss"]
            after = cand_metrics
            best_snap = cand
            restored = name != "current"
    _apply_weights(circuit, best_snap)
    circuit.trained = True
    set_circuit(circuit)
    return {
        "kind": circuit.kind,
        "n_cells": circuit.n_cells,
        "miss_before": miss_before,
        "miss_after": after["miss"],
        "hit_rate_before": None,
        "hit_rate_after": after["hit_rate"],
        "t_guide_after": after["t_guide"],
        "ref_dev_after": after["ref_dev"],
        "restored_best": restored,
        "trained": True,
    }


def train(*, kind: str = "stub", episodes: int = 36, lr: float = 0.04, progress: Progress | None = None) -> dict[str, Any]:
    circuit = get_circuit(kind)
    aspects = ("head-on", "beam", "tail-chase")
    before = [
        _rollout_update(circuit, Scenario(aspect=aspect, mode="bio", t_max=12, dt=0.02, circuit_gain=circuit.gain), lr=0.0)
        for aspect in aspects
    ]
    miss_before = float(np.median([r["miss"] for r in before]))
    history: list[dict[str, Any]] = []
    for ep in range(episodes):
        # единый протокол DR с профилями скорости (детерминизм по номеру эпизода)
        sc = _episode_scenario(ep, circuit.gain)
        out = _rollout_update(circuit, sc, lr=lr)
        history.append({k: out[k] for k in ("miss", "hit", "t_guide", "ref_dev")})
        if progress and (ep % 3 == 0 or ep == episodes - 1):
            progress(
                {
                    "episode": ep + 1,
                    "episodes": episodes,
                    "miss": out["miss"],
                    "kind": circuit.kind,
                    "n_cells": circuit.n_cells,
                }
            )
    after = [
        _rollout_update(circuit, Scenario(aspect=aspect, mode="bio", t_max=12, dt=0.02, circuit_gain=circuit.gain), lr=0.0)
        for aspect in aspects
    ]
    miss_after = float(np.median([r["miss"] for r in after]))
    circuit.trained = True
    set_circuit(circuit)
    return {
        "kind": circuit.kind,
        "n_cells": circuit.n_cells,
        "episodes": episodes,
        "protocol": PROTOCOL,
        "miss_before": miss_before,
        "miss_after": miss_after,
        "hit_rate_after": sum(1 for r in after if r["hit"]) / len(after),
        "t_guide_after": _median_or_none([r["t_guide"] for r in after]),
        "ref_dev_before": float(np.median([r["ref_dev"] for r in before])),
        "ref_dev_after": float(np.median([r["ref_dev"] for r in after])),
        "history": history[-12:],
        "trained": True,
    }


# ─── обучение «по результату»: эволюционная доводка выхода до попадания ──────
# Рой попадает, потому что оптимизирует промах напрямую. Здесь тот же принцип,
# применённый к весам выхода НАСТОЯЩЕГО мозга: CEM (кросс-энтропийный метод) с
# малой популяцией по W_dn. Скрытые слои не трогаются — их динамика от W_dn не
# зависит, так что мутации выхода безопасны и обратимы.

_evolve_state: dict[str, dict[str, Any]] = {}
_EVOLVE_ASPECTS = ("head-on", "weave", "tail-chase")


def evolve_start(kind: str = "stub", generations: int = 6, sigma: float = 0.3, pop: int = 4, gain: float = 1.0, tau_s: float = 0.025, batch_scenarios: list[Scenario] | None = None) -> dict[str, Any]:
    """Начало доводки: фиксируем базу (текущие веса) и стартовый промах по трио.
    batch_scenarios — внешний батч (коэволюция: опаснейшие цели) вместо случайного пула."""
    circuit = get_circuit(kind, tau_s=tau_s, gain=gain)
    before = _eval_canonical(circuit)
    _evolve_state[kind] = {
        "gen": 0,
        "generations": max(2, int(generations)),
        "pop": max(2, int(pop)),
        "sigma": float(np.clip(sigma, 0.05, 1.0)),
        "mean_w": circuit.W_dn.copy(),  # центр распределения кандидатов
        "mean_fx": circuit.W_fx.copy() if hasattr(circuit, "W_fx") else None,
        "best_miss": float(before["miss"]),
        "best_w": circuit.W_dn.copy(),
        "best_fx": circuit.W_fx.copy() if hasattr(circuit, "W_fx") else None,
        "start_snapshot": _weights_snapshot(circuit),
        "scen_i": 0,
        "gain": float(gain),
        "tau_s": float(tau_s),
        "batch_scenarios": list(batch_scenarios) if batch_scenarios else None,
        "rng": np.random.default_rng(1000 + len(_evolve_state)),
    }
    return {
        "kind": circuit.kind,
        "n_cells": circuit.n_cells,
        "generations": int(generations),
        "sigma": float(sigma),
        "miss_before": float(before["miss"]),
    }


def evolve_step(kind: str = "stub") -> dict[str, Any]:
    """Одно поколение CEM: популяция кандидатов вокруг среднего, отбор элиты."""
    st = _evolve_state.get(kind)
    if st is None:
        raise ValueError("evolve не запущен: сначала /api/train/evolve/start")
    circuit = get_circuit(kind, tau_s=st.get("tau_s", 0.025), gain=st.get("gain", 1.0))
    rng: np.random.Generator = st["rng"]

    if st.get("batch_scenarios"):
        # внешний батч (коэволюция: опаснейшие цели против текущего мозга)
        batch = list(st["batch_scenarios"])
    else:
        # РОБАСТНАЯ доводка: каждый кандидат — на ЧЕТЫРЁХ случайных сценариях из пула роя
        # (все аспекты, манёвры от простых к сложным, разные дальности и перегрузки цели).
        # Фитнес — МЕДИАННЫЙ промах батча: устойчив к одному «выигрышному» сценарию.
        # ВАЖНО: шаг интегрирования чередуется (0.02/0.01/0.005) — замкнутый контур на
        # декодированных углах чувствителен к дискретизации (ступенчатость фовеи);
        # без вариации dt мозг переобучается под один шаг и мажет на остальных.
        base_sc = Scenario(aspect="head-on", mode="bio", t_max=10, dt=0.02, circuit_gain=circuit.gain)
        dts = (0.02, 0.01, 0.02, 0.005)
        batch = [
            replace(
                sample_generation_scenario(base_sc, st["gen"] * 4 + k, 500 + st["gen"] * 17 + k),
                noise_az_deg=float((k % 2) * 1.2),
                dt=dts[k % len(dts)],
            )
            for k in range(4)
        ]

    results: list[tuple[float, dict[str, Any], np.ndarray, np.ndarray | None]] = []
    for pi in range(st["pop"]):
        if pi == 0 and st["gen"] > 0:
            # элитизм: чемпион прошлых поколений летит без мутаций — поколение
            # в принципе не может стать хуже уже найденного лучшего
            cand_dn = st["best_w"]
            cand_fx = st.get("best_fx")
        else:
            cand_dn = np.clip(st["mean_w"] + rng.normal(0, st["sigma"], circuit.W_dn.shape), -4.0, 4.0)
            cand_fx = None
            if st.get("mean_fx") is not None:
                # каналы признаков дообучаются вдвое мягче — они задают «зрение» мозга
                cand_fx = np.clip(st["mean_fx"] + rng.normal(0, st["sigma"] * 0.5, st["mean_fx"].shape), -2.0, 2.0)
        if cand_fx is not None:
            circuit.W_fx = cand_fx
        circuit.W_dn = cand_dn
        circuit.reset()
        rs = [_rollout_update(circuit, sc, lr=0.0) for sc in batch]
        misses = [r["miss"] for r in rs]
        # фитнес = промах (медиана батча), лёгкий штраф за перегрузку — только
        # чтобы отбор не поощрял «рваные» траектории с бесконечной перегрузкой
        fitness = float(np.median(misses)) + 0.05 * float(np.mean([r.get("n_avg", 0.0) for r in rs]))
        hit_share = float(np.mean([r["hit"] for r in rs]))
        results.append((fitness, {"miss": float(np.median(misses)), "hit": hit_share >= 0.5}, cand_dn, cand_fx))
    results.sort(key=lambda x: x[0])

    # CEM-обновление: новое среднее — по элите; σ адаптивно (прогресс сжимает, застой расширяет)
    elite_w = np.stack([w for _, _, w, _fx in results[:2]])
    st["mean_w"] = elite_w.mean(axis=0)
    if st.get("mean_fx") is not None:
        st["mean_fx"] = np.stack([fx for _, _, _w, fx in results[:2] if fx is not None]).mean(axis=0)
    gen_best_fit, gen_best_r, gen_best_w, gen_best_fx = results[0]
    gen_best_miss = float(gen_best_r["miss"])
    if gen_best_miss < st["best_miss"] - 0.5:
        st["best_miss"] = gen_best_miss
        st["best_w"] = gen_best_w.copy()
        if gen_best_fx is not None:
            st["best_fx"] = gen_best_fx.copy()
        st["sigma"] = float(np.clip(st["sigma"] * 0.9, 0.08, 1.0))
    else:
        # застой: σ растёт, но НЕ выше 0.6 — взрыв σ (до 1.0) уничтожал поиск на
        # большом мозге: мутанты становятся не «вариантами», а шумом
        st["sigma"] = float(np.clip(st["sigma"] * 1.15, 0.08, 0.6))
    st["gen"] += 1

    return {
        "gen": int(st["gen"]),
        "generations": st["generations"],
        "miss_best_gen": gen_best_miss,
        "miss_best_overall": float(st["best_miss"]),
        "hit_rate": float(np.mean([r["hit"] for _, r, _w, _fx in results])),
        "sigma": st["sigma"],
        # телеметрия этапа: кандидаты оцениваются только полётом (lr=0), teacher выключен
        "training_stage": "outcome_evolution",
        "teacher_active": False,
        "weights_updated_by": "outcome_fitness",
        "scenario_split": "train",
    }


def evolve_finish(kind: str = "stub") -> dict[str, Any]:
    """Финал: из кандидатов (текущие/лучшие поколения/стартовые) оставляем лучший по трио и сохраняем."""
    st = _evolve_state.pop(kind, None)
    circuit = get_circuit(kind, tau_s=(st or {}).get("tau_s", 0.025), gain=(st or {}).get("gain", 1.0))
    if st is None:
        raise ValueError("evolve не запущен")

    candidates: list[tuple[str, dict[str, Any]]] = [("current", _weights_snapshot(circuit))]
    if st.get("best_w") is not None:
        candidates.append(("best_generation", {"w": st["best_w"], "pool": st.get("best_pool"), "fx": st.get("best_fx")}))
    if st.get("start_snapshot"):
        candidates.append(("start", st["start_snapshot"]))
    best_name: str | None = None
    best_miss: float | None = None
    best_snap: dict[str, Any] | None = None
    for name, snap in candidates:
        circuit.W_dn = snap["w"].copy()
        if snap.get("pool") is not None and getattr(circuit, "W_pool", None) is not None:
            circuit.W_pool = snap["pool"].copy()
        if snap.get("fx") is not None and hasattr(circuit, "W_fx"):
            circuit.W_fx = snap["fx"].copy()
        circuit.reset()
        r = _eval_canonical(circuit)
        if best_miss is None or r["miss"] < best_miss:
            best_miss = float(r["miss"])
            best_name = name
            best_snap = snap
    if best_snap is not None:
        circuit.W_dn = best_snap["w"].copy()
        if best_snap.get("pool") is not None and getattr(circuit, "W_pool", None) is not None:
            circuit.W_pool = best_snap["pool"].copy()
        if best_snap.get("fx") is not None and hasattr(circuit, "W_fx"):
            circuit.W_fx = best_snap["fx"].copy()
        circuit.reset()
    after = _eval_canonical(circuit)
    circuit.trained = True
    set_circuit(circuit)
    _evolve_state.pop(kind, None)
    return {
        "kind": circuit.kind,
        "n_cells": circuit.n_cells,
        "miss_after": float(after["miss"]),
        "hit_rate_after": after["hit_rate"],
        "t_guide_after": after["t_guide"],
        "ref_dev_after": after["ref_dev"],
        "restored_best": best_name,
        "trained": True,
        # телеметрия этапа: доводка «по результату» — teacher выключен
        "training_stage": "outcome_evolution",
        "teacher_active": False,
        "weights_updated_by": "outcome_fitness",
        "fitness_kind": "median_miss_batch",  # legacy-режим UI; научный pipeline — law.py (лексикографический)
    }
