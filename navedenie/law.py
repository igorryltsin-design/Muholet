"""Двухэтапный конвейер «закон наведения» (law discovery).

Вопрос исследования: во что превращается грубо выученная ПН после СВОБОДНОЙ
эволюционной оптимизации по физическому результату перехвата, если BIO получает
только сенсорные признаки (без истинной дальности, Vc, t_go и абсолютного времени)?

Этап A — coarse warm start (имитация, DAgger-подобная):
  oracle-ПН даёт teacher-команду; BIO летит по своему сенсорному контуру;
  на посещённых BIO состояниях выход подтягивается к teacher. Режимы teacher:
  full_command | direction_only | clipped_coarse. Основной исследовательский —
  direction_only: этап A не должен фиксировать постоянный коэффициент ПН.
Этап B — outcome evolution (свободная доводка):
  teacher полностью отключён (learn_step не вызывается), в fitness НЕТ ни N_eff,
  ни сходства с ПН, ни коэффициентов суррогата — только физический результат:
  перехват → робастный промах → цена манёвра. Лексикографическое сравнение
  кандидатов (никаких произвольных сумм).

Честность выбора: чемпион выбирается по фиксированному validation-батчу;
held-out test не участвует ни в одном решении (только финальная оценка).

Диагностика (постфактум, истинная геометрия) — в navedenie.diagnostics;
сюда она не входит по построению: EpisodeMetrics не содержит ни одного поля,
вычисленного из oracle-геометрии, кроме самого физического факта встречи.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from navedenie.circuit import FEATURE_SCHEMA_VERSION, ConnectomeCircuit, FlyCircuit
from navedenie.pn import ppn_accel
from navedenie.seeker import ObservationBuffer, body_axes, dead_mask_for, observe
from navedenie.sim import (
    G,
    Scenario,
    clip_accel,
    integrate,
    integrate_target,
    spawn,
    step_encounter,
)
from navedenie.swarm import sample_generation_scenario
from navedenie.train import PROTOCOL, _episode_scenario

Progress = Callable[[dict[str, Any]], None]

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "law_discovery"

# этапы обучения (телеметрия pipeline)
STAGE_WARM_START = "warm_start"
STAGE_OUTCOME = "outcome_evolution"


# ── метрики эпизода: ТОЛЬКО физический результат, без oracle-подсказок ───────


@dataclass
class EpisodeMetrics:
    """Результат одного эпизода. Никаких N_eff/сходства с ПН — это fitness-носитель.

    geometric_cpa_m — истинное минимальное расстояние при диагностическом
    «теневом продолжении» (после входа в сферу БЧ оба летят по инерции) —
    НЕ зависит от kill radius;
    trigger_range_m — дальность в момент входа в сферу БЧ (срабатывание БЧ)."""

    hit: bool
    t_hit: float | None
    trigger_range_m: float
    geometric_cpa_m: float
    effort_gs: float
    sat_frac: float
    lock_frac: float
    jerk: float  # средняя скорость смены команды, доля n_max·g за секунду, 1/с
    duration_s: float
    scenario_split: str = "train"  # train | validation | test — откуда сценарий
    v_m: float = 0.0
    v_t: float = 0.0
    aspect: str = ""
    speed_mode: str = "constant"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatchMetrics:
    """Агрегаты батча эпизодов (paired: у всех кандидатов батч один и тот же)."""

    episodes: list[EpisodeMetrics]

    @property
    def hit_rate(self) -> float:
        return float(np.mean([e.hit for e in self.episodes])) if self.episodes else 0.0

    def _vals(self, f: Callable[[EpisodeMetrics], float]) -> list[float]:
        return [f(e) for e in self.episodes]

    @property
    def cpa_median(self) -> float:
        return float(np.median(self._vals(lambda e: e.geometric_cpa_m))) if self.episodes else 1e9

    @property
    def cpa_p90(self) -> float:
        return float(np.percentile(self._vals(lambda e: e.geometric_cpa_m), 90)) if self.episodes else 1e9

    @property
    def cpa_cvar90(self) -> float:
        """CVaR худших 10%: среднее по худшему децилю промаха (робастность к хвосту)."""
        xs = sorted(self._vals(lambda e: e.geometric_cpa_m))
        if not xs:
            return 1e9
        k = max(1, int(np.ceil(0.1 * len(xs))))
        return float(np.mean(xs[-k:]))

    @property
    def effort_median(self) -> float:
        return float(np.median(self._vals(lambda e: e.effort_gs))) if self.episodes else 0.0

    @property
    def sat_median(self) -> float:
        return float(np.median(self._vals(lambda e: e.sat_frac))) if self.episodes else 0.0

    @property
    def lock_frac_min(self) -> float:
        return float(np.min(self._vals(lambda e: e.lock_frac))) if self.episodes else 0.0

    @property
    def jerk_median(self) -> float:
        return float(np.median(self._vals(lambda e: e.jerk))) if self.episodes else 0.0

    @property
    def t_hit_median(self) -> float | None:
        xs = [e.t_hit for e in self.episodes if e.t_hit is not None]
        return float(np.median(xs)) if xs else None

    def components(self) -> dict[str, Any]:
        """Все компоненты fitness РАЗДЕЛЬНО — свести в одно число запрещено."""
        return {
            "hit_rate": round(self.hit_rate, 4),
            "cpa_median_m": round(self.cpa_median, 2),
            "cpa_p90_m": round(self.cpa_p90, 2),
            "cpa_cvar90_m": round(self.cpa_cvar90, 2),
            "effort_median_gs": round(self.effort_median, 3),
            "sat_frac_median": round(self.sat_median, 4),
            "lock_frac_min": round(self.lock_frac_min, 4),
            "jerk_median": round(self.jerk_median, 4),
            "t_hit_median_s": round(self.t_hit_median, 3) if self.t_hit_median is not None else None,
            "n_episodes": len(self.episodes),
        }

    def rank_key(self) -> tuple:
        """Лексикографический ключ: МЕНЬШЕ — ЛУЧШЕ. Порядок приоритетов:
        1) максимум вероятности перехвата (−hit_rate);
        2) робастный промах: медиана, затем p90, затем CVaR90 geometric CPA;
        3) цена манёвра: интеграл перегрузки;
        4) доля насыщения;
        5) потеря захвата (−min lock_frac);
        6) резкость управления (jerk);
        7) время перехвата — вторичный критерий после надёжности."""
        return (
            -self.hit_rate,
            round(self.cpa_median, 3),
            round(self.cpa_p90, 3),
            round(self.cpa_cvar90, 3),
            round(self.effort_median, 4),
            round(self.sat_median, 5),
            -round(self.lock_frac_min, 5),
            round(self.jerk_median, 5),
            round(self.t_hit_median, 4) if self.t_hit_median is not None else 1e9,
        )

    def better_than(self, other: "BatchMetrics") -> bool:
        return self.rank_key() < other.rank_key()


# ── эпизод: сенсорный контур без учителя + теневое продолжение ────────────────


def _shadow_cpa(p_m_e: np.ndarray, v_m: np.ndarray, p_t_e: np.ndarray, v_t: np.ndarray) -> float:
    """Геометрический CPA «теневого продолжения»: с момента входа в сферу БЧ
    диагностическая копия траектории НЕ прекращается — оба тела летят с текущими
    скоростями до прохождения точки ближайшего сближения. Чистая аналитика,
    не зависит от kill radius и от dt (сходится при измельчении шага)."""
    r = p_t_e - p_m_e
    vr = v_t - v_m
    vv = float(np.dot(vr, vr))
    if vv < 1e-12:
        return float(np.linalg.norm(r))
    t_star = max(0.0, -float(np.dot(r, vr)) / vv)
    return float(np.linalg.norm(r + vr * t_star))


def evaluate_episode(
    circuit,
    sc: Scenario,
    *,
    teacher: bool = False,
    lr: float = 0.0,
    teacher_mode: str = "full_command",
    teacher_kind: str = "pn",
    t_cap: float = 16.0,
    feats: list[np.ndarray] | None = None,
    scenario_split: str = "train",
    controller: Callable[[dict, np.ndarray, float], np.ndarray] | None = None,
) -> EpisodeMetrics:
    """Один эпизод с сенсорным контуром (без кадров и снимков состояния).

    teacher=False (этап B): learn_step НЕ вызывается ни разу, oracle-ПН не
    вычисляется вовсе — оценка кандидата слепа к учителю по построению.
    teacher=True (этап A): oracle-закон даёт целевую команду обновления;
    ракету ведёт ТОЛЬКО контур BIO.
    controller — альтернатива circuit (closed-loop проверка извлечённых
    законов, P8): command(obs_delayed, v_m, n_max) → ускорение; контроллер
    получает ТОЛЬКО отложенный кадр ГСН — истинная геометрия недоступна
    по построению сигнатуры."""

    missile, target = spawn(sc)
    if circuit is not None:
        circuit.reset()
    prev: dict | None = None
    t = 0.0
    dt = float(sc.dt)
    half_fov = np.deg2rad(max(getattr(sc, "bio_fov_deg", 165.0), 2.0)) / 2.0
    effort = 0.0
    sat_steps = 0
    lock_time = 0.0
    steps = 0
    jerk_sum = 0.0
    a_prev: np.ndarray | None = None
    hit = False
    t_hit: float | None = None
    trigger_range = 1e9
    geo_cpa = 1e9
    sat_lim = 0.98 * float(sc.n_max) * G
    noise_rng = np.random.default_rng(sc.seed * 977 + 13)
    dead_mask = dead_mask_for(sc.retina_death_p, noise_rng)
    buffer = ObservationBuffer(sc.seeker_delay_s, sc.seeker_jitter_s, dt, noise_rng)

    while t < min(sc.t_max, t_cap):
        r = target.p - missile.p
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
        if controller is None:
            circuit.step(delayed, dt)

            if teacher:
                a_teacher = _teacher_command(teacher_kind, r, missile.v, target.v, sc.pn_n)
                tgt = _shape_teacher(a_teacher, missile.v, sc.n_max, teacher_mode)
                lr_hidden = lr * 0.3 if getattr(circuit, "kind", "") in ("full", "connectome") else 0.0
                circuit.learn_step(tgt, lr, lr_hidden)

            if feats is not None:
                fv = getattr(circuit, "feat", None)
                if fv is None:
                    fv = getattr(getattr(circuit, "state", None), "feat", None)
                if fv is not None and np.size(fv):
                    feats.append(np.asarray(fv, dtype=np.float64).copy())

            a_cmd = circuit.accel_cmd(missile.v, sc.n_max) if delayed["lock"] else np.zeros(3)
        else:
            a_cmd = controller(delayed, missile.v, sc.n_max) if delayed["lock"] else np.zeros(3)
        a_cmd = clip_accel(a_cmd, sc.n_max)
        mag = float(np.linalg.norm(a_cmd))
        effort += (mag / G) * dt
        sat_steps += int(mag >= sat_lim)
        if delayed["lock"]:
            lock_time += dt
        if a_prev is not None:
            jerk_sum += float(np.linalg.norm(a_cmd - a_prev)) / sat_lim
        a_prev = a_cmd

        p_m0, p_t0 = missile.p.copy(), target.p.copy()
        integrate_target(target, sc, t, dt)
        integrate(missile, a_cmd, dt)
        enc = step_encounter(p_m0, missile.p, p_t0, target.p, sc.kill_radius_m)
        if enc["hit"]:
            hit = True
            alpha = float(enc["alpha_in"])
            t_hit = t + alpha * dt
            # точка входа в сферу БЧ (интерполяция внутри шага) и её дальность
            p_m_e = p_m0 + (missile.p - p_m0) * alpha
            p_t_e = p_t0 + (target.p - p_t0) * alpha
            trigger_range = float(np.linalg.norm(p_t_e - p_m_e))
            geo_cpa = _shadow_cpa(p_m_e, missile.v, p_t_e, target.v)
            break
        if enc["cpa"] < geo_cpa:
            geo_cpa = enc["cpa"]  # непрерывный минимум по летенному участку
        if t > 0.5 and float(-np.dot(r, target.v - missile.v) / (np.linalg.norm(r) + 1e-9)) < 0 and np.linalg.norm(r) > geo_cpa + 80:
            break
        t += dt
        steps += 1

    duration = max((steps + 1) * dt, 1e-9)
    if not hit:
        trigger_range = geo_cpa  # срабатывания не было — «trigger» не наступил
    return EpisodeMetrics(
        hit=hit,
        t_hit=t_hit,
        trigger_range_m=float(trigger_range),
        geometric_cpa_m=float(geo_cpa),
        effort_gs=float(effort),
        sat_frac=float(sat_steps / max(steps + 1, 1)),
        lock_frac=float(lock_time / duration),
        jerk=float(jerk_sum / duration),
        duration_s=float(duration),
        scenario_split=scenario_split,
        v_m=float(sc.v_m),
        v_t=float(sc.v_t),
        aspect=str(sc.aspect),
        speed_mode=str(sc.target_speed_mode),
    )


def _teacher_command(kind: str, r: np.ndarray, v_m: np.ndarray, v_t: np.ndarray, n_const: float) -> np.ndarray:
    """Oracle-teacher этапа A: «pn» — ПН, «pursuit» — пеленговый рефлекс погони.
    Истинная геометрия разрешена ТОЛЬКО здесь (teacher) и никогда не входит в BIO."""
    if kind == "pursuit":
        return r  # направление на цель; амплитуду срежет _shape_teacher
    return ppn_accel(r, v_m, v_t, n_const, n_max=1e9)  # без клипа — амплитуду задаёт режим


def _shape_teacher(a: np.ndarray, v_m: np.ndarray, n_max: float, mode: str) -> np.ndarray:
    """Режимы teacher-сигнала этапа A:
    full_command — точная нормированная команда (две проекции в −1…1);
    direction_only — только направление (без навязывания амплитуды ПН);
    clipped_coarse — грубо квантованная ограниченная амплитуда."""
    _x, y, z = body_axes(v_m)
    scale = n_max * G + 1e-9
    cmd = np.clip(np.array([float(np.dot(a, z)) / scale, float(np.dot(a, y)) / scale]), -1.0, 1.0)
    if mode == "direction_only":
        n = float(np.linalg.norm(cmd))
        if n > 1e-6:
            cmd = cmd / n
    elif mode == "clipped_coarse":
        cmd = np.clip(np.round(cmd * 2.0) / 2.0, -0.5, 0.5)  # шаг 0.5, амплитуда ≤ 0.5
    return cmd


# ── батчи сценариев: train (свежие каждый поколение) / validation (фикс) ──────


def _base_scenario(gain: float, t_max: float = 16.0) -> Scenario:
    return Scenario(aspect="head-on", mode="bio", t_max=t_max, dt=0.02, circuit_gain=gain, n_max=30.0, kill_radius_m=45.0)


def train_batch(generation: int, cfg: "EvolveConfig", gain: float) -> list[Scenario]:
    """Paired train-батч поколения: НОВЫЕ сценарии в каждом поколении, один и тот
    же батч для ВСЕХ кандидатов поколения. Шаг интегрирования чередуется —
    замкнутый контур на декодированных углах чувствителен к дискретизации."""
    base = _base_scenario(gain, cfg.t_cap)
    batch: list[Scenario] = []
    for k in range(cfg.scenarios_per_gen):
        sc = sample_generation_scenario(base, generation, cfg.seed * 131 + generation * 17 + k)
        sc = replace(
            sc,
            dt=cfg.dt_cycle[k % len(cfg.dt_cycle)],
            noise_az_deg=float((k % 3) * 0.9),
            n_max=30.0,
            kill_radius_m=45.0,
        )
        batch.append(sc)
    return batch


def validation_batch(cfg: "EvolveConfig", gain: float) -> list[Scenario]:
    """Фиксированный validation-батч: разные аспекты, скорости, профили скорости,
    манёвры, шумы, задержки и отказы сетчатки. НЕ меняется между поколениями;
    по нему выбирается чемпион. Каноническое трио остаётся smoke-тестом."""
    rng = np.random.default_rng(cfg.seed + 777)
    aspects = ("head-on", "beam", "tail-chase")
    maneuvers = ("straight", "weave", "break", "dive")
    modes = ("constant", "accelerate", "decelerate", "pulse", "sine")
    scens: list[Scenario] = []
    base = _base_scenario(gain, cfg.t_cap)
    for i in range(cfg.validation_size):
        period = float(rng.uniform(3.0, 8.0))
        mode = modes[i % len(modes)]
        sc = replace(
            base,
            aspect=aspects[i % 3],
            maneuver=maneuvers[i % len(maneuvers)],
            v_m=float(rng.uniform(*PROTOCOL["v_m"])),
            v_t=float(rng.uniform(*PROTOCOL["v_t"])),
            range_m=float(rng.uniform(4500.0, 9500.0)),
            off_axis_m=float(rng.uniform(80.0, 900.0)),
            n_target=float(rng.choice([0.0, 2.0, 3.0, 4.0])),
            dt=cfg.dt_cycle[i % len(cfg.dt_cycle)],
            noise_az_deg=float(rng.choice([0.0, 0.9, 1.8])),
            noise_range_m=float(rng.choice([0.0, 40.0])),
            seeker_delay_s=float(rng.uniform(0.0, 0.04)),
            retina_death_p=float(rng.choice([0.0, 0.1])),
            seed=cfg.seed + 2000 + i,
        )
        if mode != "constant":
            sc = replace(
                sc,
                target_speed_mode=mode,  # type: ignore[arg-type]
                target_longitudinal_g=float(rng.uniform(0.5, 1.5)),
                target_speed_min=float(max(PROTOCOL["target_speed_bounds"]["abs_min"], sc.v_t * 0.55)),
                target_speed_max=float(min(PROTOCOL["target_speed_bounds"]["abs_max"], sc.v_t * 1.5)),
                target_speed_period_s=period,
                target_speed_phase=float(rng.uniform(0.0, 1.0)) * period,
            )
        scens.append(sc)
    return scens


def canonical_trio(gain: float) -> list[Scenario]:
    """Каноническое трио — smoke-тест, НЕ критерий выбора чемпиона."""
    return [
        replace(_base_scenario(gain, 16.0), aspect=a, range_m=8000.0, off_axis_m=420.0, v_t=260.0, seed=3)
        for a in ("head-on", "beam", "tail-chase")
    ]


def evaluate_batch(
    circuit,
    scenarios: list[Scenario],
    *,
    teacher: bool = False,
    lr: float = 0.0,
    teacher_mode: str = "full_command",
    teacher_kind: str = "pn",
    t_cap: float = 16.0,
    scenario_split: str = "train",
    controller: Callable[[dict, np.ndarray, float], np.ndarray] | None = None,
) -> BatchMetrics:
    eps = [
        evaluate_episode(
            circuit,
            sc,
            teacher=teacher,
            lr=lr,
            teacher_mode=teacher_mode,
            teacher_kind=teacher_kind,
            t_cap=t_cap,
            scenario_split=scenario_split,
            controller=controller,
        )
        for sc in scenarios
    ]
    return BatchMetrics(episodes=eps)


# ── этап A: coarse PN warm start ─────────────────────────────────────────────


@dataclass
class WarmStartConfig:
    episodes: int = 24
    lr: float = 0.04
    teacher_mode: str = "direction_only"  # full_command | direction_only | clipped_coarse
    teacher_kind: str = "pn"  # pn | pursuit
    teacher_weight: float = 1.0
    eval_every: int = 4  # оценка на validation каждые N эпизодов
    early_stop_hit_frac: float | None = 0.95  # стоп: validation hit rate достигнута
    patience: int = 8  # стоп: нет улучшения validation N оценок подряд
    seed: int = 11
    t_cap: float = 16.0


def warm_start(
    circuit,
    cfg: WarmStartConfig | None = None,
    *,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Этап A: грубый навигационный рефлекс подражанием oracle-teacher.
    Лучшие веса — по VALIDATION (каноническое трио + фиксированные train-ячейки),
    не по train-эпизодам. history несёт флаги этапа для отчётности."""
    cfg = cfg or WarmStartConfig()
    rng = np.random.default_rng(cfg.seed)
    history: list[dict[str, Any]] = []

    val_trio = canonical_trio(circuit.gain)
    best_key: tuple | None = None
    best_snapshot: dict[str, Any] | None = None
    best_val: BatchMetrics | None = None
    no_improve = 0

    def _snapshot() -> dict[str, Any]:
        snap: dict[str, Any] = {"w": circuit.W_dn.copy()}
        if getattr(circuit, "W_fx", None) is not None:
            snap["fx"] = circuit.W_fx.copy()
        return snap

    def _restore(snap: dict[str, Any] | None) -> None:
        if snap is None:
            return
        circuit.W_dn = snap["w"].copy()
        if "fx" in snap and getattr(circuit, "W_fx", None) is not None:
            circuit.W_fx = snap["fx"].copy()

    for ep in range(cfg.episodes):
        sc = _episode_scenario(ep, circuit.gain, rng=rng)
        # эпизод этапа A: учимся на посещённых BIO состояниях (teacher-команда)
        res = evaluate_episode(
            circuit,
            sc,
            teacher=True,
            lr=cfg.lr * cfg.teacher_weight,
            teacher_mode=cfg.teacher_mode,
            teacher_kind=cfg.teacher_kind,
            t_cap=cfg.t_cap,
            scenario_split="train",
        )
        history.append(
            {
                "stage": STAGE_WARM_START,
                "episode": ep,
                "split": "train",
                "teacher_active": True,
                "teacher_weight": cfg.teacher_weight,
                "teacher_mode": cfg.teacher_mode,
                "teacher_kind": cfg.teacher_kind,
                **{k: res.to_dict()[k] for k in ("hit", "t_hit", "geometric_cpa_m", "effort_gs", "lock_frac")},
            }
        )
        if (ep + 1) % cfg.eval_every == 0 or ep == cfg.episodes - 1:
            val = evaluate_batch(circuit, val_trio, teacher=False, t_cap=cfg.t_cap, scenario_split="validation")
            history.append(
                {
                    "stage": STAGE_WARM_START,
                    "episode": ep,
                    "split": "validation",
                    "teacher_active": False,
                    **val.components(),
                }
            )
            key = val.rank_key()
            if best_key is None or key < best_key:
                best_key, best_snapshot, best_val = key, _snapshot(), val
                no_improve = 0
            else:
                no_improve += 1
            if progress:
                progress({"stage": STAGE_WARM_START, "episode": ep + 1, "episodes": cfg.episodes, "validation": val.components()})
            if cfg.early_stop_hit_frac is not None and val.hit_rate >= cfg.early_stop_hit_frac:
                break
            if no_improve >= cfg.patience:
                break
    _restore(best_snapshot)
    return {
        "stage": STAGE_WARM_START,
        "config": asdict(cfg),
        "history": history,
        "validation": best_val.components() if best_val else None,
        "episodes_done": len([h for h in history if h.get("split") == "train"]),
        "early_stopped": len([h for h in history if h.get("split") == "train"]) < cfg.episodes,
    }


# ── этап B: outcome evolution (CEM по физическому результату) ─────────────────


@dataclass
class EvolveConfig:
    preset: str = "quick"  # quick | research
    generations: int = 10
    pop: int = 8
    elite_frac: float = 0.15
    scenarios_per_gen: int = 4
    validation_size: int = 6
    sigma: float = 0.3
    sigma_min: float = 0.05
    sigma_max: float = 0.6
    fresh_scenarios: bool = True
    dt_cycle: tuple[float, ...] = (0.02, 0.01, 0.02, 0.005)
    seed: int = 101
    t_cap: float = 16.0
    mutate_fx: bool = True  # коннектом: мутации каналов W_fx


PRESETS: dict[str, dict[str, Any]] = {
    # демонстрационный/тестовый режим
    "quick": {"generations": 10, "pop": 8, "scenarios_per_gen": 4, "validation_size": 6, "sigma": 0.3},
    # полноценный эксперимент по поиску закона (P3): 24–64 кандидата, 30–100 поколений,
    # элита 10–20%, 8–20 сценариев на поколение, новые train-сценарии каждый раз
    "research": {"generations": 40, "pop": 32, "scenarios_per_gen": 10, "validation_size": 12, "elite_frac": 0.12, "sigma": 0.35},
}


def outcome_evolve(
    circuit,
    cfg: EvolveConfig | None = None,
    *,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Этап B: свободная доводка по результату перехвата.

    - teacher отключён полностью: evaluate_batch вызывается с teacher=False
      (learn_step не выполняется ни для одного кандидата);
    - fitness — лексикографический ранг BatchMetrics (см. rank_key): hit rate,
      робастный geometric CPA, усилие, насыщение, захват, jerk, время;
    - все кандидаты поколения летят ОДИН И ТОТ ЖЕ paired train-батч;
    - train-батчи НОВЫЕ в каждом поколении; validation-батч ФИКСИРОВАННЫЙ,
      чемпион выбирается только по нему; test не существует внутри этапа."""
    cfg = cfg or EvolveConfig(preset="quick")
    if cfg.preset in PRESETS:
        for k, v in PRESETS[cfg.preset].items():
            setattr(cfg, k, v)
    rng = np.random.default_rng(cfg.seed)
    val_scens = validation_batch(cfg, circuit.gain)
    history: list[dict[str, Any]] = []

    mean_w = circuit.W_dn.copy()
    mean_fx = circuit.W_fx.copy() if (cfg.mutate_fx and getattr(circuit, "W_fx", None) is not None) else None
    sigma = float(cfg.sigma)
    champion_w = circuit.W_dn.copy()
    champion_fx = mean_fx.copy() if mean_fx is not None else None
    champion_val = evaluate_batch(circuit, val_scens, teacher=False, t_cap=cfg.t_cap, scenario_split="validation")
    champion_key = champion_val.rank_key()
    history.append({"stage": STAGE_OUTCOME, "generation": 0, "split": "validation", "teacher_active": False, **champion_val.components()})

    def _apply(w: np.ndarray, fx: np.ndarray | None) -> None:
        circuit.W_dn = w
        if fx is not None and getattr(circuit, "W_fx", None) is not None:
            circuit.W_fx = fx

    elite_k = max(2, int(round(cfg.pop * cfg.elite_frac)))
    for gen in range(cfg.generations):
        batch = train_batch(gen, cfg, circuit.gain)
        scored: list[tuple[tuple, np.ndarray, np.ndarray | None, BatchMetrics]] = []
        for pi in range(cfg.pop):
            if pi == 0 and gen > 0:
                w, fx = champion_w, champion_fx  # элита: чемпион летит без мутаций
            else:
                w = np.clip(mean_w + rng.normal(0, sigma, mean_w.shape), -4.0, 4.0)
                fx = None
                if mean_fx is not None:
                    # каналы признаков («зрение») мутируют вдвое мягче
                    fx = np.clip(mean_fx + rng.normal(0, sigma * 0.5, mean_fx.shape), -2.0, 2.0)
            _apply(w, fx)
            circuit.reset()
            bm = evaluate_batch(circuit, batch, teacher=False, t_cap=cfg.t_cap, scenario_split="train")
            scored.append((bm.rank_key(), w, fx, bm))
        scored.sort(key=lambda x: x[0])

        # CEM-обновление: новое среднее по элите; σ адаптивно (прогресс сжимает)
        elite_w = np.stack([w for _k, w, _fx, _bm in scored[:elite_k]])
        mean_w = elite_w.mean(axis=0)
        if mean_fx is not None:
            fx_stack = [fx for _k, _w, fx, _bm in scored[:elite_k] if fx is not None]
            if fx_stack:
                mean_fx = np.stack(fx_stack).mean(axis=0)
        gen_best_key, _w, _fx, gen_best_bm = scored[0]
        sigma = float(np.clip(sigma * (0.88 if gen_best_key < champion_key else 1.15), cfg.sigma_min, cfg.sigma_max))

        # чемпион — ТОЛЬКО по validation: кандидат-победитель поколения
        gen_best_w, gen_best_fx = scored[0][1], scored[0][2]
        _apply(gen_best_w, gen_best_fx)
        circuit.reset()
        val = evaluate_batch(circuit, val_scens, teacher=False, t_cap=cfg.t_cap, scenario_split="validation")
        if val.rank_key() < champion_key:
            champion_key = val.rank_key()
            champion_w = gen_best_w.copy()
            champion_fx = gen_best_fx.copy() if gen_best_fx is not None else None
            champion_val = val
        _apply(champion_w, champion_fx)
        circuit.reset()

        history.append(
            {
                "stage": STAGE_OUTCOME,
                "generation": gen + 1,
                "split": "train",
                "teacher_active": False,
                "sigma": round(sigma, 4),
                **gen_best_bm.components(),
            }
        )
        history.append({"stage": STAGE_OUTCOME, "generation": gen + 1, "split": "validation", "teacher_active": False, **val.components()})
        if progress:
            progress({"stage": STAGE_OUTCOME, "generation": gen + 1, "generations": cfg.generations, "validation": val.components(), "sigma": sigma})

    _apply(champion_w, champion_fx)
    circuit.reset()
    return {
        "stage": STAGE_OUTCOME,
        "config": asdict(cfg),
        "history": history,
        "validation": champion_val.components(),
        "generations_done": cfg.generations,
    }


# ── полный конвейер + артефакты ───────────────────────────────────────────────

INIT_MODES = (
    "pn_full_warmstart",     # A(full_command) → B
    "pn_coarse_warmstart",   # A(direction_only) → B — основной исследовательский
    "pursuit_warmstart",     # A(пеленговый рефлекс) → B
    "random_initialization", # случайные веса → B (без этапа A)
    "warmstart_only",        # только этап A (для сравнения вклада этапов)
)


def _init_circuit(kind: str, init: str, seed: int):
    """Экземпляр мозга для эксперимента: НЕ живой стенд, НЕ data/-файлы.
    random_initialization — честно случайные веса (не врождённый рефлекс)."""
    rng = np.random.default_rng(seed + 5)
    if kind == "connectome":
        c = ConnectomeCircuit()
    elif kind == "full":
        c = FlyCircuit(kind="full")
    else:
        c = FlyCircuit(kind="stub")
    c.reset()
    if init == "random_initialization":
        c.W_dn = np.clip(rng.normal(0, 0.25, c.W_dn.shape), -2.0, 2.0)
        if getattr(c, "W_fx", None) is not None:
            fx = c.W_fx.copy()
            mask = fx != 0.0
            noise = rng.normal(0, 0.25, fx.shape) * mask
            c.W_fx = np.clip(fx + noise, -2.0, 2.0)
        c.trained = False
    else:
        c.trained = False
    return c


def _git_hash() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def save_artifacts(run_dir: Path, payload: dict[str, Any], circuit, config: dict[str, Any], history: list[dict]) -> Path:
    """Checkpoint эксперимента: конфиг, веса, история, метрики, отчёт, README.
    Прод-веса data/ не затрагиваются никогда."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(
        run_dir / "weights.npz",
        W_dn=circuit.W_dn,
        **({"W_fx": circuit.W_fx} if getattr(circuit, "W_fx", None) is not None else {}),
        feat_schema=np.int64(FEATURE_SCHEMA_VERSION),
        kind=np.array([str(circuit.kind)], dtype="U16"),
    )
    (run_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")
    for name in ("train_metrics", "validation_metrics", "test_metrics", "law_report"):
        if name in payload:
            (run_dir / f"{name}.json").write_text(json.dumps(payload[name], ensure_ascii=False, indent=2), encoding="utf-8")
    if not (run_dir / "README.md").exists():
        (run_dir / "README.md").write_text(
            "# law_discovery run\n\n"
            f"- kind: `{config.get('kind')}` · init: `{config.get('init')}` · seed: `{config.get('seed')}`\n"
            f"- feature_schema: v{FEATURE_SCHEMA_VERSION} · git: `{config.get('git_hash', 'unknown')}`\n"
            "- Этап A: imitation warm start (teacher-команда oracle на посещённых BIO состояниях)\n"
            "- Этап B: outcome evolution (teacher отключён, fitness — только физический результат)\n"
            "- Чемпион выбран по validation-батчу; test_metrics.json появляется только после\n"
            "  явной финальной оценки (held-out дисциплина).\n",
            encoding="utf-8",
        )
    return run_dir


def discover_law(
    kind: str = "stub",
    *,
    init: str = "pn_coarse_warmstart",
    preset: str = "quick",
    seed: int = 0,
    run_id: str | None = None,
    warm_cfg: WarmStartConfig | None = None,
    evolve_cfg: EvolveConfig | None = None,
    evaluate_test: bool = False,
    test_scenarios: list[Scenario] | None = None,
    feat_zero_columns: tuple[int, ...] = (),
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Полный конвейер: этап A (если init ≠ random) → этап B → артефакты.
    evaluate_test=True — единственный способ получить test_metrics.json:
    held-out оценка выполняется ОДИН раз по окончании всех решений.
    feat_zero_columns — retraining-абляция (P9): столбцы признаков (4=theta,
    6=rho) обнуляются feat_patch'ем НА ВСЁ обучение и оценку — мозг растёт
    без них (не путать с inference-абляцией в science.feature_ablation).
    warm_cfg/evolve_cfg — явные конфиги; если warm_cfg не передан, режим
    teacher берётся из init (full_command / direction_only / pursuit)."""
    if init not in INIT_MODES:
        raise ValueError(f"init должен быть одним из {INIT_MODES}")
    t0 = time.time()
    circuit = _init_circuit(kind, init, seed)
    if warm_cfg is None:
        warm_cfg = WarmStartConfig(
            episodes=24,
            teacher_mode=("full_command" if init == "pn_full_warmstart" else "direction_only"),
            teacher_kind=("pursuit" if init == "pursuit_warmstart" else "pn"),
            seed=seed + 11,
        )
    if feat_zero_columns:
        circuit.feat_patch = {int(i): 0.0 for i in feat_zero_columns}
    evolve_cfg = evolve_cfg or EvolveConfig(preset=preset, seed=seed + 101)

    warm_report: dict[str, Any] | None = None
    evolution: dict[str, Any] | None = None
    if init != "random_initialization":
        warm_report = warm_start(circuit, warm_cfg, progress=progress)
    if init == "warmstart_only":
        # только этап A: свободной доводки нет — чемпион это результат имитации
        _cfg = evolve_cfg or EvolveConfig(preset="custom")
        val = evaluate_batch(circuit, validation_batch(_cfg, circuit.gain), teacher=False, t_cap=_cfg.t_cap, scenario_split="validation")
        evolution = {"stage": STAGE_WARM_START + "_only", "config": asdict(_cfg), "history": [], "validation": val.components(), "generations_done": 0}
    else:
        evolution = outcome_evolve(circuit, evolve_cfg, progress=progress)

    validation_metrics = evolution["validation"]
    train_last = [h for h in evolution["history"] if h.get("split") == "train"]
    test_metrics: dict[str, Any] | None = None
    if evaluate_test:
        scens = test_scenarios if test_scenarios is not None else canonical_trio(circuit.gain)
        tm = evaluate_batch(circuit, scens, teacher=False, t_cap=evolve_cfg.t_cap, scenario_split="test")
        test_metrics = tm.components()

    config = {
        "kind": circuit.kind,
        "init": init,
        "preset": evolve_cfg.preset,
        "seed": seed,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "git_hash": _git_hash(),
        "warm_start": asdict(warm_cfg) if warm_report else None,
        "evolve": asdict(evolve_cfg),
        "teacher_blind_stage_b": True,
        "feat_zero_columns": list(feat_zero_columns),
    }
    payload = {
        "train_metrics": {"last_generation": train_last[-1] if train_last else None},
        "validation_metrics": validation_metrics,
    }
    if test_metrics is not None:
        payload["test_metrics"] = test_metrics  # только по явному evaluate_test=True
    rid = run_id or f"{circuit.kind}_{init}_{preset}_s{seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = save_artifacts(ARTIFACTS_DIR / rid, payload, circuit, config, evolution["history"])

    return {
        "run_id": rid,
        "run_dir": str(run_dir),
        "kind": circuit.kind,
        "init": init,
        "preset": evolve_cfg.preset,
        "seed": seed,
        "warm_start": warm_report,
        "evolution": evolution,
        "validation": validation_metrics,
        "test": test_metrics,
        "seconds": round(time.time() - t0, 1),
        "weights": circuit.W_dn.copy(),
    }
