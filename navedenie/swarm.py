"""Рой мух: популяция наводящихся аппаратов, отбор лучших, эволюция генома.

Вид «bio» — мозг дрозофилы (веса выхода DN + усиление), вид «pn» — эталонная
пропорциональная навигация с собственной постоянной N. Оба вида летят одну и ту
же цель в одном сценарии; приспособленность — минимальный промах за прыжок.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from navedenie.circuit import FEAT_DIM, FlyCircuit, migrate_feature_columns
from navedenie.parallel import parallel_map
from navedenie.pn import ppn_accel
from navedenie.seeker import body_axes, observe
from navedenie.sim import G, Scenario, clip_accel, closing_speed, integrate, integrate_target, spawn, target_accel

W_DIM = (2, FEAT_DIM)
W_LIMIT = 4.0
GAIN_RANGE = (0.4, 2.5)
PN_RANGE = (1.5, 8.0)
DEFAULT_MUTATION = 0.25


@dataclass
class FlyGenome:
    kind: str = "bio"  # "bio" | "pn"
    w: np.ndarray = field(default_factory=lambda: np.zeros(W_DIM))  # только bio
    gain: float = 1.0  # только bio
    pn_n: float = 4.0  # только pn

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "w": [float(x) for x in self.w.reshape(-1)] if self.kind == "bio" else [],
            "gain": float(self.gain),
            "pn_n": float(self.pn_n),
        }


def fly_from_json(blob: dict) -> FlyGenome:
    kind = "pn" if blob.get("kind") == "pn" else "bio"
    w = np.zeros(W_DIM)
    if kind == "bio":
        flat = np.asarray(blob.get("w") or [], dtype=np.float64).reshape(-1)
        want = W_DIM[0] * W_DIM[1]
        if flat.size == want:
            w = flat.reshape(W_DIM)
        elif flat.size == 2 * 8:  # старый геном схемы v1: миграция нулями на theta/rho
            w = migrate_feature_columns(flat.reshape(2, 8), axis=1).reshape(W_DIM)
    w = np.clip(w, -W_LIMIT, W_LIMIT)
    gain = float(np.clip(float(blob.get("gain") or 1.0), *GAIN_RANGE))
    pn_n = float(np.clip(float(blob.get("pn_n") or 4.0), *PN_RANGE))
    return FlyGenome(kind=kind, w=w, gain=gain, pn_n=pn_n)


def default_w() -> np.ndarray:
    """Стартовые веса как у необученной схемы: пеленг → рули."""
    from navedenie.circuit import default_W

    return default_W().copy()


def init_population(n: int, seed: int = 7) -> list[FlyGenome]:
    rng = np.random.default_rng(seed)
    flies: list[FlyGenome] = []
    for i in range(max(n, 2)):
        if i % 2 == 0:
            w = default_w() + rng.normal(0.0, 0.35, W_DIM)
            flies.append(FlyGenome(kind="bio", w=np.clip(w, -W_LIMIT, W_LIMIT), gain=float(rng.uniform(*GAIN_RANGE))))
        else:
            flies.append(FlyGenome(kind="pn", pn_n=float(rng.uniform(*PN_RANGE))))
    return flies


def sample_generation_scenario(base: Scenario, gen: int, seed: int) -> Scenario:
    """Геометрия поколения: детерминированно новая на каждое поколение — рой
    не должен учиться на одной и той же траектории цели. Манёвры — от простых
    к сложным: чем дальше поколение, тем богаче набор. В ~40% поколений цель
    меняет скорость внутри эпизода (разгон/торможение/импульс/синус)."""
    rng = np.random.default_rng((seed * 7919 + gen * 104729) % (2**32))
    aspects = ("head-on", "beam", "tail-chase")
    easy = ("straight", "turn", "weave")
    hard = ("weave_var", "break", "scissors", "dive", "combo")
    pool = easy if gen < 4 else easy + hard
    sc = replace(
        base,
        aspect=aspects[gen % 3],
        range_m=float(rng.uniform(4500, 9000)),
        v_t=float(rng.uniform(180, 330)),
        off_axis_m=float(rng.uniform(80, 900)),
        maneuver=pool[int(rng.integers(0, len(pool)))],
        n_target=float(rng.choice([0.0, 0.0, 2.0, 3.0])),
        alt_m=float(rng.uniform(3500, 4500)),
    )
    if rng.random() < 0.4:
        modes = ("accelerate", "decelerate", "pulse", "sine")
        period = float(rng.uniform(3.0, 8.0))
        sc = replace(
            sc,
            target_speed_mode=modes[int(rng.integers(0, len(modes)))],
            target_longitudinal_g=float(rng.uniform(0.5, 1.5)),
            target_speed_min=float(max(120.0, sc.v_t * 0.55)),
            target_speed_max=float(min(520.0, sc.v_t * 1.5)),
            target_speed_period_s=period,
            target_speed_phase=float(rng.uniform(0.0, 1.0)) * period,
        )
    return sc


def canonical_scenarios(base: Scenario) -> list[Scenario]:
    """Фиксированное эталонное трио для валидации чемпиона роя."""
    out = []
    for aspect in ("head-on", "beam", "tail-chase"):
        out.append(
            replace(
                base,
                aspect=aspect,
                range_m=6000.0,
                off_axis_m=250.0,
                v_t=240.0,
                maneuver="straight",
                n_target=0.0,
                alt_m=4000.0,
            )
        )
    return out


# фитнес: ЛЮБОЙ перехват лучше любого промаха; внутри — градиенты по времени/цене и CPA
HIT_BASE = 1000.0
CPA_CAP = 5000.0


def rollout(sc: Scenario, fly: FlyGenome, *, dt: float = 0.02, sample_dt: float = 0.25) -> dict:
    """Лёгкий прогон одной мухи: без кадров и снимков контура.

    Зрение и задержки — как в основном движке (фовеальная сетчатка, декодирование
    из изображения, постоянная карта отказов, очередь задержки/джиттера).
    Перехват — межшаговое пересечение сферы БЧ; fitness:
    - перехват: t_hit + 0.02·effort (быстрее и дешевле — лучше);
    - промах:    HIT_BASE + CPA (любой перехват лучше любого промаха,
                  среди промахов сохраняется градиент по непрерывному CPA)."""
    from navedenie.seeker import ObservationBuffer, dead_mask_for, observe
    from navedenie.sim import step_encounter

    t_max = min(sc.t_max, 20.0)
    missile, target = spawn(sc)
    circuit: FlyCircuit | None = None
    if fly.kind == "bio":
        circuit = FlyCircuit(kind="stub", tau_s=sc.tau_s, gain=fly.gain)
        circuit.W_dn = fly.w.copy()
    prev: dict | None = None
    t = 0.0
    cpa = 1e9
    effort = 0.0
    hit = False
    t_hit: float | None = None
    traj_m: list[list[float]] = []
    traj_t: list[list[float]] = []
    tel: list[dict] = []
    next_sample = 0.0
    noise_rng = np.random.default_rng(sc.seed * 977 + 13)
    dead_mask = dead_mask_for(sc.retina_death_p, noise_rng)
    half_fov_bio = np.deg2rad(max(getattr(sc, "bio_fov_deg", 165.0), 2.0)) / 2.0
    half_fov_pn = np.deg2rad(sc.fov_deg) / 2.0
    buffer = ObservationBuffer(sc.seeker_delay_s, sc.seeker_jitter_s, dt, noise_rng)

    def sample(dn_pitch: float = 0.0, dn_yaw: float = 0.0, n_req: float = 0.0) -> None:
        traj_m.append([float(missile.p[0]), float(missile.p[1]), float(missile.p[2])])
        traj_t.append([float(target.p[0]), float(target.p[1]), float(target.p[2])])
        tel.append(
            {
                "t": float(t),
                "az": 0.0,
                "el": 0.0,
                "lock": False,
                "size": float(prev_size),
                "sizeDot": 0.0,
                "azDot": 0.0,
                "elDot": 0.0,
                "pitch": float(dn_pitch),
                "yaw": float(dn_yaw),
                "nReq": float(n_req),
                "miss": float(cpa),
                "rng": float(np.linalg.norm(target.p - missile.p)),
                "v": [float(missile.v[0]), float(missile.v[1]), float(missile.v[2])],
            }
        )

    prev_size = 0.0
    sample()
    while t < t_max:
        r = target.p - missile.p
        rng_n = float(np.linalg.norm(r))

        if fly.kind == "pn":
            # «pn»-муха — oracle-закон с постоянной N (сенсорных pn-мух в рое нет)
            a_cmd = ppn_accel(r, missile.v, target.v, fly.pn_n, sc.n_max)
            obs = None
            lock = True
        else:
            obs = observe(
                r,
                missile.v,
                prev,
                half_fov=half_fov_bio,
                dt_s=dt,
                noise_az=np.deg2rad(sc.noise_az_deg),
                noise_range=sc.noise_range_m,
                lock_drop_p=sc.lock_drop_p,
                dead_mask=dead_mask,
                dropout_p=sc.retina_dropout_p,
                rng=noise_rng,
                brightness=getattr(sc, "target_brightness", 1.0),
            )
            prev = obs
            buffer.push(t, obs)
            delayed = buffer.sample(t)
            lock = bool(delayed["lock"])
            if lock:
                assert circuit is not None
                circuit.step(delayed, dt)
                a_cmd = circuit.accel_cmd(missile.v, sc.n_max)
            else:
                a_cmd = np.zeros(3)
        a_cmd = clip_accel(a_cmd, sc.n_max)
        n_req = float(np.linalg.norm(a_cmd) / G)
        effort += n_req * dt

        # проекции команды на оси корпуса — телеметрия «рулей»
        _fx, fy, fz = body_axes(missile.v)
        dn_pitch = float(np.clip(np.dot(a_cmd, fz) / (sc.n_max * G + 1e-9), -1.0, 1.0))
        dn_yaw = float(np.clip(np.dot(a_cmd, fy) / (sc.n_max * G + 1e-9), -1.0, 1.0))

        # интеграция (|v| ракеты сохраняется; цель — с профилем скорости) и встреча
        p_m0, p_t0 = missile.p.copy(), target.p.copy()
        integrate_target(target, sc, t, dt)
        integrate(missile, a_cmd, dt)
        enc = step_encounter(p_m0, missile.p, p_t0, target.p, sc.kill_radius_m)
        cpa = min(cpa, enc["cpa"])

        if enc["hit"]:
            hit = True
            t_hit = t + float(enc["alpha_in"]) * dt
            sample(dn_pitch, dn_yaw, n_req)
            break
        v_rel = target.v - missile.v
        if t > 0.6 and closing_speed(r, v_rel) < 0 and rng_n > cpa + 50:
            break

        t += dt
        if t >= next_sample:
            sample(dn_pitch, dn_yaw, n_req)
            next_sample += sample_dt
            if obs is not None:
                last = tel[-1]
                last["az"] = float(obs["az"])
                last["el"] = float(obs["el"])
                last["lock"] = lock
                last["size"] = float(obs["size"])
                last["sizeDot"] = float(obs["size_dot"])
                last["azDot"] = float(obs["az_dot"])
                last["elDot"] = float(obs["el_dot"])
        prev_size = float(obs["size"]) if obs is not None else prev_size

    if hit:
        fitness = float(t_hit) + 0.02 * effort
    else:
        fitness = HIT_BASE + min(cpa, CPA_CAP)
    return {
        "fitness": float(fitness),
        "fitness_parts": {
            "hit": bool(hit),
            "t_hit": float(t_hit) if t_hit is not None else None,
            "cpa_m": float(cpa),
            "effort_gs": float(effort),
        },
        "miss_m": float(cpa),
        "cpa_m": float(cpa),
        "hit": bool(hit),
        "n_int": float(effort),
        "traj_m": traj_m,
        "traj_t": traj_t,
        "tel": tel,
    }


def evaluate_population(sc: Scenario, flies: list[FlyGenome]) -> list[dict]:
    # бои независимы — на всех ядрах поколение роя считается в разы быстрее,
    # а процесс стенда не зажимает интерактивные запуски (см. parallel.py)
    return parallel_map(rollout, [(sc, fly) for fly in flies])


def _mutate(fly: FlyGenome, sigma: float, rng: np.random.Generator) -> FlyGenome:
    out = FlyGenome(kind=fly.kind, w=fly.w.copy(), gain=fly.gain, pn_n=fly.pn_n)
    if out.kind == "bio":
        out.w = np.clip(out.w + rng.normal(0.0, sigma, W_DIM), -W_LIMIT, W_LIMIT)
        out.gain = float(np.clip(out.gain + rng.normal(0.0, sigma * 0.3), *GAIN_RANGE))
    else:
        out.pn_n = float(np.clip(out.pn_n + rng.normal(0.0, sigma * 1.5), *PN_RANGE))
    return out


def _crossover(a: FlyGenome, b: FlyGenome, rng: np.random.Generator) -> FlyGenome:
    kind = a.kind if (a.kind == b.kind or rng.random() < 0.5) else b.kind
    if kind == "pn":
        return FlyGenome(kind="pn", w=default_w(), gain=1.0, pn_n=float(a.pn_n if rng.random() < 0.5 else b.pn_n))
    mask = rng.random(W_DIM) < 0.5
    w = np.where(mask, a.w, b.w)
    gain = a.gain if rng.random() < 0.5 else b.gain
    return FlyGenome(kind="bio", w=w.copy(), gain=gain, pn_n=4.0)


def evolve(
    flies: list[FlyGenome],
    fitness: list[float],
    *,
    elite_k: int = 3,
    mutation: float = DEFAULT_MUTATION,
    seed: int = 7,
) -> list[FlyGenome]:
    """Элита без изменений + турнирный отбор, скрещивание, мутации, свежая кровь."""
    n = len(flies)
    if n < 2:
        return [f for f in flies]
    rng = np.random.default_rng(seed)
    order = np.argsort(np.asarray(fitness))
    elite_k = max(1, min(elite_k, n - 1))
    nxt: list[FlyGenome] = [flies[int(i)] for i in order[:elite_k]]

    def tournament() -> FlyGenome:
        i, j = rng.integers(0, n, 2)
        return flies[int(i)] if fitness[int(i)] <= fitness[int(j)] else flies[int(j)]

    while len(nxt) < n:
        if rng.random() < 0.08:
            nxt.append(init_population(2, seed=int(rng.integers(0, 1 << 30)))[rng.integers(0, 2)])
            continue
        child = _crossover(tournament(), tournament(), rng)
        nxt.append(_mutate(child, mutation, rng))
    return nxt[:n]
