"""Цикл прогона: ГСН → (законы | BIO) → интеграция. BOTH считает призрак ПН.

Разделение честности:
- «oracle»-законы (pn/apn/pure/clos/pn_sched_oracle) используют точную геометрию
  r, v — это эталоны с подсказкой;
- «pn_gsn», «pn_sched_sensor» и BIO получают ТОЛЬКО декодированные из изображения
  измерения (после шумов, отказов и задержки) — сенсорные участники.

N_eff — диагностическая метрика (pn.n_eff_from): для КАЖДОГО кадра команда
сравнивается с базисом ПН при N=1 на ИСТИННОЙ геометрии. Истинная геометрия
используется постфактум для анализа и никогда не поступает в сенсорные законы
или BIO.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from navedenie.brain_store import get_circuit, get_evader_circuit
from navedenie.circuit import FlyCircuit
from navedenie.evader import evader_accel
from navedenie.evader_brain import EVADER_BRAIN_LAW, EvaderBrainSensor
from navedenie.pn import (
    apn_accel,
    clos_accel,
    n_eff_from,
    pn_seeker_accel,
    ppn_accel,
    pure_pursuit_accel,
    scheduled_pn_accel,
    scheduled_pn_seeker_accel,
    tpn_accel,
)
from navedenie.physics import FlightState, PhysicsParams, mach_at, n_available, rho_at_h, step_physics
from navedenie.seeker import ObservationBuffer, body_axes, dead_mask_for, observe
from navedenie.sim import (
    Body,
    Frame,
    RunResult,
    Scenario,
    _corr,
    _percentile,
    _perp,
    clip_accel,
    closing_speed,
    integrate,
    integrate_target,
    los_omega,
    radial_time,
    spawn,
    step_encounter,
    target_accel,
    time_to_cpa,
)
from navedenie.sim import G


def _seeker_metrics(r, v_m, v_t) -> tuple[float, float, float]:
    rng = float(np.linalg.norm(r)) + 1e-12
    vc = closing_speed(r, v_t - v_m)
    om = float(np.linalg.norm(los_omega(r, v_t - v_m)))
    return rng, vc, om


def _terminal_zem(m_p, m_v, t_p, t_v) -> float:
    """h_cv — прогноз минимального расстояния при неизменных текущих скоростях.

    Прямолинейная геометрическая экстраполяция: оба вектора скорости считаются
    постоянными, сближение до t_cpa. Это НЕ h₀ (промах при нулевой команде):
    в физической модели h₀ требует интегрирования free-полёта. В кинематической
    модели (постоянная скорость, нет сил) h₀ ≡ h_cv, поэтому legacy-поле
    `terminal_zem_m` сохраняет именно это значение.
    Русское название: «Прогноз минимального расстояния при неизменных скоростях»."""
    r = t_p - m_p
    vr = t_v - m_v
    t_cpa = max(0.0, time_to_cpa(r, vr))
    pip = t_p + t_v * t_cpa
    mz = m_p + m_v * t_cpa
    return float(np.linalg.norm(pip - mz))


def _phys_params_from_scenario(sc: Scenario) -> PhysicsParams:
    """Параметры трёхстепенной модели из сценария (общий источник для run и h₀)."""
    return PhysicsParams(
        mass_kg=sc.phys_mass_kg, ref_area_m2=sc.phys_ref_area_m2, drag_cx=sc.phys_drag_cx,
        rho_air=sc.phys_rho_air, thrust_n=sc.phys_thrust_n, burn_time_s=sc.phys_burn_time_s,
        tau_a_s=sc.phys_tau_a_s, n_avail_max=sc.phys_n_avail_max, da_dt_max=sc.phys_da_dt_max,
        atmos=bool(getattr(sc, "phys_atmos", False)),
        wn_act=float(getattr(sc, "phys_wn_act", 0.0)),
        zeta_act=float(getattr(sc, "phys_zeta_act", 1.0)),
        cx_wave=float(getattr(sc, "phys_cx_wave", 0.0)),
        mach_kr=float(getattr(sc, "phys_mach_kr", 1.0)),
        mach_band=float(getattr(sc, "phys_mach_band", 0.10)),
        cn_max=float(getattr(sc, "phys_cn_max", 0.0)),
        g=G if sc.phys_gravity else 0.0,
    )


def _h0_from(model: str, m_p, m_v, t_p, t_v, dt: float,
             phys_params: PhysicsParams | None) -> float:
    """h₀ — прогнозируемый промах при нулевой дальнейшей команде.

    Свободное интегрирование МОДЕЛИ ДВИЖЕНИЯ ракеты вперёд (a_cmd = 0) при
    инерционном (постоянной скорости) движении цели; минимум расстояния. В
    кинематической модели совпадает с h_cv (нет сил, скорость постоянна). В
    физической — отличается: работают тяжесть/сопротивление/тяга и откат
    исполнительного контура. Русское название: «Прогнозируемый промах при
    нулевой дальнейшей команде»."""
    if model != "point_mass_3dof" or phys_params is None:
        return _terminal_zem(m_p, m_v, t_p, t_v)
    rng0 = float(np.linalg.norm(t_p - m_p))
    vc = max(closing_speed(t_p - m_p, t_v - m_v), 50.0)
    horizon = min(30.0, 3.0 * rng0 / vc + 1.0)
    h = max(dt, 1e-3)
    st = FlightState(p=np.asarray(m_p, dtype=float).copy(),
                     v=np.asarray(m_v, dtype=float).copy())
    zero = np.zeros(3)
    best = float(np.linalg.norm(t_p - st.p))
    prev = best
    tt = 0.0
    steps = max(1, int(round(horizon / h)))
    for i in range(steps):
        st = step_physics(st, zero, phys_params, tt, h)
        tt += h
        rng = float(np.linalg.norm((t_p + t_v * tt) - st.p))
        if rng < best:
            best = rng
        elif i > 2 and rng > prev:
            break  # прошли точку ближайшего сближения
        prev = rng
    return float(best)


def _neff_components(a_cmd: np.ndarray, q: np.ndarray, v_m: np.ndarray, q_min_axis: float = 0.25) -> tuple[float | None, float | None]:
    """Компонентные оценки N_eff по осям корпуса (рыскание/тангаж). None — ось
    вырождена (проекция базиса на ось слишком мала)."""
    _x, y, z = body_axes(v_m)
    qy, qz = float(np.dot(q, y)), float(np.dot(q, z))
    qn = float(np.linalg.norm(q)) + 1e-12
    n_yaw = float(np.dot(a_cmd, y)) / qy if abs(qy) > q_min_axis * qn else None
    n_pitch = float(np.dot(a_cmd, z)) / qz if abs(qz) > q_min_axis * qn else None
    return n_yaw, n_pitch


def run(sc: Scenario, circuit: FlyCircuit | None = None, evader_circuit: FlyCircuit | None = None) -> Iterator[Frame]:
    missile, target = spawn(sc)
    launch_p = missile.p.copy()  # точка пуска — для CLOS (3-точка)
    # физическая трёхстепенная модель ракеты (цель и призрак остаются кинематическими)
    phys = sc.model == "point_mass_3dof"
    phys_params: PhysicsParams | None = None
    flight: FlightState | None = None
    if phys:
        phys_params = _phys_params_from_scenario(sc)
        flight = FlightState(p=missile.p.copy(), v=missile.v.copy())
    # инерция рулевого привода кинематического контура (1-е звено); призрак не затронут
    lag_active = (not phys) and float(getattr(sc, "tau_act_s", 0.0)) > 1e-12
    a_exec = np.zeros(3)
    # призрак: эталонная ПН летит ту же цель с той же стартовой точки —
    # метрика «похожести на oracle-закон» и его время наведения
    ghost = Body(missile.p.copy(), missile.v.copy())
    # начальная истинная дальность — масштаб порога события «lost»
    # (в «свободной расстановке» range_m не задаёт спавн — дальность даёт сама геометрия)
    range_init = float(np.linalg.norm(target.p - missile.p))
    if circuit is None:
        circuit = get_circuit(getattr(sc, "brain", "stub"), tau_s=sc.tau_s, gain=sc.circuit_gain)
    prev_obs: dict | None = None
    cpa = 1e9
    t = 0.0
    lock_steps = 0
    steps = 0
    n_int = 0.0
    n_peak = 0.0
    event: str | None = "launch"
    prev_locked = True
    ghost_t_hit: float | None = None
    noise_rng = np.random.default_rng(sc.seed * 977 + 13)
    # ПОСТОЯННАЯ карта отказов: ровно одна на прогон, seed-стабильная
    dead_mask = dead_mask_for(sc.retina_death_p, noise_rng)

    # ГСН BIO видит широкое фовеальное поле; oracle-ПН — узкий кадр (её «прибор»)
    half_fov = np.deg2rad(sc.bio_fov_deg if sc.mode in {"bio", "both"} else sc.fov_deg) / 2.0
    buffer = ObservationBuffer(sc.seeker_delay_s, sc.seeker_jitter_s, sc.dt, noise_rng)
    # ── дуэль: «боевая жизнь» ракеты короче общего t_max — не взяла за fuse_life_s,
    # значит выдохлась и проиграла честно (финал reason="fuse_expired")
    t_cap = min(sc.t_max, float(sc.fuse_life_s)) if sc.duel else sc.t_max
    # мозг-уклонист: цель с обучаемой схемой (фаза 3 дуэли) — свой сенсор и контур
    ev_brain: EvaderBrainSensor | None = None
    if sc.duel and sc.evader_law == EVADER_BRAIN_LAW:
        brain = evader_circuit if evader_circuit is not None else get_evader_circuit(
            tau_s=sc.tau_s, gain=sc.circuit_gain
        )
        ev_brain = EvaderBrainSensor(sc, brain, seed=int(sc.seed) + 5)
    while t < t_cap:
        r = target.p - missile.p
        rng, vc, om = _seeker_metrics(r, missile.v, target.v)
        obs = observe(
            r,
            missile.v,
            prev_obs,
            half_fov=half_fov,
            dt_s=sc.dt,
            noise_az=np.deg2rad(sc.noise_az_deg),
            noise_range=sc.noise_range_m,
            lock_drop_p=sc.lock_drop_p,
            dead_mask=dead_mask,
            dropout_p=sc.retina_dropout_p,
            rng=noise_rng,
            brightness=getattr(sc, "target_brightness", 1.0),
        )
        obs["retina_alive"] = 1.0 - sc.retina_death_p
        prev_obs = obs
        buffer.push(t, obs)
        delayed = buffer.sample(t)
        if delayed.get("lock"):
            lock_steps += 1

        a_pn = ppn_accel(r, missile.v, target.v, sc.pn_n, sc.n_max)
        a_ghost = ppn_accel(target.p - ghost.p, ghost.v, target.v, sc.pn_n, sc.n_max)
        # фактическое нормальное ускорение цели этого шага: реактивный уклонист
        # в дуэли, прежний слепой манёвр — в обычном прогоне (APN и интеграция
        # цели берут ОДНО и то же a_t — закон цели честен для обоих контуров)
        if ev_brain is not None:
            a_t = ev_brain.accel(target, missile, t, sc.dt)
        elif sc.duel:
            a_t = evader_accel(target, missile, sc)
        else:
            a_t = target_accel(target, sc, t)
        circuit.step(delayed, sc.dt)
        a_bio = circuit.accel_cmd(missile.v, sc.n_max)

        if sc.mode == "pn":
            if sc.law == "tpn":
                # истинная ПН: команда N·V_c·ω по нормали к линии визирования
                a_cmd = tpn_accel(r, missile.v, target.v, sc.pn_n, sc.n_max)
            elif sc.law == "apn":
                a_cmd = apn_accel(r, missile.v, target.v, a_t, sc.pn_n, sc.n_max)
            elif sc.law == "pure":
                a_cmd = pure_pursuit_accel(r, missile.v, sc.n_max)
            elif sc.law == "clos":
                a_cmd = clos_accel(missile.p, r, missile.v, target.p, launch_p, sc.n_max)
            elif sc.law == "pn_gsn":
                # сенсорная ПН: декодированные угловые скорости, сглаженные фильтром
                # ГСН (0.1 с) — без фильтра квантование фовеи уводит команду в насыщение
                a_cmd = pn_seeker_accel(delayed["az_dot_s"], delayed["el_dot_s"], missile.v, sc.pn_n, sc.n_max) if delayed["lock"] else np.zeros(3)
            elif sc.law == "pn_sched_oracle":
                # scheduled PN, oracle-вариант: истинное Vc/R ≡ 1/t_go (подсказка геометрией)
                rho_true = vc / max(rng, 1.0)
                a_cmd = scheduled_pn_accel(r, missile.v, target.v, rho_true, sc.pn_sched_n0, sc.pn_sched_k_rho, sc.pn_sched_n_min, sc.pn_sched_n_max, sc.n_max)
            elif sc.law == "pn_sched_sensor":
                # scheduled PN, сенсорный вариант: N из ИЗМЕРЕННОГО rho, команда из
                # сглаженных декодированных угловых скоростей — pn_gsn + планирование N
                a_cmd = (
                    scheduled_pn_seeker_accel(
                        delayed["az_dot_s"], delayed["el_dot_s"], delayed["rho"],
                        missile.v, sc.pn_sched_n0, sc.pn_sched_k_rho,
                        sc.pn_sched_n_min, sc.pn_sched_n_max, sc.n_max,
                    )
                    if delayed["lock"]
                    else np.zeros(3)
                )
            else:
                a_cmd = a_pn  # oracle
        elif sc.mode == "bio":
            a_cmd = a_bio if delayed["lock"] else np.zeros(3)
        else:
            a_cmd = a_bio if delayed["lock"] else a_pn

        a_cmd = clip_accel(a_cmd, sc.n_max)
        n_req = float(np.linalg.norm(a_cmd) / 9.81)

        # ── диагностика N_eff: истинная геометрия — постфактум, НЕ для управления
        n_eff, n_eff_reason, q_basis = n_eff_from(a_cmd, r, missile.v, target.v, n_max=sc.n_max)
        n_yaw = n_pitch = None
        if n_eff is not None:
            n_yaw, n_pitch = _neff_components(a_cmd, q_basis, missile.v)
        sat = n_req >= 0.98 * sc.n_max

        # шаг интегрирования: кинематическая точка-масса (|v| сохраняется) либо
        # трёхстепенная физическая модель (тяга/сопротивление/тяжесть + инерция контура)
        p_m0, p_t0 = missile.p.copy(), target.p.copy()
        v_m0, v_t0 = missile.v.copy(), target.v.copy()
        g0 = ghost.p.copy()
        integrate_target(target, sc, t, sc.dt, a_norm=a_t)
        n_cmd = n_act = n_avail = speed_ms = None
        atm_rho: float | None = None
        mach_val: float | None = None
        if phys and flight is not None and phys_params is not None:
            n_cmd = float(np.linalg.norm(a_cmd) / 9.81)
            n_avail = n_available(flight.speed, float(flight.p[2]), phys_params)
            # плотность, которая будет применена в сопротивлении этого шага
            atm_rho = rho_at_h(phys_params, float(flight.p[2]))
            flight = step_physics(flight, a_cmd, phys_params, t, sc.dt)
            missile.p, missile.v = flight.p, flight.v
            n_act = float(np.linalg.norm(flight.a_act) / 9.81)
            speed_ms = flight.speed
            # M считаем по post-step скорости — в одном ряду с speed_ms
            mach_val = mach_at(speed_ms, float(flight.p[2]))
        else:
            # кинематический контур: опциональное 1-е апериодическое звено привода
            # (τ·a_act' + a_act = a_cmd, неявная устойчивая дискретизация — как в physics.py).
            # Призрак-эталон остаётся с идеальным исполнением: метрика ref_dev честно
            # включает ошибку, вносимую лагом привода.
            if lag_active:
                k_act = sc.dt / (sc.tau_act_s + sc.dt)
                a_exec = _perp(a_exec + (a_cmd - a_exec) * k_act, missile.v)
                n_cmd = float(np.linalg.norm(a_cmd) / 9.81)
                n_act = float(np.linalg.norm(a_exec) / 9.81)
                integrate(missile, a_exec, sc.dt)
            else:
                integrate(missile, a_cmd, sc.dt)
        integrate(ghost, a_ghost, sc.dt)

        # непрерывная встреча на шаге: CPA и пересечение сферы БЧ
        enc = step_encounter(p_m0, missile.p, p_t0, target.p, sc.kill_radius_m)
        cpa = min(cpa, enc["cpa"])
        # призрак летит параллельно — его встречу с целью тоже считаем непрерывно
        # (предыдущая точка — собственная позиция призрака g0, а не ракеты p_m0)
        if ghost_t_hit is None:
            enc_g = step_encounter(g0, ghost.p, p_t0, target.p, sc.kill_radius_m)
            if enc_g["hit"]:
                ghost_t_hit = t + float(enc_g["alpha_in"]) * sc.dt
        n_int += n_req * sc.dt
        n_peak = max(n_peak, n_req)
        steps += 1

        snap = circuit.state.snapshot(kind=circuit.kind, n_cells=circuit.n_cells, trained=circuit.trained)
        seeker_dict = {
            "image": np.asarray(delayed["image"]).tolist(),
            "size": delayed["size"],
            "az_dot": delayed["az_dot"],
            "el_dot": delayed["el_dot"],
            "az": delayed["az"],
            "el": delayed["el"],
            "theta": delayed["theta"],
            "theta_dot": delayed["theta_dot"],
            "rho": delayed["rho"],
            "tau_contact": delayed["tau_contact"],
        }
        v_rel = v_t0 - v_m0  # относительная скорость в состоянии кадра (до шага)
        t_radial_diag = radial_time(r, v_rel)  # R/Vc, с (None при отсутствии сближения)
        t_cpa_diag = time_to_cpa(r, v_rel)  # время до ближайшего сближения, с

        frame_kw = dict(
            a_cmd=a_cmd.copy(),
            a_pn=a_pn.copy(),
            n_req=n_req,
            n_lim=sc.n_max,
            range_m=rng,
            v_c=vc,
            omega_los=om,
            az=float(obs["truth_az"]),
            el=float(obs["truth_el"]),
            lock=bool(delayed["lock"]),
            miss=cpa,
            seeker=seeker_dict,
            circuit=snap,
            evader=ev_brain.snapshot() if ev_brain is not None else None,
            layers=snap["layers"],
            ghost_t_hit=ghost_t_hit,
            theta=float(delayed["theta"]),
            theta_dot=float(delayed["theta_dot"]),
            rho=float(delayed["rho"]),
            tau_contact=float(delayed["tau_contact"]),
            tgo=t_radial_diag,  # DEPRECATED alias = t_radial
            t_radial=t_radial_diag,
            t_cpa=t_cpa_diag,
            n_eff=n_eff,
            n_eff_valid=n_eff is not None,
            n_eff_reason=n_eff_reason,
            n_eff_yaw=n_yaw,
            n_eff_pitch=n_pitch,
            sat=sat,
            n_cmd=n_cmd,
            n_act=n_act,
            n_avail=n_avail,
            speed_ms=speed_ms,
            atm_rho=atm_rho,
            mach=mach_val,
            speed_mode=sc.target_speed_mode,
            target_speed=float(np.linalg.norm(target.v)),
        )

        if enc["hit"]:
            t_hit_step = t + float(enc["alpha_in"]) * sc.dt
            yield Frame(
                t=t_hit_step,
                missile=missile.p.copy(),
                target=target.p.copy(),
                missile_v=missile.v.copy(),
                target_v=target.v.copy(),
                event="hit",
                **frame_kw,
            )
            break

        yield Frame(
            t=t,
            missile=p_m0,
            target=p_t0,
            missile_v=v_m0,
            target_v=v_t0,
            ghost=g0,
            event=event,
            **frame_kw,
        )
        event = None

        if t > 0.6 and vc < 0 and rng > cpa + 50:
            yield Frame(
                t=t,
                missile=missile.p.copy(),
                target=target.p.copy(),
                missile_v=missile.v.copy(),
                target_v=target.v.copy(),
                ghost=ghost.p.copy(),
                event="miss_pass",
                **frame_kw,
            )
            return

        t += sc.dt
        if prev_locked and not delayed["lock"] and t > 1.0 and rng > range_init * 0.2:
            event = "lost"
        prev_locked = bool(delayed["lock"])


def collect(
    sc: Scenario, stride: int = 8, circuit: FlyCircuit | None = None, evader_circuit: FlyCircuit | None = None,
    on_frame=None,
) -> RunResult:
    """Прогон до конца → метрики. Необязательный коллбэк on_frame(i, fr, keep)
    вызывается на каждом кадре (keep — тот же критерий отбора в выдачу, что и
    append в frames) — для потоковой передачи по websocket, чтобы сцена
    трогалась, пока хвост траектории ещё считается; без него collect —
    бит-в-бит прежний."""
    frames: list[Frame] = []
    last: Frame | None = None
    locks = 0
    total = 0
    n_int = 0.0
    n_peak = 0.0
    lock_time = 0.0
    dev_sum = 0.0
    dev_sq = 0.0
    dev_n = 0
    range_sum = 0.0
    range0: float | None = None
    ghost_hit_t: float | None = None
    hit = False
    t_hit: float | None = None
    # диагностика N_eff по ВСЕМ кадрам (не только прореженным)
    _neff_vals: list[float] = []
    _neff_rho: list[float] = []
    _neff_tgo: list[float] = []
    _neff_steps = 0
    _neff_sat = 0
    # argmin накопленного CPA — момент наибольшего сближения для η (угол встречи)
    _best_miss = 1e18
    _eta_vm = None
    _eta_vt = None
    for i, fr in enumerate(run(sc, circuit=circuit, evader_circuit=evader_circuit)):
        last = fr
        total += 1
        if fr.miss < _best_miss - 1e-12:
            _best_miss = fr.miss
            _eta_vm, _eta_vt = fr.missile_v, fr.target_v
        _neff_steps += 1
        _neff_sat += int(fr.sat)
        if fr.n_eff_valid and fr.n_eff is not None and np.isfinite(fr.n_eff):
            _neff_vals.append(fr.n_eff)
            _neff_rho.append(fr.rho)
            _neff_tgo.append(fr.tgo if fr.tgo is not None else float("nan"))
        if fr.event == "hit":
            hit = True
            t_hit = fr.t
        # дубли-кадры терминальных событий не попадают в интегралы дважды
        is_terminal = fr.event in ("hit", "miss_pass")
        if not is_terminal:
            n_int += fr.n_req * sc.dt
            n_peak = max(n_peak, fr.n_req)
            if fr.lock:
                locks += 1
                lock_time += sc.dt
            if fr.ghost is not None:
                d = fr.missile - fr.ghost
                dev_sum += float(np.linalg.norm(d))
                dev_sq += float(np.dot(d, d))
                dev_n += 1
                if range0 is None:
                    range0 = float(fr.range_m)
                range_sum += float(np.linalg.norm(fr.missile - fr.target))
                if ghost_hit_t is None and float(np.linalg.norm(fr.target - fr.ghost)) < sc.kill_radius_m:
                    ghost_hit_t = fr.t
        keep = i % stride == 0 or bool(fr.event)
        if on_frame is not None:
            on_frame(i, fr, keep)
        if keep:
            frames.append(fr)
        if fr.event == "hit":
            break
    miss = last.miss if last else 1e9  # = непрерывный CPA
    hit_final = hit or miss <= sc.kill_radius_m
    ghost_hit_t = last.ghost_t_hit if (last is not None and last.ghost_t_hit is not None) else None
    ref_rms = float(np.sqrt(dev_sq / max(dev_n, 1))) if dev_n else None
    ref_nrms = (
        float(np.sqrt(dev_sq / max(dev_n, 1)) / range0) if dev_n and range0 and range0 > 0 else None
    )
    _pair_rho = [(v, r) for v, r in zip(_neff_vals, _neff_rho) if np.isfinite(r)]
    _pair_tgo = [(v, tg) for v, tg in zip(_neff_vals, _neff_tgo) if np.isfinite(tg)]
    _neff_agg = {
        "n_eff_median": _percentile(_neff_vals, 50),
        "n_eff_q25": _percentile(_neff_vals, 25),
        "n_eff_q75": _percentile(_neff_vals, 75),
        "n_eff_min": min(_neff_vals) if _neff_vals else None,
        "n_eff_max": max(_neff_vals) if _neff_vals else None,
        "n_eff_valid_frac": (_neff_val_count := len(_neff_vals)) / max(_neff_steps, 1),
        "sat_frac": _neff_sat / max(_neff_steps, 1),
        "corr_n_eff_rho": _corr([p[0] for p in _pair_rho], [p[1] for p in _pair_rho]),
        "corr_n_eff_tgo": _corr([p[0] for p in _pair_tgo], [p[1] for p in _pair_tgo]),
        "n_eff_count": _neff_val_count,
    }
    t_end = last.t if last else None
    # дуэль: ракета «выдохлась» — прогон дошёл до конца боевой жизни без перехвата
    _t_cap = min(sc.t_max, float(sc.fuse_life_s)) if sc.duel else sc.t_max
    fuse_expired = bool(
        sc.duel and not hit_final and t_end is not None and t_end >= _t_cap - 1.5 * sc.dt
    )
    h_cv = _terminal_zem(last.missile, last.missile_v, last.target, last.target_v) if last else None
    h0 = (_h0_from(sc.model, last.missile, last.missile_v, last.target, last.target_v,
                  sc.dt, _phys_params_from_scenario(sc) if sc.model == "point_mass_3dof" else None)
          if last else None)
    end_range = float(np.linalg.norm(last.target - last.missile)) if last else None
    # η — угол между векторами скоростей в момент наибольшего сближения
    impact_angle: float | None = None
    if _eta_vm is not None and _eta_vt is not None:
        n_m = float(np.linalg.norm(_eta_vm))
        n_t = float(np.linalg.norm(_eta_vt))
        if n_m > 1e-9 and n_t > 1e-9:
            cos_eta = float(np.dot(_eta_vm, _eta_vt)) / (n_m * n_t)
            impact_angle = round(
                float(np.degrees(np.arccos(max(-1.0, min(1.0, cos_eta))))), 2
            )
    return RunResult(
        frames=frames,
        miss_m=miss,
        cpa_m=miss,
        trigger_range_m=sc.kill_radius_m,
        hit=hit_final,
        t_hit=t_hit,
        t_end=t_end,
        fov_lock_frac=locks / max(total, 1),
        lock_time_s=lock_time,
        lock_fraction=lock_time / max(t_end, 1e-9),
        n_int=n_int,
        n_mean_g=n_int / max(t_end, 1e-9),
        n_peak=n_peak,
        reason="hit" if hit_final else ("fuse_expired" if fuse_expired else "miss"),
        t_guide=t_hit,
        ref_dev_m=dev_sum / max(dev_n, 1) if dev_n else None,
        ref_rms_m=ref_rms,
        ref_nrms=ref_nrms,
        t_ref=ghost_hit_t,
        terminal_zem_m=h_cv,
        h_cv_m=h_cv,
        h0_m=h0,
        end_range_m=end_range,
        impact_angle_deg=impact_angle,
        model=sc.model,
        duel=bool(sc.duel),
        duel_result=(None if not sc.duel else ("missile" if hit_final else "evader")),
        fuse_expired=fuse_expired,
        t_survived=t_end if sc.duel else None,
        **_neff_agg,
    )


def frame_to_dict(fr: Frame) -> dict:
    return {
        "t": round(fr.t, 4),
        "missile": fr.missile.tolist(),
        "target": fr.target.tolist(),
        "missile_v": fr.missile_v.tolist(),
        "target_v": fr.target_v.tolist(),
        "a_cmd": fr.a_cmd.tolist(),
        "a_pn": fr.a_pn.tolist(),
        "n_req": fr.n_req,
        "n_lim": fr.n_lim,
        "range_m": fr.range_m,
        "v_c": fr.v_c,
        "omega_los": fr.omega_los,
        "az": fr.az,
        "el": fr.el,
        "lock": fr.lock,
        "miss": fr.miss,
        "seeker": fr.seeker,
        "circuit": fr.circuit,
        "evader": fr.evader,
        "layers": fr.layers,
        "event": fr.event,
        "ghost": fr.ghost.tolist() if fr.ghost is not None else None,
        "theta": fr.theta,
        "theta_dot": fr.theta_dot,
        "rho": fr.rho,
        "tau_contact": fr.tau_contact,
        "tgo": fr.tgo,  # DEPRECATED alias = t_radial
        "t_radial": fr.t_radial,
        "t_cpa": fr.t_cpa,
        "n_eff": fr.n_eff,
        "n_eff_valid": fr.n_eff_valid,
        "n_eff_reason": fr.n_eff_reason,
        "n_eff_yaw": fr.n_eff_yaw,
        "n_eff_pitch": fr.n_eff_pitch,
        "sat": fr.sat,
        "n_cmd": fr.n_cmd,
        "n_act": fr.n_act,
        "n_avail": fr.n_avail,
        "speed_ms": fr.speed_ms,
        "atm_rho": fr.atm_rho,
        "mach": fr.mach,
        "speed_mode": fr.speed_mode,
        "target_speed": fr.target_speed,
    }
