"""Честная диагностика возникшего закона управления (постфактум).

Истинная геометрия (дальность, Vc, t_go, базис ПН q) используется ЗДЕСЬ и
только для анализа: ни один признак отсюда не поступает в BIO или сенсорные
законы во время управления.

P6 — разложение команды: a_bio = N·q + residual.
  Покадровое N_eff = ⟨a,q⟩/⟨q,q⟩ плохо обусловлено при малом |q|; вместо
  интерпретации его IQR считаются оконная WLS-оценка N_wls с даун-вейтом
  вырожденной геометрии, косинус угла «команда–базис», доля объяснённой
  команды, относительный остаток, знак N и доли кадров (малое q, насыщение,
  нет захвата, плохое направление) + bootstrap CI ПО ЭПИЗОДАМ.
P7 — сравнение гипотез H1–H5 (разбиение по эпизодам ЦЕЛИКОМ).
P8 — closed-loop проверка: deployable-закон ставится в контур вместо BIO
  и летит held-out; diagnostic_geometry_model в контур не ставится.
P9 — фотометрическая устойчивость (brightness ≠ 1) и matched-абляции.

Единицы (P11): rho — 1/с; k_rho в N = N0 + k_rho·rho — секунды;
q — м/с² на единицу N; «закон наведения» — только для closed-loop
подтверждённых моделей, иначе «суррогат команды»/«локальная аппроксимация».
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from navedenie.circuit import FEAT_DIM, _features
from navedenie.formula import _poly_design
from navedenie.pn import n_eff_basis, pn_seeker_accel, scheduled_pn_seeker_accel
from navedenie.seeker import ObservationBuffer, body_axes, dead_mask_for, observe
from navedenie.sim import G, Scenario, clip_accel, integrate, integrate_target, spawn, step_encounter


# ── P6: полночастотная запись эпизода ────────────────────────────────────────


@dataclass
class EpisodeTrace:
    """Полночастотные записи эпизода: команда BIO + истинная геометрия (анализ)
    + сенсорные признаки контура (для deployable-гипотез)."""

    t: np.ndarray
    a_cmd: np.ndarray  # (N,3)
    r: np.ndarray  # (N,3) истинное относительное положение (ДО шага, только анализ)
    v_m: np.ndarray  # (N,3)
    v_t: np.ndarray  # (N,3)
    lock: np.ndarray  # (N,) bool
    sat: np.ndarray  # (N,) bool
    rho_meas: np.ndarray  # (N,) измеренный rho сенсора, 1/с
    feat: np.ndarray  # (N,10) признаки контура
    hit: bool
    geometric_cpa_m: float
    meta: dict[str, Any] = field(default_factory=dict)

    def q_basis(self) -> np.ndarray:
        """Базис ПН q на истинной геометрии (N,3) — постфактум, не для управления."""
        out = np.zeros_like(self.a_cmd)
        for i in range(self.a_cmd.shape[0]):
            out[i] = n_eff_basis(self.r[i], self.v_m[i], self.v_t[i])
        return out


def episode_trace(circuit, sc: Scenario, *, t_cap: float = 16.0, want_feats: bool = True) -> EpisodeTrace:
    """Полночастотная запись эпизода (lr=0, teacher выключен). Тот же сенсорный
    контур, что в law.evaluate_episode; запись — каждый шаг интегрирования."""
    missile, target = spawn(sc)
    circuit.reset()
    prev: dict | None = None
    t = 0.0
    dt = float(sc.dt)
    half_fov = np.deg2rad(max(getattr(sc, "bio_fov_deg", 165.0), 2.0)) / 2.0
    ts: list[float] = []
    a_list: list[np.ndarray] = []
    r_list: list[np.ndarray] = []
    vm_list: list[np.ndarray] = []
    vt_list: list[np.ndarray] = []
    lock_l: list[bool] = []
    sat_l: list[bool] = []
    rho_l: list[float] = []
    feat_l: list[np.ndarray] = []
    hit = False
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
        circuit.step(delayed, dt)
        a_cmd = circuit.accel_cmd(missile.v, sc.n_max) if delayed["lock"] else np.zeros(3)
        a_cmd = clip_accel(a_cmd, sc.n_max)
        ts.append(t)
        a_list.append(a_cmd.copy())
        r_list.append(r.copy())
        vm_list.append(missile.v.copy())
        vt_list.append(target.v.copy())
        lock_l.append(bool(delayed["lock"]))
        sat_l.append(float(np.linalg.norm(a_cmd)) >= sat_lim)
        rho_l.append(float(delayed.get("rho", 0.0)))
        if want_feats:
            fv = getattr(circuit, "feat", None)
            if fv is None:
                fv = getattr(getattr(circuit, "state", None), "feat", None)
            feat_l.append(np.asarray(fv, dtype=np.float64).copy() if fv is not None and np.size(fv) else np.zeros(FEAT_DIM))

        p_m0, p_t0 = missile.p.copy(), target.p.copy()
        integrate_target(target, sc, t, dt)
        integrate(missile, a_cmd, dt)
        enc = step_encounter(p_m0, missile.p, p_t0, target.p, sc.kill_radius_m)
        if enc["hit"]:
            hit = True
            break
        if enc["cpa"] < geo_cpa:
            geo_cpa = enc["cpa"]
        if t > 0.5 and float(-np.dot(r, target.v - missile.v) / (np.linalg.norm(r) + 1e-9)) < 0 and np.linalg.norm(r) > geo_cpa + 80:
            break
        t += dt
    return EpisodeTrace(
        t=np.asarray(ts),
        a_cmd=np.asarray(a_list),
        r=np.asarray(r_list),
        v_m=np.asarray(vm_list),
        v_t=np.asarray(vt_list),
        lock=np.asarray(lock_l, dtype=bool),
        sat=np.asarray(sat_l, dtype=bool),
        rho_meas=np.asarray(rho_l),
        feat=np.asarray(feat_l) if feat_l else np.zeros((0, FEAT_DIM)),
        hit=hit,
        geometric_cpa_m=float(geo_cpa),
        meta={"v_m": float(sc.v_m), "v_t": float(sc.v_t), "aspect": str(sc.aspect), "speed_mode": str(sc.target_speed_mode), "maneuver": str(sc.maneuver)},
    )


# ── P6: разложение a_bio = N·q + residual ─────────────────────────────────────


def _wls_window(a: np.ndarray, q: np.ndarray, q_min: float) -> dict[str, float]:
    """Окно WLS: вес кадра w = qn²/(qn² + q_min²) — плавный даун-вейт вырожденной
    геометрии: при |q|→0 оценка N_wls НЕ взрывается (проблема N_eff решена весом).
    Возвращает N_wls и медианы окна: alignment, explained, relative residual."""
    qn2 = np.einsum("ij,ij->i", q, q)
    w = qn2 / (qn2 + q_min * q_min)
    aq = np.einsum("ij,ij->i", a, q)
    denom = float(np.sum(w * qn2))
    n_wls = float(np.sum(w * aq) / denom) if denom > 1e-9 else 0.0
    amag = np.linalg.norm(a, axis=1)
    qn = np.sqrt(qn2)
    with np.errstate(invalid="ignore"):
        cos = np.where(amag > 1e-6, aq / np.maximum(amag * qn, 1e-9), np.nan)
    pred = n_wls * q
    resid = np.linalg.norm(a - pred, axis=1) / np.maximum(amag, 1e-9)
    expl = np.where(amag > 1e-9, np.einsum("ij,ij->i", pred, a) / np.maximum(amag**2, 1e-12), np.nan)

    def _median(xs: np.ndarray) -> float:
        # все-NaN окно (команда нулевая во всём эпизоде) — NaN без RuntimeWarning
        return float(np.nanmedian(xs)) if bool(np.isfinite(xs).any()) else float("nan")

    return {
        "n_wls": n_wls,
        "alignment": _median(cos),
        "explained": _median(expl),
        "residual": _median(resid),
    }


def _r(x: float | None, d: int = 4) -> float | None:
    return round(float(x), d) if x is not None and np.isfinite(x) else None


def decompose_episode(tr: EpisodeTrace, *, window: int = 25, q_min: float = 0.5) -> dict[str, Any]:
    """Разложение команды эпизода по базису ПН: оконная N_wls, alignment,
    explained fraction, относительный остаток, знак, доли кадров."""
    q = tr.q_basis()
    a = tr.a_cmd
    n = len(tr.t)
    qn = np.linalg.norm(q, axis=1)
    amag = np.linalg.norm(a, axis=1)
    aq = np.einsum("ij,ij->i", a, q)
    with np.errstate(invalid="ignore"):
        alignment = np.where(amag > 1e-6, aq / np.maximum(amag * qn, 1e-9), np.nan)

    win: list[dict[str, float]] = []
    step_res = np.full(n, np.nan)
    stride = max(window // 2, 1)
    for s in range(0, n, stride):
        e = min(s + window, n)
        if e - s < 3:
            continue
        w_res = _wls_window(a[s:e], q[s:e], q_min)
        win.append(w_res)
        pred = w_res["n_wls"] * q[s:e]
        step_res[s:e] = np.linalg.norm(a[s:e] - pred, axis=1) / np.maximum(amag[s:e], 1e-9)

    med_n = float(np.median([w["n_wls"] for w in win])) if win else None
    med_res = float(np.nanmedian(step_res)) if np.isfinite(step_res).any() else None
    med_al = float(np.nanmedian(alignment)) if np.isfinite(alignment).any() else None
    med_ex = float(np.median([w["explained"] for w in win])) if win else None
    sign_consistency = (
        float(np.mean([np.sign(w["n_wls"]) == np.sign(med_n) for w in win if w["n_wls"] != 0.0]))
        if win and med_n
        else None
    )
    return {
        "n_steps": int(n),
        "n_windows": len(win),
        "n_wls_median": _r(med_n),
        "n_wls_iqr": _r(float(np.percentile([w["n_wls"] for w in win], 75) - np.percentile([w["n_wls"] for w in win], 25)) if len(win) >= 4 else None),
        "alignment_median": _r(med_al),
        "explained_median": _r(med_ex),
        "residual_median": _r(med_res),
        "sign": float(np.sign(med_n)) if med_n else 0.0,
        "sign_consistency": _r(sign_consistency, 3),
        "frac_low_q": _r(float(np.mean(qn < q_min)) if n else None),
        "frac_sat": _r(float(np.mean(tr.sat)) if n else None),
        "frac_no_lock": _r(float(np.mean(~tr.lock)) if n else None),
        "frac_bad_direction": _r(float(np.nanmean(alignment < 0.5)) if np.isfinite(alignment).any() else None),
        "hit": tr.hit,
        **tr.meta,
    }


def _classify_law(meds: dict[str, float | None], iqr_n: float | None) -> str:
    """Правило интерпретации (P6): большой ортогональный остаток запрещает
    читать N_eff как коэффициент ПН; большой IQR сам по себе ничего не значит."""
    residual, alignment, n_med = meds["residual_median"], meds["alignment_median"], meds["n_wls_median"]
    if residual is None or alignment is None or n_med is None:
        return "insufficient_data"
    if residual <= 0.3 and alignment >= 0.85:
        spread = iqr_n / max(abs(n_med), 1e-9) if iqr_n is not None else 1e9
        return "pn_like_constant_N" if spread <= 0.15 else "pn_like_variable_N"
    return "not_representable_as_scalar_pn"


def neff_decomposition(
    circuit,
    scenarios: list[Scenario],
    *,
    window: int = 25,
    q_min: float = 0.5,
    n_boot: int = 200,
    seed: int = 0,
    t_cap: float = 16.0,
) -> dict[str, Any]:
    """P6 по множеству эпизодов: медианы, IQR, bootstrap CI ПО ЭПИЗОДАМ и
    классификация по правилу интерпретации:
    - residual мал И alignment высок → «ПН-подобный» (N постоянный/переменный
      по разбросу N_wls);
    - residual велик → N_eff НЕ интерпретируется как коэффициент ПН;
    - большой IQR сам по себе ничего не доказывает."""
    per_ep: list[dict[str, Any]] = []
    for sc in scenarios:
        tr = episode_trace(circuit, sc, t_cap=t_cap)
        per_ep.append(decompose_episode(tr, window=window, q_min=q_min))

    def _col(name: str) -> list[float]:
        return [e[name] for e in per_ep if e.get(name) is not None]

    def _boot(name: str) -> tuple[float | None, float | None]:
        xs = _col(name)
        if len(xs) < 3:
            return None, None
        rng = np.random.default_rng(seed)
        boots = [float(np.median(rng.choice(xs, size=len(xs), replace=True))) for _ in range(n_boot)]
        return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    keys = (
        "n_wls_median", "alignment_median", "explained_median", "residual_median",
        "frac_low_q", "frac_sat", "frac_no_lock", "frac_bad_direction",
    )
    meds = {k: (float(np.median(_col(k))) if _col(k) else None) for k in keys}
    ci = {k: _boot(k) for k in ("n_wls_median", "residual_median", "alignment_median")}
    ns = _col("n_wls_median")
    iqr_n = float(np.percentile(ns, 75) - np.percentile(ns, 25)) if len(ns) >= 4 else None
    verdict = _classify_law(meds, iqr_n)
    return {
        "note_units": "rho — 1/с; k_rho в N=N0+k_rho·rho — секунды; q — м/с² на единицу N",
        "window_steps": window,
        "q_min_mps2": q_min,
        "n_episodes": len(per_ep),
        "per_episode": per_ep,
        "medians": {k: _r(v) for k, v in meds.items()},
        "iqr_n_wls": _r(iqr_n),
        "bootstrap_ci95": {k: {"low": _r(v[0]), "high": _r(v[1])} for k, v in ci.items() if v[0] is not None},
        "verdict": verdict,
        "interpretation": {
            "pn_like_constant_N": "закон воспроизводится скалярной ПН с почти постоянным N",
            "pn_like_variable_N": "ПН-подобный закон с переменным N (residual мал, alignment высок)",
            "not_representable_as_scalar_pn": (
                "ортогональный остаток велик: команда НЕ сводится к N·q — интерпретация "
                "N_eff как коэффициента ПН запрещена; это другой закон управления"
            ),
            "insufficient_data": "мало валидных кадров/эпизодов",
        }[verdict],
    }


# ── P7: конкурирующие гипотезы закона ─────────────────────────────────────────

HYPOTHESES = ("H1_const_pn", "H2_sched_pn", "H3_nonlinear_sched", "H4_hybrid", "H5_narx")
HYPOTHESIS_LABELS = {
    "H1_const_pn": "a = N·q (постоянная ПН)",
    "H2_sched_pn": "a = (N0 + k_rho·rho)·q (scheduled PN)",
    "H3_nonlinear_sched": "a = N(признаки)·q (нелинейная scheduled)",
    "H4_hybrid": "a = N(x)·q + K·bearing + bias (гибрид ПН+погони)",
    "H5_narx": "a_t = f(x_t, x_{t-1}, a_{t-1}) (динамический с памятью)",
}


def _mono_design(x: np.ndarray, degree: int) -> np.ndarray:
    """Полные мономы признаков x (N,10) до степени degree (с кросс-членами)."""
    from itertools import combinations_with_replacement

    cols = [np.ones(len(x))]
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(x.shape[1]), d):
            col = np.ones(len(x))
            for j in combo:
                col = col * x[:, j]
            cols.append(col)
    return np.stack(cols, axis=1)


def _design_scalar(hyp: str, q: np.ndarray, rho: np.ndarray, x: np.ndarray, b_hat: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Дизайн скалярно-векторных моделей: ОДИН общий коэффициент на все оси —
    скалярный N(x) умножает весь q сразу (иначе модель «читерит» раздельными
    N по осям). Разметка строк: row = 3*i + axis; столбец терма j:
    col[ax::3] = f_j(·) * base[:, ax]. Возвращает (A, y): y — команда (3N,)."""
    n = q.shape[0]

    def term_col(func: np.ndarray, base: np.ndarray) -> np.ndarray:
        col = np.zeros(3 * n)
        for ax in range(3):
            col[ax::3] = func * base[:, ax]
        return col

    cols: list[np.ndarray] = []
    if hyp == "H1_const_pn":
        cols.append(term_col(np.ones(n), q))
    elif hyp == "H2_sched_pn":
        cols.append(term_col(np.ones(n), q))
        cols.append(term_col(rho, q))
    elif hyp == "H3_nonlinear_sched":
        M = _mono_design(x, 2)
        for i in range(M.shape[1]):
            cols.append(term_col(M[:, i], q))
    elif hyp == "H4_hybrid":
        M = _mono_design(x, 1)
        for i in range(M.shape[1]):  # N(x)·q — линейный по признакам
            cols.append(term_col(M[:, i], q))
        cols.append(term_col(np.ones(n), b_hat))  # один общий K при пеленге
        for ax in range(3):  # bias на каждую ось
            col = np.zeros(3 * n)
            col[ax::3] = 1.0
            cols.append(col)
    else:
        raise ValueError(hyp)
    y = np.zeros(3 * n)
    for ax in range(3):
        y[ax::3] = a[:, ax]
    return np.stack(cols, axis=1), y


def _design_narx(
    x: np.ndarray,
    a: np.ndarray,
    episode_starts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """H5: a_t = c + A1·x_t + A2·x_{t-1} + B·a_{t-1} — малый линейный NARX
    (лаг 1), отдельные коэффициенты на ось. Возвращает (X, y).

    episode_starts отмечает первые кадры независимых полётов. На этих строках
    лаги обязаны быть нулевыми: последний кадр предыдущего эпизода физически не
    является состоянием перед первым кадром следующего."""
    n = x.shape[0]
    if a.shape[0] != n:
        raise ValueError("x и a должны содержать одинаковое число кадров")
    starts = np.zeros(n, dtype=bool) if episode_starts is None else np.asarray(episode_starts, dtype=bool).copy()
    if starts.shape != (n,):
        raise ValueError("episode_starts должен иметь форму (N,)")
    if n:
        starts[0] = True
    X = np.hstack([np.ones((n, 1)), x, np.roll(x, 1, axis=0), np.roll(a, 1, axis=0)])
    lag0 = 1 + x.shape[1]
    # Текущие признаки x_t на старте эпизода валидны. Неизвестны только лаги.
    X[starts, lag0:] = 0.0
    return X, a


def fit_hypotheses(traces: list[EpisodeTrace], *, val_frac: float = 0.3, seed: int = 7, n_boot: int = 60) -> dict[str, Any]:
    """P7: сравнение H1–H5 на полночастотных трассах, разбиение ПО ЭПИЗОДАМ
    целиком (кадры одного полёта не смешиваются между train и validation)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(traces))
    n_val = max(1, int(round(len(traces) * val_frac)))
    val_idx = sorted(order[:n_val].tolist())
    train_idx = [i for i in range(len(traces)) if i not in set(val_idx)]

    def _bundle(idxs: list[int]) -> dict[str, Any]:
        qs, rhos, xs, bs, aa, starts = [], [], [], [], [], []
        for i in idxs:
            tr = traces[i]
            qs.append(tr.q_basis())
            rhos.append(tr.rho_meas)
            xs.append(tr.feat)
            bs.append(tr.r / (np.linalg.norm(tr.r, axis=1, keepdims=True) + 1e-9))
            aa.append(tr.a_cmd)
            boundary = np.zeros(len(tr.t), dtype=bool)
            if len(boundary):
                boundary[0] = True
            starts.append(boundary)
        return {
            "q": np.vstack(qs),
            "rho": np.concatenate(rhos),
            "x": np.vstack(xs),
            "b": np.vstack(bs),
            "a": np.vstack(aa),
            "episode_starts": np.concatenate(starts),
        }

    tr_b, va_b = _bundle(train_idx), _bundle(val_idx)
    coefs: dict[str, np.ndarray] = {}
    results: dict[str, Any] = {}

    def _fit(hyp: str, b: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if hyp == "H5_narx":
            A, y = _design_narx(b["x"], b["a"], b["episode_starts"])
        else:
            A, y = _design_scalar(hyp, b["q"], b["rho"], b["x"], b["b"], b["a"])
        # лёгкий ridge для богатых дизайнов (H3/H4): flight-домен узкий, МНК
        # без регуляризации даёт абсурдные экстраполяции на validation
        lam = 1e-4 * float(np.mean(np.diag(A.T @ A))) if A.size else 0.0
        coef = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
        return A, y, coef

    for hyp in HYPOTHESES:
        A_va, y_va, coef = _fit(hyp, va_b)
        if hyp == "H5_narx":
            _A_tr, _y_tr, coef = _fit(hyp, tr_b)  # NARX фитится по train-строкам
            pred = A_va @ coef
            # coef уже содержит отдельный столбец для каждой из трёх осей.
            n_params = int(coef.size)
        else:
            _A_tr, _y_tr, coef = _fit(hyp, tr_b)
            A_va, y_va, _ = _fit(hyp, va_b)
            pred = A_va @ coef
            n_params = int(coef.size)
        coefs[hyp] = coef
        err = (pred - y_va).reshape(-1)
        ss_tot = float(np.sum((y_va.reshape(-1) - np.mean(y_va.reshape(-1))) ** 2)) or 1.0
        r2 = 1.0 - float(np.sum(err**2)) / ss_tot
        P = pred.reshape(-1, 3)
        T = np.asarray(y_va).reshape(-1, 3)
        pn, tn = np.linalg.norm(P, axis=1), np.linalg.norm(T, axis=1)
        ok = (pn > 1e-6) & (tn > 1e-6)
        cos = np.clip(np.einsum("ij,ij->i", P[ok], T[ok]) / (pn[ok] * tn[ok]), -1.0, 1.0)
        dir_err = float(np.degrees(np.arccos(cos)).mean()) if ok.any() else None

        def _r2_mask(mask_rows: np.ndarray) -> float | None:
            if not mask_rows.any():
                return None
            e = err[mask_rows]
            t = y_va.reshape(-1)[mask_rows]
            st = float(np.sum((t - np.mean(t)) ** 2)) or 1.0
            return float(1.0 - np.sum(e**2) / st)

        def _stretch(mask_frames: np.ndarray) -> np.ndarray:
            out = np.zeros(3 * len(mask_frames), dtype=bool)
            for ax in range(3):
                out[ax::3] = mask_frames
            return out

        xva = va_b["x"]
        central = (np.abs(xva[:, 0]) <= 0.6) & (np.abs(xva[:, 1]) <= 0.6)
        rho_med = float(np.median(va_b["rho"]))
        low_rho = va_b["rho"] < rho_med
        results[hyp] = {
            "label": HYPOTHESIS_LABELS[hyp],
            "n_params": n_params,
            "r2_val": round(float(r2), 4),
            "mae_val": round(float(np.mean(np.abs(err))), 4),
            "direction_err_deg": round(dir_err, 2) if dir_err is not None else None,
            "unexplained": round(float(1.0 - r2), 4),
            "r2_central": _r(_r2_mask(_stretch(central))),
            "r2_peripheral": _r(_r2_mask(_stretch(~central))),
            "r2_low_rho": _r(_r2_mask(_stretch(low_rho))),
            "r2_high_rho": _r(_r2_mask(_stretch(~low_rho))),
            "model_kind": "deployable_sensor_model" if hyp in ("H1_const_pn", "H2_sched_pn", "H5_narx") else "diagnostic_geometry_model",
            "deployable_form_exists": hyp in ("H1_const_pn", "H2_sched_pn", "H5_narx"),
        }
        results[hyp]["coef_summary"] = _coef_summary(hyp, coef)

    # bootstrap CI по R²: ресэмплинг validation-ЭПИЗОДОВ
    for hyp in HYPOTHESES:
        boots: list[float] = []
        for _ in range(n_boot):
            pick = rng.choice(val_idx, size=len(val_idx), replace=True)
            sub = _bundle([int(i) for i in pick])
            if hyp == "H5_narx":
                A_b, y_b = _design_narx(sub["x"], sub["a"], sub["episode_starts"])
            else:
                A_b, y_b = _design_scalar(hyp, sub["q"], sub["rho"], sub["x"], sub["b"], sub["a"])
            e = ((A_b @ coefs[hyp]) - y_b).reshape(-1)
            t_b = np.asarray(y_b).reshape(-1)
            st = float(np.sum((t_b - np.mean(t_b)) ** 2)) or 1.0
            boots.append(1.0 - float(np.sum(e**2)) / st)
        results[hyp]["r2_ci95"] = [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]

    return {
        "n_episodes": len(traces),
        "n_episodes_train": len(train_idx),
        "n_episodes_val": len(val_idx),
        "split": "по эпизодам целиком (кадры одного полёта не смешиваются)",
        "hypotheses": results,
        "winner_by_unexplained": min(results, key=lambda h: results[h]["unexplained"]),
        # сырые коэффициенты H5 (для установки в контур, /api/science/closed-loop);
        # не сериализуются в JSON-отчёты напрямую
        "coef_narx": coefs.get("H5_narx"),
        "caveat": (
            "аппроксимация команды ≠ закон: гипотеза признаётся содержательной только "
            "после closed-loop воспроизведения результата (P8)"
        ),
    }


def _coef_summary(hyp: str, coef: np.ndarray) -> dict[str, Any]:
    if hyp == "H1_const_pn":
        return {"N": round(float(coef[0]), 3)}
    if hyp == "H2_sched_pn":
        return {"N0": round(float(coef[0]), 3), "k_rho_s": round(float(coef[1]), 4), "units": "k_rho — секунды (rho — 1/с)"}
    if hyp == "H5_narx":
        return {"форма": "a_t = c + A1·x_t + A2·x_{t-1} + B·a_{t-1} (на ось)"}
    return {"n_terms": int(np.sum(np.abs(np.asarray(coef).reshape(-1)) > 1e-6))}


# ── P8: closed-loop проверка deployable-законов ───────────────────────────────


class DeployableLaw:
    """Сенсорный закон: command(obs, v_m, n_max) — доступ ТОЛЬКО к отложенному
    кадру ГСН. Истинная геометрия недоступна по сигнатуре (проверяется тестом)."""

    name = "law"
    label = ""
    model_kind = "deployable_sensor_model"

    def reset(self) -> None:
        raise NotImplementedError

    def command(self, obs: dict, v_m: np.ndarray, n_max: float) -> np.ndarray:
        raise NotImplementedError


class SensorConstantPN(DeployableLaw):
    def __init__(self, n_const: float) -> None:
        self.N = float(n_const)
        self.name = "cl_const_pn"
        self.label = f"сенсорная ПН, N={self.N:g}"

    def reset(self) -> None:
        pass

    def command(self, obs: dict, v_m: np.ndarray, n_max: float) -> np.ndarray:
        return pn_seeker_accel(obs["az_dot_s"], obs["el_dot_s"], v_m, self.N, n_max)


class SensorScheduledPN(DeployableLaw):
    def __init__(self, n0: float, k_rho: float, n_min: float = 2.0, n_max_n: float = 6.0) -> None:
        self.n0, self.k, self.nmin, self.nmax = n0, k_rho, n_min, n_max_n
        self.name = "cl_sched_pn"
        self.label = f"scheduled PN (сенсорная): N0={n0:g}, k_rho={k_rho:g} с"

    def reset(self) -> None:
        pass

    def command(self, obs: dict, v_m: np.ndarray, n_max: float) -> np.ndarray:
        return scheduled_pn_seeker_accel(obs["az_dot_s"], obs["el_dot_s"], obs["rho"], v_m, self.n0, self.k, self.nmin, self.nmax, n_max)


class PolyReadout(DeployableLaw):
    """Полиномиальный readout dn(feat) — суррогат, способный лететь: обе оси
    считаются ТОЛЬКО из признаков кадра (без истинной геометрии)."""

    def __init__(self, coef_pitch: np.ndarray, coef_yaw: np.ndarray, degree: int) -> None:
        self.coef = np.asarray([coef_pitch, coef_yaw], dtype=np.float64)
        self.degree = int(degree)
        self.name = "cl_poly_readout"
        self.label = f"полином-суррогат readout (степень {self.degree}) в контуре"

    def reset(self) -> None:
        pass

    def command(self, obs: dict, v_m: np.ndarray, n_max: float) -> np.ndarray:
        t4 = np.zeros(4)
        loom = max(0.0, float(obs.get("size_dot") or 0.0) * 40.0) if obs.get("lock") else 0.0
        feat = _features(obs, t4, loom)
        A, _terms = _poly_design(feat[None, :], self.degree)
        dn = A @ self.coef.T
        _x, y, z = body_axes(v_m)
        a = z * float(dn[0, 0]) * n_max * G + y * float(dn[0, 1]) * n_max * G
        return clip_accel(a, n_max)


class NarxLaw(DeployableLaw):
    """H5 в замкнутом контуре: a_t = c + A1·x_t + A2·x_{t-1} + B·a_{t-1}.
    Память только на собственных прошлых признаках/командах — deployable."""

    def __init__(self, coef: np.ndarray) -> None:
        # coef: (1+2*FEAT_DIM+3, 3) — из _design_narx
        self.coef = np.asarray(coef, dtype=np.float64)
        self.name = "cl_narx"
        self.label = "NARX (лаг 1) в контуре"
        self.reset()

    def reset(self) -> None:
        self.prev_feat = np.zeros(FEAT_DIM)
        self.prev_a = np.zeros(3)

    def command(self, obs: dict, v_m: np.ndarray, n_max: float) -> np.ndarray:
        t4 = np.zeros(4)
        loom = max(0.0, float(obs.get("size_dot") or 0.0) * 40.0) if obs.get("lock") else 0.0
        feat = _features(obs, t4, loom)
        row = np.concatenate([[1.0], feat, self.prev_feat, self.prev_a])
        a = row @ self.coef
        self.prev_feat = feat
        self.prev_a = np.asarray(a, dtype=np.float64)
        return np.asarray(a, dtype=np.float64)


def closed_loop_check(
    bio_circuit,
    laws: list[DeployableLaw],
    test_scenarios: list[Scenario],
    *,
    t_cap: float = 16.0,
) -> dict[str, Any]:
    """P8: каждый deployable-закон ставится в контур вместо BIO и летит ОДНИ И ТЕ
    ЖЕ held-out сценарии. Закон содержателен, только если он (1) объясняет
    команды (P7) И (2) сохраняет результат в замкнутом контуре. Сравнение — по
    физическим метрикам (никаких аппроксимационных R²)."""
    from navedenie.law import evaluate_batch

    bio = evaluate_batch(bio_circuit, test_scenarios, teacher=False, t_cap=t_cap, scenario_split="test")
    rows: list[dict[str, Any]] = [
        {
            "name": "bio",
            "label": "BIO (обученный мозг)",
            "model_kind": "reference",
            "deployable": True,
            "closed_loop_flown": True,
            **bio.components(),
        }
    ]
    for law in laws:
        law.reset()

        def ctrl(obs: dict, v_m: np.ndarray, n_max: float, _law=law) -> np.ndarray:
            return _law.command(obs, v_m, n_max)

        lm = evaluate_batch(None, test_scenarios, teacher=False, t_cap=t_cap, scenario_split="test", controller=ctrl)
        rows.append(
            {
                "name": law.name,
                "label": law.label,
                "model_kind": law.model_kind,
                "deployable": True,
                "closed_loop_flown": True,
                **lm.components(),
                "degradation_vs_bio": {
                    "hit_rate": round(lm.hit_rate - bio.hit_rate, 4),
                    "cpa_median_m": round(lm.cpa_median - bio.cpa_median, 2),
                    "effort_median_gs": round(lm.effort_median - bio.effort_median, 3),
                },
            }
        )
    return {
        "n_test_episodes": len(test_scenarios),
        "rows": rows,
        "rule": (
            "содержательный закон: объясняет команды (P7) И держит hit rate/CPA в контуре; "
            "высокий R² без closed-loop = локальный readout-суррогат, не закон наведения"
        ),
    }


# ── P9: фотометрическая устойчивость и matched-абляции ────────────────────────


def replace_brightness(sc: Scenario, factor: float) -> Scenario:
    from dataclasses import replace

    return replace(sc, target_brightness=float(factor))


def photometric_robustness(
    circuit,
    scenarios: list[Scenario],
    *,
    factors: tuple[float, ...] = (0.7, 1.0, 1.4),
    t_cap: float = 16.0,
) -> dict[str, Any]:
    """Яркость пятна (альбедо/подсветка) масштабируется, геометрия та же:
    theta декодируется из суммарной яркости в известной модели оптики — если
    политика использовала фиксированную яркость как скрытый датчик дальности,
    метрики при brightness ≠ 1 просядут."""
    from navedenie.law import evaluate_batch

    out: list[dict[str, Any]] = []
    base: dict[str, Any] | None = None
    for f in factors:
        bm = evaluate_batch(
            circuit,
            [replace_brightness(sc, f) for sc in scenarios],
            teacher=False,
            t_cap=t_cap,
            scenario_split="test",
        )
        comp = bm.components()
        comp["brightness"] = f
        if f == 1.0:
            base = comp
        out.append(comp)
    deltas = None
    if base is not None:
        deltas = [
            {
                "brightness": c["brightness"],
                "d_hit_rate": round(c["hit_rate"] - base["hit_rate"], 4),
                "d_cpa_median_m": round(c["cpa_median_m"] - base["cpa_median_m"], 2),
            }
            for c in out
            if c["brightness"] != 1.0
        ]
    robust = deltas is not None and all(
        abs(d["d_hit_rate"]) <= 0.15 and abs(d["d_cpa_median_m"]) <= 60.0 for d in deltas
    )
    return {
        "factors": list(factors),
        "rows": out,
        "delta_vs_nominal": deltas,
        "verdict": ("устойчива: скрытого дальномера по яркости нет" if robust else "НЕ устойчива: политика читает яркость как дальность"),
    }


def matched_ablation_run(circuit, sc: Scenario, mode: str, *, t_cap: float = 16.0) -> dict[str, Any]:
    """Контрфакт на СОВПАДАЮЩИХ условиях: одна геометрия, один seed шума, одна
    маска отказов — меняется только подмена признака (feat_patch)."""
    from navedenie.law import evaluate_episode
    from navedenie.science import _apply_rho_ablation

    saved = circuit.feat_patch
    circuit.feat_patch = None
    base = evaluate_episode(circuit, sc, teacher=False, t_cap=t_cap, scenario_split="test")
    _apply_rho_ablation(circuit, mode, [], 0, sc.seed, int(min(sc.t_max, t_cap) / max(sc.dt, 1e-6)) + 2)
    abl = evaluate_episode(circuit, sc, teacher=False, t_cap=t_cap, scenario_split="test")
    circuit.feat_patch = saved
    return {
        "mode": mode,
        "baseline": base.to_dict(),
        "ablated": abl.to_dict(),
        "delta": {
            "hit": int(abl.hit) - int(base.hit),
            "geometric_cpa_m": round(abl.geometric_cpa_m - base.geometric_cpa_m, 2),
            "effort_gs": round(abl.effort_gs - base.effort_gs, 3),
        },
    }
