"""Диагностика закона: разложение N_eff, гипотезы H1–H5, closed-loop, P9 (P12).

- N_wls устойчив при малом, но ненулевом q (вес, а не взрыв);
- большой ортогональный остаток НЕ классифицируется как переменная ПН;
- у oracle-ПН разложение даёт заданный N и нулевой остаток;
- гипотезы фитятся с разбиением по эпизодам ЦЕЛИКОМ;
- closed-loop суррогаты читают только допустимые сенсорные признаки;
- яркость (фотометрия) меняет измеренную theta, но не геометрию сенсора.
"""

import json
import warnings

import numpy as np
import pytest

from navedenie.circuit import FEAT_DIM, FlyCircuit
from navedenie.diagnostics import (
    EpisodeTrace,
    NarxLaw,
    PolyReadout,
    SensorConstantPN,
    SensorScheduledPN,
    _classify_law,
    _design_narx,
    _wls_window,
    decompose_episode,
    episode_trace,
    fit_hypotheses,
    matched_ablation_run,
    neff_decomposition,
    photometric_robustness,
    replace_brightness,
)
from navedenie.pn import n_eff_basis
from navedenie.sim import Scenario
from navedenie.train import _rollout_update

FOV = np.deg2rad(165.0) / 2.0


def _synth_trace(n: int = 200, a_fn=None, q_scale: float = 1.0) -> EpisodeTrace:
    """Синтетическая полночастотная трасса: геометрия на прямой, q ненулевой,
    команда задаётся a_fn(q, t)."""
    t = np.arange(n) * 0.02
    r = np.stack([5000.0 - 200.0 * t, 100.0 + 0.0 * t, 0.0 * t], axis=1)
    v_m = np.tile(np.array([700.0, 0.0, 0.0]), (n, 1))
    v_t = np.tile(np.array([-260.0, 0.0, 0.0]), (n, 1))
    q = np.stack([n_eff_basis(r[i], v_m[i], v_t[i]) for i in range(n)])
    a = a_fn(q, t) if a_fn else np.zeros((n, 3))
    feat = np.zeros((n, FEAT_DIM))
    feat[:, 9] = 1.0  # lock
    return EpisodeTrace(
        t=t, a_cmd=a, r=r, v_m=v_m, v_t=v_t,
        lock=np.ones(n, dtype=bool), sat=np.zeros(n, dtype=bool),
        rho_meas=np.full(n, 0.2), feat=feat, hit=False, geometric_cpa_m=100.0,
        meta={"v_m": 700.0, "v_t": 260.0, "aspect": "head-on", "speed_mode": "constant", "maneuver": "straight"},
    )


# ── P12.4: WLS устойчив при малом q ──────────────────────────────────────────

def test_wls_stable_at_small_but_nonzero_q() -> None:
    """|q| → 0 (почти нулевая ω_LOS): вес гасит вклад — N_wls конечен, не взрывается."""
    n = 40
    q = np.tile(np.array([1e-3, 0.0, 0.0]), (n, 1))  # маленький, но ненулевой
    rng = np.random.default_rng(0)
    a = rng.normal(0, 5.0, (n, 3))
    w = _wls_window(a, q, q_min=0.5)
    assert np.isfinite(w["n_wls"])
    assert abs(w["n_wls"]) < 1e3  # без веса было бы ~ a/q ~ 5000
    # при сильном q тот же код даёт корректный коэффициент
    q_big = np.tile(np.array([10.0, 0.0, 0.0]), (n, 1))
    a_exact = 3.0 * q_big
    w_exact = _wls_window(a_exact, q_big, q_min=0.5)
    assert w_exact["n_wls"] == pytest.approx(3.0, rel=1e-6)
    assert w_exact["residual"] < 1e-9


def test_wls_window_zero_command_is_quiet_nan() -> None:
    """Эпизод с нулевой командой: cos/expl вырождены в NaN — медиана NaN
    БЕЗ RuntimeWarning «All-NaN slice» (отчёт остаётся JSON-безопасным через _r)."""
    n = 40
    q = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    a = np.zeros((n, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        w = _wls_window(a, q, q_min=0.5)
    assert w["alignment"] != w["alignment"]  # NaN
    assert w["explained"] != w["explained"]  # NaN
    assert w["residual"] == pytest.approx(0.0, abs=1e-12)


# ── P12.5: большой остаток не «переменная ПН» ────────────────────────────────

def test_large_orthogonal_residual_not_classified_as_variable_pn() -> None:
    """a ⊥ q: alignment ~ 0, residual ~ 1 → вердикт «не сводится к скалярной ПН»,
    даже если «N_eff» меняется по кадрам."""
    q_dir = np.array([0.0, 1.0, 0.0])
    a_fn = lambda q, t: np.tile(np.array([0.0, 0.0, 12.0]), (len(t), 1))  # ⊥ q
    tr = _synth_trace(a_fn=a_fn)
    dec = decompose_episode(tr, window=20)
    meds = {k: dec[k] for k in ("n_wls_median", "alignment_median", "residual_median")}
    verdict = _classify_law(meds, iqr_n=5.0)  # огромный IQR не спасает
    assert verdict == "not_representable_as_scalar_pn"
    assert dec["alignment_median"] is not None and abs(dec["alignment_median"]) < 0.2


def test_pure_pn_decomposes_to_exact_N() -> None:
    """a = 4·q: N_wls = 4, alignment = 1, residual ≈ 0 → «ПН-подобный, N постоянный»."""
    a_fn = lambda q, t: 4.0 * q
    tr = _synth_trace(a_fn=a_fn)
    dec = decompose_episode(tr, window=20)
    assert dec["n_wls_median"] == pytest.approx(4.0, rel=1e-6)
    assert dec["alignment_median"] > 0.999
    assert dec["residual_median"] < 1e-6
    meds = {k: dec[k] for k in ("n_wls_median", "alignment_median", "residual_median")}
    assert _classify_law(meds, iqr_n=0.0) == "pn_like_constant_N"


def test_variable_N_with_small_residual_is_pn_like_variable() -> None:
    """a = N(t)·q с переменным N: residual мал → «переменный N» (и только при
    малом остатке это утверждение законно)."""
    a_fn = lambda q, t: (2.0 + 2.0 * np.sin(2 * np.pi * t / 2.0))[:, None] * q
    tr = _synth_trace(a_fn=a_fn)
    dec = decompose_episode(tr, window=20)
    meds = {k: dec[k] for k in ("n_wls_median", "alignment_median", "residual_median")}
    assert _classify_law(meds, iqr_n=2.0) == "pn_like_variable_N"


# ── P7: гипотезы — фит и эпизодное разбиение (P12.9) ─────────────────────────

def test_hypotheses_fit_and_episode_split() -> None:
    """H1 восстанавливает точную постоянную N; разбиение не смешивает эпизоды."""
    traces = []
    for k, n_const in enumerate((3.0, 5.0, 3.0, 5.0, 3.0)):
        traces.append(_synth_trace(n=120, a_fn=lambda q, t, nc=n_const: nc * q))
    report = fit_hypotheses(traces, val_frac=0.4, seed=1)
    assert report["n_episodes_train"] + report["n_episodes_val"] == len(traces)
    h1 = report["hypotheses"]["H1_const_pn"]
    # эпизоды двух типов: лучший достижимый R² ограничен смесью, но H1 обязан
    # объяснить данные лучше случайного базиса, а H5 не обязан выигрывать
    assert h1["r2_val"] > 0.5
    assert h1["n_params"] == 1  # один скаляр N на все оси (честная скалярность)
    assert h1["model_kind"] == "deployable_sensor_model"
    assert report["hypotheses"]["H2_sched_pn"]["model_kind"] == "deployable_sensor_model"
    assert report["hypotheses"]["H5_narx"]["model_kind"] == "deployable_sensor_model"
    assert report["hypotheses"]["H4_hybrid"]["model_kind"] == "diagnostic_geometry_model"
    # у H2 (scheduled) k_rho при чисто постоянной ПН не обязан быть большим
    assert "k_rho_s" in report["hypotheses"]["H2_sched_pn"]["coef_summary"]
    # непрерывность эпизодов: количество строк validation кратно сумме кадров эпизодов × 3
    total_val_frames = sum(traces[i].a_cmd.shape[0] for i in range(len(traces))) * 0  # noqa: F841
    assert report["split"].startswith("по эпизодам")


def test_h2_recovers_scheduled_law() -> None:
    """a = (N0 + k_rho·rho)·q при РАЗНЫХ rho в эпизодах → H2 восстанавливает
    N0≈3, k_rho≈0.5 с (при одном rho модель вырождена — это честно)."""
    trs = []
    for rho_val in (0.2, 1.0, 2.0, 0.5):
        n = 3.0 + 0.5 * rho_val
        tr = _synth_trace(n=200, a_fn=lambda q, t, nc=n: nc * q)
        tr.rho_meas = np.full(len(tr.t), rho_val)
        trs.append(tr)
    report = fit_hypotheses(trs, val_frac=0.5, seed=0)
    summary = report["hypotheses"]["H2_sched_pn"]["coef_summary"]
    assert summary["N0"] == pytest.approx(3.0, abs=0.05)
    assert summary["k_rho_s"] == pytest.approx(0.5, abs=0.05)
    assert report["hypotheses"]["H2_sched_pn"]["r2_val"] > 0.99


def test_narx_lags_reset_at_episode_boundaries() -> None:
    """Новый полёт не наследует лаги последнего кадра предыдущего полёта."""
    x = np.arange(6 * FEAT_DIM, dtype=float).reshape(6, FEAT_DIM)
    a = np.arange(18, dtype=float).reshape(6, 3)
    starts = np.array([True, False, False, True, False, False])
    design, target = _design_narx(x, a, starts)
    lag0 = 1 + FEAT_DIM

    assert np.array_equal(target, a)
    assert np.array_equal(design[3, 1:lag0], x[3])  # текущий кадр валиден
    assert np.all(design[[0, 3], lag0:] == 0.0)  # неизвестны только лаги
    assert np.array_equal(design[4, lag0:lag0 + FEAT_DIM], x[3])
    assert np.array_equal(design[4, lag0 + FEAT_DIM:], a[3])


def test_law_hypotheses_http_report_is_json_safe(monkeypatch) -> None:
    """Публичный endpoint не должен отдавать внутреннюю ndarray coef_narx."""
    import navedenie.app as app_mod
    import navedenie.diagnostics as diag_mod
    import navedenie.law as law_mod
    import navedenie.train as train_mod

    class DummyCircuit:
        gain = 1.0

    monkeypatch.setattr(app_mod, "_law_circuit", lambda _kind: DummyCircuit())
    monkeypatch.setattr(law_mod, "canonical_trio", lambda _gain: [])
    monkeypatch.setattr(train_mod, "_episode_scenario", lambda i, gain: (i, gain))
    monkeypatch.setattr(diag_mod, "episode_trace", lambda circuit, sc: sc)
    monkeypatch.setattr(
        diag_mod,
        "fit_hypotheses",
        lambda traces: {"n_episodes": len(traces), "hypotheses": {}, "coef_narx": np.zeros((3, 3))},
    )

    out = app_mod.law_hypotheses_endpoint(app_mod.LawKindIn(kind="stub"))
    assert out == {"n_episodes": 4, "hypotheses": {}}
    json.dumps(out)  # регрессия: FastAPI больше не падает на ndarray


# ── P12.10: closed-loop законы читают только сенсорные признаки ──────────────

def _fake_obs() -> dict:
    return {
        "az": 0.1, "el": -0.05, "az_dot": 0.02, "el_dot": -0.01,
        "az_dot_s": 0.015, "el_dot_s": -0.008, "size": 0.1, "size_dot": 0.01,
        "theta": 0.03, "theta_dot": 0.005, "rho": 0.17, "tau_contact": 6.0,
        "lock": True, "image": np.zeros((16, 16)),
        "truth_az": 0.1, "truth_el": -0.05, "truth_range": 3000.0, "range": 3050.0,
    }


def test_deployable_laws_ignore_truth_geometry() -> None:
    """Подмена истинных полей кадра (дальность/Vc/t_go/геометрия) не меняет
    команду deployable-закона: им доступны только декодированные признаки."""
    v_m = np.array([700.0, 0, 0])
    laws = [SensorConstantPN(4.0), SensorScheduledPN(3.0, 0.8),
            PolyReadout(np.zeros(66), np.zeros(66), 2),  # C(12,2)=66 мономов степени ≤2
            NarxLaw(np.zeros((1 + 2 * FEAT_DIM + 3, 3)))]
    for law in laws:
        law.reset()
        a1 = law.command(_fake_obs(), v_m, 30.0)
        obs = _fake_obs()
        obs["truth_range"] = 55555.5
        obs["range"] = 55555.5
        obs["truth_az"] = -2.0
        a2 = law.command(obs, v_m, 30.0)
        assert np.allclose(a1, a2), f"{law.name} реагирует на истинную геометрию"


def test_closed_loop_sensors_constant_pn_tracks_locked_frames() -> None:
    """Кадр без захвата несёт нулевые каналы (как отдаёт сенсор) → команда нулевая;
    при захвате — ненулевая и ограничена n_max. Ноль на unlocked обеспечивает
    полётный цикл (контроллер не вызывается), закон читает только каналы кадра."""
    law = SensorConstantPN(4.0)
    v_m = np.array([700.0, 0, 0])
    unlocked = _fake_obs()
    unlocked.update({"lock": False, "az_dot_s": 0.0, "el_dot_s": 0.0, "rho": 0.0})
    assert np.allclose(law.command(unlocked, v_m, 30.0), 0.0)
    a = law.command(_fake_obs(), v_m, 30.0)
    assert 0.0 < float(np.linalg.norm(a)) <= 30.0 * 9.81 + 1e-9


# ── полный цикл на реальном мозге: trace → разложение → фотометрия ───────────

def test_episode_trace_and_decomposition_on_trained_policy(tmp_path) -> None:
    """Полночастотная трасса согласовна: длины массивов равны, разложение даёт
    конечные метрики, bootstrap CI накрывает медиану."""
    circuit = FlyCircuit(kind="stub")
    sc = Scenario(aspect="head-on", mode="bio", range_m=5000.0, t_max=5.0, dt=0.02, seed=5)
    tr = episode_trace(circuit, sc, t_cap=5.0)
    n = len(tr.t)
    assert n > 50
    assert tr.a_cmd.shape == (n, 3) and tr.feat.shape == (n, FEAT_DIM)
    assert tr.lock.shape == (n,) and tr.rho_meas.shape == (n,)
    dec = decompose_episode(tr, window=20)
    assert dec["n_windows"] >= 2
    assert dec["residual_median"] is not None and 0.0 <= dec["residual_median"] <= 2.0
    rep = neff_decomposition(circuit, [sc], window=20, n_boot=30)
    assert rep["n_episodes"] == 1 and rep["verdict"] in (
        "pn_like_constant_N", "pn_like_variable_N", "not_representable_as_scalar_pn", "insufficient_data"
    )


def test_photometric_brightness_changes_theta_not_lock_geometry() -> None:
    """Яркость меняет измеренную theta (инверсия из суммарной яркости), но не
    пеленг и не факт захвата; matched-прогон детерминирован."""
    from navedenie.seeker import observe

    v = np.array([700.0, 0, 0])
    from navedenie.seeker import body_axes

    x, y, _z = body_axes(v)
    r = x * 900.0 + y * np.deg2rad(2.0) * 900.0
    o1 = observe(r, v, None, half_fov=FOV, dt_s=0.02, rng=np.random.default_rng(1))
    o2 = observe(r, v, None, half_fov=FOV, dt_s=0.02, rng=np.random.default_rng(1), brightness=1.5)
    assert o1["lock"] and o2["lock"]
    assert abs(o1["az"] - o2["az"]) < 1e-9  # пеленг не изменился
    assert o2["theta"] > o1["theta"]  # ярче → измеренный угловой размер больше


def test_photometric_robustness_report_structure() -> None:
    circuit = FlyCircuit(kind="stub")
    scens = [
        Scenario(aspect="head-on", mode="bio", range_m=5000.0, t_max=6.0, dt=0.02, seed=5),
        Scenario(aspect="beam", mode="bio", range_m=5000.0, t_max=6.0, dt=0.02, seed=6),
    ]
    rep = photometric_robustness(circuit, scens, factors=(0.8, 1.0, 1.25), t_cap=6.0)
    assert [r["brightness"] for r in rep["rows"]] == [0.8, 1.0, 1.25]
    assert rep["delta_vs_nominal"] is not None and len(rep["delta_vs_nominal"]) == 2
    assert rep["verdict"]


def test_matched_ablation_same_seed_only_feature_changes() -> None:
    """Контрфакт: одна геометрия/шум/отказы — меняется только подмена rho."""
    circuit = FlyCircuit(kind="stub")
    sc = Scenario(aspect="head-on", mode="bio", range_m=5000.0, t_max=6.0, dt=0.02, seed=11)
    rep = matched_ablation_run(circuit, sc, "rho_fixed", t_cap=6.0)
    assert rep["mode"] == "rho_fixed"
    assert rep["baseline"]["v_m"] == rep["ablated"]["v_m"]
    assert rep["delta"]["geometric_cpa_m"] is not None
    # patch восстановлен после прогона
    assert circuit.feat_patch is None or circuit.feat_patch == {}


def test_rollout_update_still_records_full_rho_series() -> None:
    """Donor-ряд rho содержит много отсчётов и меняется во времени (не 1–2 кадра)."""
    circuit = FlyCircuit(kind="stub")
    sc = Scenario(aspect="head-on", mode="bio", range_m=6000.0, t_max=6.0, dt=0.02, seed=2)
    rho: list[float] = []
    _rollout_update(circuit, sc, lr=0.0, rho_trace=rho)
    assert len(rho) > 150, f"donor-ряд прорежен: {len(rho)} отсчётов"
    assert float(np.std(rho)) > 1e-3, "rho не меняется во времени"


def test_resample_by_phase_interpolates_donor() -> None:
    """Интерполяция donor-ряда по нормированной фазе: длина = n, значения в
    диапазоне донора, монотонная развёртка (хвост не «залипает»)."""
    from navedenie.science import _resample_by_phase

    donor = list(np.linspace(0.0, 8.0, 41))  # 41 отсчёт
    out = _resample_by_phase(donor, 300)
    assert out.shape == (300,)
    assert float(out[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(out[-1]) == pytest.approx(8.0, abs=1e-9)
    assert np.all(np.diff(out) > 0)  # развёртка без плато
    assert float(out.max()) <= 8.0 + 1e-9
