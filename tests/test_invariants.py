"""Математические инварианты и воспроизводимость (P2).

Каждый тест — обязательное свойство честной модели: размерности законов,
непрерывный перехват, фовеальное зрение без утечек, метрики v2, детерминизм.
"""

import numpy as np
import pytest

from navedenie.engine import collect
from navedenie.pn import apn_accel, pn_seeker_accel, ppn_accel
from navedenie.seeker import (
    ObservationBuffer,
    body_axes,
    dead_mask_for,
    decode_image,
    fovea_angles,
    observe,
    raster_foveal,
)
from navedenie.sim import G, Scenario, integrate, spawn, step_encounter, target_accel


def _sc(**kw) -> Scenario:
    base = dict(aspect="head-on", v_m=780, v_t=240, range_m=6000, off_axis_m=200,
                n_max=30, t_max=14, dt=0.02, mode="pn", pn_n=4, seeker_delay_s=0)
    base.update(kw)
    return Scenario(**base)  # type: ignore[arg-type]


# ── 1–2. APN: сведение к PN и размерности ────────────────────────────────────

def test_apn_with_zero_target_accel_equals_ppn() -> None:
    rng = np.random.default_rng(3)
    for _ in range(10):
        r = rng.uniform(-4000, 4000, 3)
        v_m = rng.uniform(300, 900, 3)
        v_t = rng.uniform(-300, 300, 3)
        a_ppn = ppn_accel(r, v_m, v_t, 4.0, 30.0)
        a_apn = apn_accel(r, v_m, v_t, np.zeros(3), 4.0, 30.0)
        assert np.allclose(a_apn, a_ppn, atol=1e-12)


def test_apn_units_and_no_gratuitous_saturation() -> None:
    r = np.array([5000.0, 100.0, 0.0])
    v_m = np.array([700.0, 0.0, 0.0])
    v_t = np.array([-240.0, 0.0, 0.0])
    a_t = np.array([0.0, 2.0 * G, 0.0])  # манёвр цели 2 g вбок
    a = apn_accel(r, v_m, v_t, a_t, 4.0, 30.0)
    # добавка (N/2)·a_t = 4g, база ~ несколько g: итог ниже предела 30g
    assert np.linalg.norm(a) < 30.0 * G
    assert np.isfinite(a).all()
    # размерность: добавка по боковой оси ~ (N/2)·2g = 4g ± базовый член
    a_base = ppn_accel(r, v_m, v_t, 4.0, 30.0)
    diff = float(np.linalg.norm(a - a_base))
    assert 0.5 * 4.0 * 2.0 * G * 0.5 < diff < 4.5 * G  # ~4 g после проецирования


# ── 3–4. Непрерывный перехват: сходимость по dt и межшаговое пересечение ─────

def test_hit_and_cpa_converge_over_dt() -> None:
    # (а) чистая геометрия: CPA на встречных траекториях сходится с машинной точностью
    p_m0, p_t0 = np.zeros(3), np.array([3000.0, 500.0, 0.0])
    v_m, v_t = np.array([780.0, 0, 0]), np.array([-240.0, 0, 0])
    geometric = []
    for dt in (0.04, 0.02, 0.01, 0.005):
        pm, pt = p_m0.copy(), p_t0.copy()
        cpa_best = 1e9
        t = 0.0
        while t < 4.0:
            pm1 = pm + v_m * dt
            pt1 = pt + v_t * dt
            enc = step_encounter(pm, pm1, pt, pt1, 45.0)
            cpa_best = min(cpa_best, enc["cpa"])
            pm, pt = pm1, pt1
            t += dt
        geometric.append(cpa_best)
    assert max(geometric) - min(geometric) < 1.0, f"геометрический CPA расходится: {geometric}"
    # (б) контур наведения: классификация устойчива, CPA в разумном допуске
    cpas = []
    hits = []
    for dt in (0.04, 0.02, 0.01, 0.005):
        res = collect(_sc(dt=dt, t_max=20), stride=100_000)
        hits.append(res.hit)
        cpas.append(res.cpa_m)
    assert all(hits), "классификация hit не должна зависеть от dt"
    assert max(cpas) - min(cpas) < 45.0, f"CPA расходится по dt: {cpas}"


def test_step_encounter_detects_mid_step_crossing() -> None:
    # сфера пересекается ВНУТРИ шага: r0=80 м, сближение 100 м/с, R=45 → вход на 35% шага
    out = step_encounter(
        np.zeros(3), np.zeros(3),
        np.array([80.0, 0, 0]), np.array([80.0 - 100.0, 0, 0]),
        45.0,
    )
    assert out["hit"]
    assert out["alpha_in"] is not None and abs(out["alpha_in"] - 0.35) < 1e-9
    assert out["cpa"] < 1e-9


# ── 5–6. Кинематика точки-массы ───────────────────────────────────────────────

def test_speed_preserved_under_normal_acceleration() -> None:
    sc = _sc(maneuver="turn", n_target=5.0)
    _m, target = spawn(sc)
    v0 = float(np.linalg.norm(target.v))
    t = 0.0
    while t < 10.0:
        integrate(target, target_accel(target, sc, t), 0.02)
        assert abs(float(np.linalg.norm(target.v)) - v0) < 1e-9
        t += 0.02


@pytest.mark.parametrize("aspect", ["head-on", "beam", "tail-chase"])
def test_initial_slant_range_exact_for_all_aspects(aspect: str) -> None:
    sc = _sc(aspect=aspect, range_m=7000)
    missile, target = spawn(sc)
    assert abs(float(np.linalg.norm(target.p - missile.p)) - 7000.0) < 1e-9


# ── 7–10. Фовеальное зрение BIO без утечек ────────────────────────────────────

def _dir_at(az: float, el: float, v_m: np.ndarray, rng_m: float) -> np.ndarray:
    x, y, z = body_axes(v_m)
    return (x * np.cos(el) * np.cos(az) + y * np.sin(az) + z * np.sin(el)) * rng_m


def _bio_obs(az_deg: float, v_m: np.ndarray | None = None):
    v = v_m if v_m is not None else np.array([700.0, 0.0, 0.0])
    r = _dir_at(np.deg2rad(az_deg), 0.0, v, 3000.0)
    return observe(r, v, None, half_fov=np.deg2rad(165.0) / 2.0, dt_s=0.02)


def test_bio_does_not_react_to_truth_without_image_change() -> None:
    """Контур читает az/el/lock ИЗ obs; замена истинных углов при том же кадре
    не меняет команду (утечки truth в управляющий путь нет)."""
    from navedenie.circuit import FlyCircuit

    img = raster_foveal(np.deg2rad(3), 0.0, 0.1, np.deg2rad(165.0) / 2.0)
    outs = []
    for truth in (0.0, 0.5, -1.2):
        c = FlyCircuit(kind="stub")
        obs = {
            "az": np.deg2rad(3), "el": 0.0, "size": 0.1, "az_dot": 0.1, "el_dot": 0.0,
            "size_dot": 0.0, "lock": True, "image": img,
            "truth_az": truth, "truth_el": 0.0,
        }
        c.step(obs, 0.02)
        outs.append((c.state.dn.copy(), c.state.feat.copy()))
    for dn, feat in outs[1:]:
        assert np.allclose(dn, outs[0][0]) and np.allclose(feat, outs[0][1])


def test_wide_periphery_detects_40_and_80_degrees() -> None:
    for deg in (40.0, 80.0):
        obs = _bio_obs(deg)
        assert obs["lock"], f"цель на {deg}° не обнаружена периферией"
        assert obs["az"] != 0.0
        # грубая локализация: ошибка декодирования много меньше самого угла
        assert abs(obs["az"] - np.deg2rad(deg)) < np.deg2rad(deg) * 0.5


def test_center_localization_is_tighter_than_periphery() -> None:
    obs2 = _bio_obs(2.0)
    obs40 = _bio_obs(40.0)
    err2 = abs(obs2["az"] - np.deg2rad(2.0))
    err40 = abs(obs40["az"] - np.deg2rad(40.0))
    assert err2 < np.deg2rad(1.0), f"центральная локализация грубее ожидаемого: {np.rad2deg(err2):.2f}°"
    assert err2 < err40


def test_zero_retina_produces_no_command() -> None:
    from navedenie.circuit import FlyCircuit

    c = FlyCircuit(kind="stub")
    c.reset()
    empty = {
        "az": 0.0, "el": 0.0, "size": 0.0, "az_dot": 0.0, "el_dot": 0.0,
        "size_dot": 0.0, "lock": False, "image": np.zeros((16, 16)),
    }
    c.step(empty, 0.02)
    c.step(empty, 0.02)
    assert np.allclose(c.state.dn, 0.0, atol=1e-9)
    v = np.array([700.0, 0.0, 0.0])
    assert np.allclose(c.accel_cmd(v, 30.0), 0.0, atol=1e-9)


def test_fovea_grid_is_dense_center_sparse_edge() -> None:
    ax = fovea_angles(np.deg2rad(82.5))
    center_step = abs(ax[8] - ax[7])
    edge_step = abs(ax[-1] - ax[-2])
    assert edge_step > 8.0 * center_step  # периферия минимум в 8 раз реже


# ── 11–12. Отказы и задержки ─────────────────────────────────────────────────

def test_persistent_dead_mask_is_seed_stable_and_single() -> None:
    m1 = dead_mask_for(0.25, np.random.default_rng(11))
    m2 = dead_mask_for(0.25, np.random.default_rng(11))
    m3 = dead_mask_for(0.25, np.random.default_rng(12))
    assert np.array_equal(m1, m2) and not np.array_equal(m1, m3)
    frac = float(m1.mean())
    assert 0.2 < frac < 0.3


def test_dead_mask_applied_once_and_dropout_per_frame() -> None:
    v = np.array([700.0, 0.0, 0.0])
    r = _dir_at(np.deg2rad(2), 0.0, v, 3000.0)
    mask = dead_mask_for(0.5, np.random.default_rng(5))
    a = observe(r, v, None, half_fov=np.deg2rad(82.5), dead_mask=mask)
    b = observe(r, v, None, half_fov=np.deg2rad(82.5), dead_mask=mask)
    assert np.array_equal(a["image"], b["image"])  # маска постоянна, второй раз не бросается
    c = observe(r, v, None, half_fov=np.deg2rad(82.5), dropout_p=0.5, rng=np.random.default_rng(3))
    d = observe(r, v, None, half_fov=np.deg2rad(82.5), dropout_p=0.5, rng=np.random.default_rng(4))
    assert not np.array_equal(c["image"], d["image"])  # дропаут — на кадр


def test_jitter_and_delay_reach_multiple_frames_back() -> None:
    rng = np.random.default_rng(1)
    buf = ObservationBuffer(delay_s=0.06, jitter_s=0.0, dt=0.02, rng=rng)
    markers = []
    for k in range(10):
        buf.push(k * 0.02, {"k": k})
        markers.append(buf.sample(k * 0.02)["k"])
    assert markers[-1] <= 7  # 0.06 с назад = минимум 3 кадра
    # джиттер: U[0, jitter] может увести глубже штатной задержки
    buf_j = ObservationBuffer(delay_s=0.0, jitter_s=0.5, dt=0.02, rng=np.random.default_rng(7))
    buf_j.push(0.0, {"k": -1})
    got_old = False
    for k in range(1, 30):
        buf_j.push(k * 0.02, {"k": k})
        if buf_j.sample(k * 0.02)["k"] < k - 1:
            got_old = True
    assert got_old


# ── 13–15. Метрики v2 ────────────────────────────────────────────────────────

def test_n_mean_g_consistency_and_lock_fraction_bounds() -> None:
    res = collect(_sc(), stride=100_000)
    assert res.t_end is not None and res.t_end > 0
    assert abs(res.n_mean_g - res.n_int / res.t_end) < 1e-9
    assert 0.0 <= res.lock_fraction <= 1.0
    assert res.cpa_m == res.miss_m  # deprecated alias с ясной семантикой


def test_terminal_frame_does_not_double_count_effort() -> None:
    from navedenie.engine import run

    sc = _sc()
    res = collect(sc, stride=100_000)
    manual = sum(fr.n_req * sc.dt for fr in run(sc) if fr.event not in ("hit", "miss_pass"))
    assert abs(res.n_int - manual) < 1e-9


def test_metrics_version_field_in_run_response() -> None:
    from navedenie.app import ScenarioIn, run_once

    out = run_once(ScenarioIn(range_m=6000, t_max=12))
    assert out["metrics_version"] == 4
    for key in ("cpa_m", "trigger_range_m", "t_end", "lock_time_s", "lock_fraction", "n_mean_g", "terminal_zem_m"):
        assert key in out
    # v4: разделение метрик промаха h_cv / h₀ / R_end + модель движения
    for key in ("h_cv_m", "h0_m", "end_range_m", "model"):
        assert key in out
    # legacy-alias terminal_zem_m == h_cv_m (численно стабилен на кинематике)
    assert out["terminal_zem_m"] == out["h_cv_m"]
    # v3: диагностика N_eff и версии стенда/схемы признаков
    for key in ("model_version", "feature_schema_version", "n_eff_median", "n_eff_valid_frac", "sat_frac"):
        assert key in out
    fr = out["frames"][0]
    for key in ("theta", "theta_dot", "rho", "n_eff", "n_eff_valid", "n_eff_reason", "sat", "tgo"):
        assert key in fr


# ── 16–17. Честность сравнения ───────────────────────────────────────────────

def test_compare_rows_share_same_t_max() -> None:
    from navedenie.app import BrainCompareIn, ScenarioIn, brain_compare

    body = BrainCompareIn(scenario=ScenarioIn(range_m=6000, t_max=9), kinds=["stub"], laws=["pn_gsn"])
    out = brain_compare(body)
    t_maxs = {r["t_max"] for r in out["results"]}
    assert len(t_maxs) == 1
    sensory = {r["kind"]: r.get("sensory") for r in out["results"]}
    assert sensory["pn_gsn"] is True and sensory["pn"] is False  # oracle помечен


def test_local_target_traj_matches_python() -> None:
    """Паритет цели локального демо-движка и python (мини-версия make check-parity)."""
    import subprocess
    import tempfile
    from pathlib import Path

    from navedenie.sim import spawn as py_spawn

    sc = _sc(maneuver="dive", n_target=3.0, t_max=2.0, dt=0.02)
    _m, t_body = py_spawn(sc)
    py_pts = []
    t = 0.0
    while t < 2.0:
        integrate(t_body, target_accel(t_body, sc, t), 0.02)
        t += 0.02
        py_pts.append(t_body.p.tolist())
    with tempfile.TemporaryDirectory() as td:
        bundle = Path(td) / "ls.mjs"
        subprocess.run(
            ["npx", "esbuild", "web/src/localSim.ts", "--bundle", "--format=esm",
             f"--outfile={bundle}", "--log-level=error"],
            check=True, cwd=Path(__file__).resolve().parent.parent,
        )
        import json

        script = (
            "import { pathToFileURL } from 'node:url'\n"
            f"const mod = await import(pathToFileURL('{bundle}').href)\n"
            f"const base = {json.dumps({'aspect': 'head-on', 'v_m': 780, 'v_t': 240, 'range_m': 6000, 'off_axis_m': 200, 'n_max': 30, 'n_target': 3, 'alt_m': 4000, 't_max': 2.2, 'dt': 0.02, 'mode': 'pn', 'pn_n': 4, 'law': 'pn', 'circuit_gain': 1.15, 'tau_s': 0.025, 'fov_deg': 14, 'bio_fov_deg': 165, 'seeker_delay_s': 0, 'kill_radius_m': 45, 'noise_az_deg': 0, 'noise_range_m': 0, 'lock_drop_p': 0, 'seeker_jitter_s': 0, 'retina_death_p': 0, 'retina_dropout_p': 0, 'brain': 'stub', 'maneuver': 'dive'})}\n"
            "const frames = [...mod.localRun(base)]\n"
            "console.log(JSON.stringify(frames.map((f) => f.target)))\n"
        )
        script_file = Path(td) / "check.mjs"
        script_file.write_text(script)
        out = subprocess.run(["node", str(script_file)], check=True, capture_output=True, text=True).stdout
    js_pts = json.loads(out)
    for k in range(len(py_pts)):
        js = js_pts[k + 1] if k + 1 < len(js_pts) else None
        assert js is not None
        assert max(abs(a - b) for a, b in zip(js, py_pts[k])) < 1e-6


# ── 18. Абляция зон причинно влияет на выход ─────────────────────────────────

def test_connectome_zone_ablation_changes_motor_output() -> None:
    from navedenie.circuit import FEAT_DIM, ConnectomeCircuit

    c = ConnectomeCircuit()
    c.reset()
    feat = np.full(FEAT_DIM, 0.4)
    c.feat = feat.copy()
    out_before = np.tanh(c.W_dn @ c.hidden())
    zones = c.channel_zones
    mask = np.array([z == "пеленг" for z in zones], dtype=bool)
    c.W_fx[mask, :] = 0.0
    out_after = np.tanh(c.W_dn @ np.tanh(c.W_fx @ feat + c.b_fx))
    assert not np.allclose(out_before, out_after, atol=1e-9)


# ── 19–20. Воспроизводимость и реальная стохастика ───────────────────────────

def test_same_seed_reproduces_with_noise_and_dropout() -> None:
    sc = _sc(noise_az_deg=3.0, noise_range_m=80.0, lock_drop_p=0.05,
             retina_death_p=0.1, retina_dropout_p=0.05, seed=42)
    a = collect(sc, stride=100_000).cpa_m
    b = collect(sc, stride=100_000).cpa_m
    assert a == b


def test_different_seed_changes_stochastic_result() -> None:
    outs = set()
    for seed in (1, 2, 3):
        sc = _sc(noise_az_deg=3.0, lock_drop_p=0.05, seed=seed, mode="bio")
        outs.add(round(collect(sc, stride=100_000).cpa_m, 3))
    assert len(outs) > 1  # источник случайности реально влияет на сенсорного участника


# ── ПН через ГСН: сенсорный закон ────────────────────────────────────────────

def test_seeker_pn_uses_only_decoded_rates() -> None:
    v = np.array([700.0, 0.0, 0.0])
    a1 = pn_seeker_accel(0.01, 0.0, v, 4.0, 30.0)
    a2 = pn_seeker_accel(0.0, 0.01, v, 4.0, 30.0)
    assert a1[1] > 0  # положительная азимутальная скорость → команда вправо
    assert a2[2] > 0  # положительная скорость угла места → команда вверх
    a0 = pn_seeker_accel(0.0, 0.0, v, 4.0, 30.0)
    assert np.allclose(a0, 0.0)
