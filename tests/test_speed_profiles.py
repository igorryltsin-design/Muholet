"""Профили скорости цели, честный v_t, scheduled PN и полином-суррогат (P1).

Проверяем: v_t — фактический модуль скорости цели во всех аспектах; режимы
constant/accelerate/decelerate/pulse/sine детерминированы и ограничены;
продольное ускорение не смешивается с нормальным манёвром; scheduled PN —
отдельный закон; суррогат — мгновенный readout с меняющимся локальным
коэффициентом и разбиением flight-домена по эпизодам.
"""

import numpy as np
import pytest

from navedenie.circuit import FEAT_DIM, FEATURE_SCHEMA_VERSION, ConnectomeCircuit, FlyCircuit
from navedenie.engine import collect
from navedenie.formula import _flight_fit, _poly_design, brain_dn, brain_formula, brain_formula_c, collect_flight_features
from navedenie.pn import scheduled_pn_accel, scheduled_pn_seeker_accel
from navedenie.sim import G, Scenario, integrate_target, spawn, target_long_accel
from navedenie.train import PROTOCOL, TEST_SPEED_MODES, TEST_VM_GRID, TEST_VT_GRID


def _sc(**kw) -> Scenario:
    base = dict(aspect="head-on", mode="pn", law="pn", v_m=780, v_t=260, range_m=6000,
                off_axis_m=200, t_max=8, dt=0.02, pn_n=4)
    base.update(kw)
    return Scenario(**base)  # type: ignore[arg-type]


# ── 10. v_t — фактический модуль скорости во всех аспектах ────────────────────

@pytest.mark.parametrize("aspect", ["head-on", "beam", "tail-chase"])
def test_v_t_means_actual_initial_speed_in_all_aspects(aspect: str) -> None:
    _m, target = spawn(_sc(aspect=aspect, v_t=300.0))
    assert abs(float(np.linalg.norm(target.v)) - 300.0) < 1e-9


# ── 11–15. Профили скорости цели ─────────────────────────────────────────────

def test_constant_mode_preserves_speed_exactly() -> None:
    """constant: модуль скорости цели сохраняется (прежнее поведение)."""
    sc = _sc(maneuver="turn", n_target=4.0, target_speed_mode="constant")
    _m, target = spawn(sc)
    v0 = float(np.linalg.norm(target.v))
    for t in np.arange(0, 6, 0.02):
        integrate_target(target, sc, float(t), 0.02)
        assert abs(float(np.linalg.norm(target.v)) - v0) < 1e-9
    # и продольное ускорение ровно ноль
    assert target_long_accel(target, sc, 1.0) == 0.0


def test_accelerate_raises_speed_to_upper_bound() -> None:
    sc = _sc(target_speed_mode="accelerate", target_longitudinal_g=1.0, target_speed_min=200, target_speed_max=420)
    _m, target = spawn(sc)
    speeds = []
    for t in np.arange(0, 30, 0.02):
        integrate_target(target, sc, float(t), 0.02)
        speeds.append(float(np.linalg.norm(target.v)))
    assert speeds[len(speeds) // 2] > speeds[0]  # растёт
    assert speeds[-1] == pytest.approx(420.0, abs=1e-6)  # вышла на верхнюю границу


def test_decelerate_lowers_speed_to_lower_bound() -> None:
    sc = _sc(target_speed_mode="decelerate", target_longitudinal_g=0.5, target_speed_min=180, target_speed_max=520)
    _m, target = spawn(sc)
    speeds = []
    for t in np.arange(0, 20, 0.02):
        integrate_target(target, sc, float(t), 0.02)
        speeds.append(float(np.linalg.norm(target.v)))
    assert speeds[len(speeds) // 2] < speeds[0]
    assert speeds[-1] == pytest.approx(180.0, abs=1e-6)


@pytest.mark.parametrize("mode", ["pulse", "sine"])
@pytest.mark.parametrize("phase", [0.0, 1.3])
def test_pulse_and_sine_reproducible_under_same_seed(mode: str, phase: float) -> None:
    def _run() -> list[float]:
        sc = _sc(
            target_speed_mode=mode,  # type: ignore[arg-type]
            target_longitudinal_g=0.8,
            target_speed_min=150,
            target_speed_max=400,
            target_speed_period_s=4.0,
            target_speed_phase=phase,
            seed=42,
        )
        _m, target = spawn(sc)
        speeds = []
        for t in np.arange(0, 5, 0.02):
            integrate_target(target, sc, float(t), 0.02)
            speeds.append(float(np.linalg.norm(target.v)))
        res = collect(_sc(target_speed_mode=mode, target_speed_phase=phase, target_speed_period_s=4.0, seed=42),
                      stride=100_000)
        speeds.append(res.cpa_m)
        return speeds

    assert _run() == _run()  # полная воспроизводимость при одинаковом seed


def test_longitudinal_accel_does_not_disturb_normal_maneuver() -> None:
    """Продольное ускорение не учитывается как нормальная перегрузка: вираж
    крутит вектор скорости, продольное — только модуль (в границах)."""
    sc = _sc(maneuver="turn", n_target=3.0, target_speed_mode="accelerate",
             target_longitudinal_g=1.0, target_speed_min=150, target_speed_max=600)
    _m, target = spawn(sc)
    heading0 = target.v / np.linalg.norm(target.v)
    turn_only = _sc(maneuver="turn", n_target=3.0)
    _m2, target2 = spawn(turn_only)
    # шаг с обоими профилями: поворот происходит и модуль растёт
    integrate_target(target, sc, 0.0, 0.02)
    heading1 = target.v / np.linalg.norm(target.v)
    assert float(np.dot(heading0, heading1)) < 1.0 - 1e-6  # вектор повернулся (норм. манёвр)
    assert float(np.linalg.norm(target.v)) > float(np.linalg.norm(target2.v)) + 0.1  # модуль вырос
    # продольная составляющая ортогональна виражу: target_long_accel — скаляр вдоль v
    a_long = target_long_accel(target, sc, 0.0)
    assert a_long == pytest.approx(1.0 * G, rel=1e-9)


# ── scheduled PN: отдельный контрольный закон ────────────────────────────────

def test_scheduled_pn_matches_pn_when_n_constant() -> None:
    """k_rho=0: scheduled PN вырождается в классическую ПН с N=N0."""
    r = np.array([4000.0, 300.0, 0.0])
    v_m = np.array([700.0, 0, 0])
    v_t = np.array([-240.0, 0, 0])
    a0 = scheduled_pn_accel(r, v_m, v_t, rho=2.0, n0=4.0, k_rho=0.0, n_sched_min=2.0, n_sched_max=6.0, n_lim=30.0)
    from navedenie.pn import ppn_accel

    a1 = ppn_accel(r, v_m, v_t, 4.0, 30.0)
    assert np.allclose(a0, a1, atol=1e-12)


def test_scheduled_pn_clip_and_sensor_variant() -> None:
    r = np.array([4000.0, 300.0, 0.0])
    v_m = np.array([700.0, 0, 0])
    v_t = np.array([-240.0, 0, 0])
    # rho=100 → N зажат сверху расписанием (предел команды остаётся 30 g)
    a_max = scheduled_pn_accel(r, v_m, v_t, rho=100.0, n0=3.0, k_rho=0.8, n_sched_min=2.0, n_sched_max=6.0, n_lim=30.0)
    from navedenie.pn import ppn_accel, n_eff_from

    a6 = ppn_accel(r, v_m, v_t, 6.0, 30.0)
    assert np.allclose(a_max, a6, atol=1e-12)
    # сенсорный вариант: команда строится из декодированных ω и измеренного rho
    v = np.array([700.0, 0, 0])
    a_s = scheduled_pn_seeker_accel(0.02, 0.0, rho=0.0, v_m=v, n0=3.0, k_rho=0.8, n_sched_min=2.0, n_sched_max=6.0, n_lim=30.0)
    assert a_s[1] > 0  # положительная ω_аз → команда вправо
    n_eff, reason, _q = n_eff_from(a_s, r, v_m, v_t, n_max=30.0)
    assert n_eff is not None  # команда умеренная — вне насыщения


# ── 16–18. Полиномиальный суррогат мгновенного readout ────────────────────────

def test_polynomial_contains_no_absolute_time() -> None:
    """В суррогате нет времени t: мономы строятся только из признаков."""
    from navedenie.formula import _grid_surrogate

    circuit = FlyCircuit(kind="stub")
    deg, terms, _fits, _flight = _grid_surrogate("stub", circuit, 2, per_ep=None)
    assert all("t" != k and "time" not in k for k, _s, _r in terms)
    # и в выгруженном коде переменная времени не появляется
    code = brain_formula_c("stub", degree=2)
    assert "feat[t]" not in code and "t_s" not in code


def test_cross_terms_change_local_angular_rate_gain() -> None:
    """Перекрёстные члены дают переменный локальный коэффициент по угловой скорости:
    d(yaw)/d(wb) зависит от rho (в отличие от чисто линейного readout)."""
    from navedenie.formula import FEATURE_KEYS, _term_exponents

    def _mono(exponents, point, skip=-1):
        m = 1.0
        for idx, (x, e) in enumerate(zip(point, exponents)):
            if idx == skip:
                continue
            if e:
                m *= x ** e
        return m

    def local_k(coef, terms, point):
        """d(yaw)/d(wb) = сумма coef·∂моном/∂wb в точке point."""
        wb = FEATURE_KEYS.index("wb")
        total = 0.0
        for j, (key, _s, _r) in enumerate(terms):
            exponents = _term_exponents(key)
            if exponents[wb] == 0:
                continue
            total += coef[j] * exponents[wb] * _mono(exponents, point, skip=wb)
        return total

    # коннектом с нелинейными каналами даёт перекрёстные члены полинома
    c = ConnectomeCircuit()
    rng = np.random.default_rng(5)
    X = rng.uniform(-1, 1, size=(4000, FEAT_DIM))
    dn = brain_dn("connectome", c, X)
    A, terms = _poly_design(X, 2)
    coef, *_ = np.linalg.lstsq(A, dn[:, 1], rcond=None)
    lock = FEATURE_KEYS.index("lock")
    rho = FEATURE_KEYS.index("rho")
    low = np.zeros(FEAT_DIM)
    low[lock] = 1.0
    low[rho] = -0.5
    high = np.zeros(FEAT_DIM)
    high[lock] = 1.0
    high[rho] = 0.5
    k_low = local_k(coef, terms, low)
    k_high = local_k(coef, terms, high)
    assert abs(k_high - k_low) > 1e-3, "локальный коэффициент по угловой скорости не меняется с rho"


# ── 18. flight-domain: разбиение по эпизодам ─────────────────────────────────

def test_flight_domain_split_is_by_whole_episodes() -> None:
    circuit = FlyCircuit(kind="stub")
    per_ep = collect_flight_features(circuit, episodes=5)
    assert len(per_ep) == 5
    fit = _flight_fit("stub", circuit, 1, per_ep=per_ep)
    assert fit is not None
    assert fit["sample"] == "flight"
    assert fit["n_episodes_train"] + fit["n_episodes_val"] == 5
    assert fit["n_episodes_val"] >= 2
    assert fit["n_points_train"] + fit["n_points_val"] == sum(p.shape[0] for p in per_ep)


def test_brain_formula_reports_flight_domain_and_versions() -> None:
    out = brain_formula("stub")
    p = out["poly"]
    assert p["title"].startswith("Полиномиальный суррогат мгновенного readout")
    assert "состояния нейронов" in p["not_included"]
    assert out["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert p["domain"] in ("flight", "grid")
    assert p["flight"] is not None and p["flight"]["seed"] == 77
    lg = p["local_gains"]
    assert {"k_az_feat", "k_el_feat", "k_az_phys", "k_el_phys", "n_poly_eff"} <= set(lg)
    # N_poly_eff показывается ТОЛЬКО при явной V_m — таблицей по скоростям
    assert {row["v_m"] for row in lg["n_poly_eff"]} == {650.0, 780.0, 950.0}


# ── 19–20. API и CSV содержат версии и диагностику ───────────────────────────

def test_api_returns_diagnostics_and_versions() -> None:
    from navedenie.app import ScenarioIn, run_once

    out = run_once(ScenarioIn(range_m=6000, t_max=10, mode="bio"))
    assert out["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert "model_version" in out and out["metrics_version"] == 4
    assert out["n_eff_median"] is not None or out["n_eff_count"] == 0
    assert "n_eff_valid_frac" in out and "sat_frac" in out
    fr = out["frames"][0]
    for key in ("theta", "rho", "tau_contact", "n_eff", "n_eff_valid", "sat", "tgo", "speed_mode", "target_speed"):
        assert key in fr


def test_compare_endpoint_labels_scheduled_pn() -> None:
    from navedenie.app import BrainCompareIn, ScenarioIn, brain_compare

    body = BrainCompareIn(scenario=ScenarioIn(range_m=6000, t_max=8), kinds=[], laws=["pn_sched_sensor"])
    out = brain_compare(body)
    row = next(r for r in out["results"] if r["kind"] == "pn_sched_sensor")
    assert row["sensory"] is True  # сенсорная помечена
    # публичное название — русское «МПС с переменным навигационным коэффициентом N»,
    # без английского «scheduled» и без буквы K (навигационный коэффициент — N, §3/§5.5)
    assert "переменным навигационным коэффициентом N" in row["label"]
    assert "scheduled" not in row["label"].lower() and "коэффициент K" not in row["label"]
    assert row["info_group"] == "sensor" and row["sensor_fov_deg"] is not None


def test_speed_matrix_cell_metrics_complete() -> None:
    """Held-out матрица: ячейка = ОДИН эпизод = ОДИН честный CPA (не квартили
    из прореженных кадров). Квартили/распределения CPA строятся по эпизодам
    вне ячейки (см. test_cpa_distribution_across_episodes)."""
    from navedenie.science import _run_cell, _speed_cell_scenario, get_live_circuit

    sc = _speed_cell_scenario(780.0, 260.0, "head-on", "constant", "straight", 1)
    assert sc.seed >= PROTOCOL["seeds"]["test_base"]  # train 11+7k, val 1 — пересечений нет
    circuit = get_live_circuit("stub")
    cell = _run_cell(circuit, sc)
    for key in ("hit", "cpa_m", "t_hit", "effort_gs", "n_eff_median",
                "n_eff_iqr", "sat_frac", "lock_frac"):
        assert key in cell
    # прореженные кадры не порождают внутриячейную «квартильную статистику»
    assert "cpa_q25" not in cell and "cpa_q75" not in cell


def test_cpa_distribution_across_episodes() -> None:
    """Распределение CPA считается ПО МНОЖЕСТВУ эпизодов (по одному CPA на эпизод)."""
    from navedenie.science import _run_cell, _speed_cell_scenario, get_live_circuit

    circuit = get_live_circuit("stub")
    cpas: list[float] = []
    hits = 0
    for cell_i in range(1, 7):
        sc = _speed_cell_scenario(780.0, 260.0, "head-on", "constant", "straight", cell_i)
        cell = _run_cell(circuit, sc)
        cpas.append(float(cell["cpa_m"]))
        hits += int(bool(cell["hit"]))
    assert len(cpas) == 6
    # распределение по эпизодам устойчиво: медиана конечна, разброс конечен
    med = float(np.median(cpas))
    assert np.isfinite(med)
    assert hits in range(0, 7)
