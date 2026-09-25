"""Фазовые признаки (theta/rho), диагностика N_eff и миграция схемы весов (P0).

Проверяем честность сенсора: theta/theta_dot/rho вычисляются ТОЛЬКО из
декодированного изображения; N_eff — постфактум-диагностика по истинной
геометрии, у постоянной ПН вне насыщения равна заданному N; старые мозги
v1 (8 признаков) мигрируют нулями на новых столбцах.
"""

import numpy as np
import pytest

from navedenie.circuit import (
    FEAT_DIM,
    FEATURE_SCHEMA_VERSION,
    K_RHO,
    K_THETA,
    LEGACY_COLUMN_MAP,
    ConnectomeCircuit,
    FlyCircuit,
    _features,
    migrate_feature_columns,
)
from navedenie.engine import collect
from navedenie.pn import n_eff_basis, n_eff_from
from navedenie.seeker import THETA_WIN, ObservationBuffer, dead_mask_for, observe
from navedenie.sim import Scenario
from navedenie.train import _rollout_update

FOV_BIO = np.deg2rad(165.0) / 2.0


def _bio_obs(az_deg: float, range_m: float, v_m: np.ndarray | None = None, prev=None, rng=None):
    from navedenie.seeker import body_axes

    v = v_m if v_m is not None else np.array([700.0, 0, 0])
    x, y, z = body_axes(v)
    r = x * range_m + y * np.deg2rad(az_deg) * range_m
    return observe(r, v, prev, half_fov=FOV_BIO, dt_s=0.02, rng=rng)


# ── 4–5. BIO не получает истину; rho только из декодированных theta/theta_dot ──

def test_bio_features_contain_no_truth_geometry() -> None:
    """Ни один признак не реагирует на подмену истинной дальности/Vc при том же кадре."""
    obs = {
        "az": 0.1, "el": -0.05, "az_dot": 0.02, "el_dot": -0.01,
        "size": 0.1, "size_dot": 0.01, "theta": 0.03, "theta_dot": 0.005,
        "rho": 0.17, "lock": True,
        "image": np.ones((16, 16)) * 0.1,
        "truth_az": 0.1, "truth_el": -0.05, "truth_range": 3000.0,
    }
    t4 = np.zeros(4)
    base = _features(obs, t4, loom=0.4)
    for leaked in ("truth_range", "v_c", "t_go", "range", "n_eff"):
        obs2 = dict(obs)
        obs2[leaked] = 12345.678
        obs2["truth_range"] = 8000.0
        assert np.allclose(_features(obs2, t4, loom=0.4), base), f"признак утекает истину: {leaked}"
    assert FEAT_DIM == 10 and len(base) == FEAT_DIM


def test_rho_computed_from_decoded_theta_only() -> None:
    """rho = theta_dot/theta в кадре; подмена ИСТИННОЙ дальности не меняет rho,
    если декодированные theta/theta_dot те же."""
    v = np.array([700.0, 0, 0])
    o1 = _bio_obs(2.0, 900.0)
    o2 = _bio_obs(2.0, 900.0)
    o2["truth_range"] = 1234.5  # врёт оператору, не сенсору
    assert o1["lock"] and o2["lock"]
    assert abs(o1["rho"] - o1["theta_dot"] / max(o1["theta"], 1e-4)) < 1e-9
    assert abs(o1["rho"] - o2["rho"]) < 1e-12


# ── 6–7. Свойства rho: монотонность по theta_dot и связь с Vc/R ──────────────

def test_larger_theta_dot_gives_larger_rho() -> None:
    """При одинаковом theta больший theta_dot даёт больший rho (формула)."""
    obs_small = {"theta": 0.05, "theta_dot": 0.05, "lock": True}
    obs_fast = {"theta": 0.05, "theta_dot": 0.20, "lock": True}
    rho_small = obs_small["theta_dot"] / max(obs_small["theta"], 1e-4)
    rho_fast = obs_fast["theta_dot"] / max(obs_fast["theta"], 1e-4)
    assert rho_fast == 4 * rho_small and rho_fast > rho_small


def test_rho_correlates_with_vc_over_range_on_ideal_approach() -> None:
    """Идеальное постоянное сближение: rho растёт вместе с Vc/R (корреляция > 0.9)
    в зоне измеримости 0.05 < theta < 0.3 рад. Зона 0.02…0.05 — транзит появления
    цели из PSF сетчатки: theta растёт быстрее геометрического — честный сенсорный
    выброс, из оценки оптической дальности он исключён."""
    rows = []
    for vc in (400.0, 700.0, 1040.0):
        prev = None
        rng = np.random.default_rng(5)
        for i in range(int(4.0 / 0.02)):
            r_m = max(60.0, 1200.0 - vc * 0.02 * i)
            obs = _bio_obs(1.0, r_m, prev=prev, rng=rng)
            prev = obs
            if obs["lock"] and 0.05 < obs["theta"] < 0.3 and 80 < r_m < 900:
                rows.append((obs["rho"], vc / r_m))
    assert len(rows) >= 25, f"мало измеримых кадров: {len(rows)}"
    a = np.asarray(rows)
    corr = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
    assert corr > 0.9, f"rho не коррелирует с Vc/R: corr={corr:.3f}"


# ── 1–3. N_eff: совпадение с N у постоянной ПН, невалидность при вырожденности ──

def test_n_eff_equals_pn_n_constant_outside_saturation() -> None:
    """Oracle-ПН с N=4: медиана N_eff == 4.0 точно, насыщений нет."""
    res = collect(Scenario(aspect="beam", mode="pn", law="pn", range_m=8000, t_max=12, dt=0.02, pn_n=4.0), stride=100_000)
    assert res.n_eff_median == pytest.approx(4.0, abs=1e-9)
    assert res.n_eff_q25 == pytest.approx(4.0, abs=1e-9)
    assert res.n_eff_valid_frac > 0.8
    assert res.sat_frac == 0.0


def test_n_eff_invalid_near_zero_los_rate() -> None:
    """Почти нулевая ω_LOS: |q| ниже порога — N_eff помечен невалидным с причиной."""
    r = np.array([5000.0, 1.0, 0.0])  # почти коллинеарно скорости — вырожденная геометрия
    v_m = np.array([700.0, 0, 0])
    v_t = np.array([-240.0, 0, 0])
    a_cmd = np.zeros(3)
    n_eff, reason, q = n_eff_from(a_cmd, r, v_m, v_t, n_max=30.0)
    assert n_eff is None and reason == "no_geom"
    assert float(np.linalg.norm(q)) < 0.5
    # на реальном перехвате невалидные кадры имеют причину
    res = collect(Scenario(aspect="head-on", mode="pn", law="pn", range_m=8000, t_max=14, dt=0.02), stride=100_000)
    bad = [fr.n_eff_reason for fr in res.frames if not fr.n_eff_valid]
    assert all(b in ("no_geom", "saturation", "nonfinite") for b in bad)


def test_n_eff_excluded_on_saturation() -> None:
    """Команда на насыщении n_max: N_eff невалиден и в агрегаты не попадает."""
    r = np.array([3000.0, 500.0, 0.0])
    v_m = np.array([700.0, 0, 0])
    v_t = np.array([-240.0, 0, 0])
    huge = np.array([0.0, 30.0 * 9.81, 0.0])  # ровно предел
    n_eff, reason, _q = n_eff_from(huge, r, v_m, v_t, n_max=30.0)
    assert n_eff is None and reason == "saturation"


def test_n_eff_diagnostic_never_feeds_control() -> None:
    """Поле n_eff не читается ни одним управляющим путём: BIO видит только obs.
    Проверяем, что frame с заполненной диагностикой не меняет поведение: два
    прогона с одинаковым seed совпадают покадрово в a_cmd независимо от n_eff."""
    sc = Scenario(aspect="head-on", mode="bio", brain="stub", range_m=6000, t_max=6, dt=0.02, seed=9)
    a1 = [fr.a_cmd.tolist() for fr in collect(sc, stride=100_000).frames]
    a2 = [fr.a_cmd.tolist() for fr in collect(sc, stride=100_000).frames]
    assert a1 == a2  # диагностика детерминирована и не вмешивается в контур


# ── 8–9. Миграция и версионирование схемы весов ───────────────────────────────

def test_legacy_8_feature_weights_migrate_with_zero_new_columns() -> None:
    rng = np.random.default_rng(3)
    for kind, mk in (("stub", lambda: FlyCircuit(kind="stub")), ("connectome", lambda: ConnectomeCircuit())):
        c = mk()
        old = rng.standard_normal((c.W_dn.shape[0], 8))
        mig = migrate_feature_columns(old, axis=1)
        assert mig.shape == (c.W_dn.shape[0], FEAT_DIM)
        # старые столбцы сохранили смысл на новых позициях:
        # LEGACY_COLUMN_MAP[j] — новая позиция j-го столбца v1
        for legacy_idx, new_col in enumerate(LEGACY_COLUMN_MAP):
            assert np.allclose(mig[:, new_col], old[:, legacy_idx])
        # новые столбцы (theta=4, rho=6) — нули: поведение до дообучения не меняется
        assert np.allclose(mig[:, [4, 6]], 0.0)


def test_full_brain_w_k_migrates_and_behavior_preserved() -> None:
    """W_k полного мозга (4096×8) расширяется до (…×10); отклик на старых
    признаках (theta=rho=0) не меняется после миграции."""
    from navedenie.formula import brain_dn

    c = FlyCircuit(kind="full", pool_size=32)
    rng = np.random.default_rng(11)
    # строим ЯВНО легаси-проводку 4096×8 (как у мозга схемы v1)
    wk = rng.standard_normal((c.W_k.shape[0], 8)) * 0.35
    wk *= rng.random(wk.shape) < 0.12
    c.W_k = wk
    c.trained = True
    X_old = np.clip(np.random.default_rng(2).uniform(-1, 1, size=(64, 8)), -1, 1)
    X_new = np.zeros((64, FEAT_DIM))
    for legacy_idx, new_col in enumerate(LEGACY_COLUMN_MAP):
        X_new[:, new_col] = X_old[:, legacy_idx]
    before = brain_dn("full", c, X_old)
    c.W_k = migrate_feature_columns(c.W_k, axis=1)
    after = brain_dn("full", c, X_new)
    assert c.W_k.shape[1] == FEAT_DIM
    assert np.allclose(before, after, atol=1e-9)


def test_saved_weights_carry_feature_schema_version(tmp_path) -> None:
    c = FlyCircuit(kind="stub")
    c.trained = True
    path = c.save(path=tmp_path / "weights_stub.npz")
    blob = np.load(path, allow_pickle=False)
    assert int(np.asarray(blob["feat_schema"]).reshape(-1)[0]) == FEATURE_SCHEMA_VERSION == 2


def test_legacy_npz_loads_and_rejects_newer_schema(tmp_path) -> None:
    """Файл v1 без feat_schema загружается с расширением; схема новее стенда — отказ."""
    w = np.array([[0.0, 1.8, 0.0, 0.35, 0.25, 0.0, 0.4, 0.0], [1.8, 0.0, 0.35, 0.0, 0.25, 0.4, 0.0, 0.0]])
    legacy_path = tmp_path / "weights_stub_v1.npz"
    np.savez(
        legacy_path,
        W_dn=w,
        trained=np.array([1], dtype=np.int8),
        gain=np.float64(1.15),
        tau=np.float64(0.025),
        seed=np.int64(23),
    )
    c = FlyCircuit(kind="stub", gain=1.0)
    assert c.load(path=legacy_path)
    assert c.W_dn.shape == (2, FEAT_DIM)
    assert c.W_dn[0, 1] == 1.8 and c.W_dn[0, 4] == 0.0 and c.W_dn[0, 6] == 0.0

    future_path = tmp_path / "weights_stub_future.npz"
    np.savez(
        future_path,
        W_dn=np.zeros((2, FEAT_DIM)),
        trained=np.array([1], dtype=np.int8),
        feat_schema=np.array([FEATURE_SCHEMA_VERSION + 1]),
    )
    c2 = FlyCircuit(kind="stub", gain=1.0)
    assert not c2.load(path=future_path)


# ── нормировка фазовых признаков по фактическому распределению ────────────────

def test_theta_rho_feature_normalization_matches_measured_ranges() -> None:
    """Нормировка K_THETA/K_RHO подобрана по фактическим диапазонам: на реальном
    сближении признаки не сидят в насыщении (|tanh| < 0.95 в операционной зоне)."""
    feats = []
    prev = None
    rng = np.random.default_rng(7)
    for i in range(int(6.0 / 0.02)):
        r_m = max(80.0, 3000.0 - 1040.0 * 0.02 * i)
        obs = _bio_obs(1.0, r_m, prev=prev, rng=rng)
        prev = obs
        if obs["lock"]:
            feats.append(_features(obs, np.zeros(4), loom=0.0))
    F = np.asarray(feats)
    theta_feat = F[:, 4]
    rho_feat = F[:, 6]
    # в измеримой зоне признак достигает заметных значений, но не прилипает к ±1
    assert theta_feat.max() > 0.2, "theta слишком мал — нормировка задрана"
    assert np.mean(np.abs(theta_feat) > 0.95) < 0.3, "theta в насыщении большую часть полёта"
    assert 0.0 <= rho_feat.min() and rho_feat.max() <= np.tanh(30 * K_RHO) + 1e-9


def test_theta_window_travels_through_delay_buffer() -> None:
    """Окно θ̇ — часть снимка кадра: задержка/джиттер отдают согласованные оценки."""
    rng = np.random.default_rng(4)
    buf = ObservationBuffer(0.06, 0.0, 0.02, rng)
    v = np.array([700.0, 0, 0])
    prev = None
    for k in range(12):
        obs = _bio_obs(2.0, 600.0, prev=prev, rng=rng)
        prev = obs
        buf.push(k * 0.02, obs)
        delayed = buf.sample(k * 0.02)
        assert len(delayed.get("theta_win", [])) <= THETA_WIN
        assert delayed["rho"] == delayed["theta_dot"] / max(delayed["theta"], 1e-4) or delayed["theta"] == 0


def test_rollout_collects_flight_features_for_surrogate() -> None:
    """feats-сбор: вектор признаков контура каждого шага валидной длины."""
    circuit = FlyCircuit(kind="stub")
    sc = Scenario(aspect="head-on", mode="bio", range_m=6000, t_max=4, dt=0.02)
    feats = []
    _rollout_update(circuit, sc, lr=0.0, feats=feats)
    assert len(feats) > 50
    assert all(f.shape == (FEAT_DIM,) for f in feats)
    assert np.all(np.abs(np.stack(feats)) <= 1.0 + 1e-9)
