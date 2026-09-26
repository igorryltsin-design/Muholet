"""Закон TPN (истинный метод пропорционального сближения): команда
a = N·V_c·(ω_ЛВ × r̂) по нормали к линии визирования (Зархан, Tactical Missiles
Guidance, гл. 2; Siourinas et al. 2021). Тесты: нулевая команда на радиальном
сближении и при убегании, аналитическое значение на числах, ортогональность V_к,
насыщение, отличие от МПС на нерастраверсной геометрии, замкнутый перехват,
реестр законов и проводка API."""

from __future__ import annotations

import math

import numpy as np

from navedenie.app import ScenarioIn, run_once
from navedenie.engine import collect
from navedenie.glossary import LAW_LABEL
from navedenie.pn import LAWS, ppn_accel, tpn_accel
from navedenie.sim import G, Scenario

BASE = dict(
    mode="pn", v_m=780, v_t=260, n_max=30, pn_n=4, dt=0.02,
    t_max=30, kill_radius_m=45, range_m=8000,
)


def test_tpn_zero_on_radial_closure() -> None:
    """Чистое лобовое сближение без траверса: ω_ЛВ = 0 — команда TPN ровно нулевая
    (корневая причина «условия параллельного сближения» Зархана)."""
    r = np.array([5000.0, 0.0, 0.0])
    a = tpn_accel(r, np.array([700.0, 0.0, 0.0]), np.array([-300.0, 0.0, 0.0]), 4.0, 30.0)
    assert np.allclose(a, 0.0, atol=1e-9)


def test_tpn_analytic_value_on_numbers() -> None:
    """Аналитика на числах: r=(4000,300,0), V_к=(700,0,0), V_ц=(−300,0,0), N=4.
    ω_z = 300·1000/16 090 000; V_c = 4·10⁶/√16 090 000; команда по ⊥ ЛВ даёт
    a_raw = N·V_c·ω_z·(−r̂_y, r̂_x, 0), стендовая проекция ⊥ V_к снимает
    x-компоненту: |a| = N·V_c·ω_z·r̂_x."""
    r = np.array([4000.0, 300.0, 0.0])
    v_m = np.array([700.0, 0.0, 0.0])
    v_t = np.array([-300.0, 0.0, 0.0])
    sqrt_r2 = math.sqrt(16_090_000.0)  # |r|² = 4000² + 300²
    omega_z = 300_000.0 / 16_090_000.0
    vc = 4_000_000.0 / sqrt_r2
    expected_y = 4.0 * vc * omega_z * (4000.0 / sqrt_r2)
    a = tpn_accel(r, v_m, v_t, 4.0, 30.0)
    assert np.allclose(a, [0.0, expected_y, 0.0], atol=1e-9), a
    assert expected_y > 74.0  # контроль величины: ~74.16 м/с² (7.56 g)


def test_tpn_orthogonal_and_clipped() -> None:
    """Команда ортогональна скорости ракеты (конвенция стенда) и не превышает
    n_max·g даже при заведомо неисполнимом коэффициенте."""
    rng = np.random.default_rng(7)
    for _ in range(50):
        r = rng.normal(size=3) * 3000.0
        v_m = np.array([800.0, 0.0, 0.0]) + rng.normal(size=3) * 50.0
        v_t = rng.normal(size=3) * 300.0
        a = tpn_accel(r, v_m, v_t, 4.0, 30.0)
        assert abs(float(np.dot(a, v_m))) <= 1e-6 * float(np.linalg.norm(a) * np.linalg.norm(v_m) + 1.0)
        assert float(np.linalg.norm(a)) <= 30.0 * G + 1e-9
    a_sat = tpn_accel(np.array([4000.0, 300.0, 0.0]), np.array([700.0, 0.0, 0.0]),
                      np.array([-300.0, 0.0, 0.0]), 1000.0, 30.0)
    assert abs(float(np.linalg.norm(a_sat)) - 30.0 * G) < 1e-9


def test_tpn_no_command_when_not_closing() -> None:
    """Классическое отличие от МПС на нерастраверсной геометрии (ЛВ ⊥ скорости,
    V_c = 0): истинный МПС команды не даёт, обычный — доворачивает с полной
    скоростью ракеты. Также убегание (V_c < 0) обрезается до нуля."""
    r = np.array([0.0, 4000.0, 0.0])
    v_m = np.array([700.0, 0.0, 0.0])
    a_tpn = tpn_accel(r, v_m, np.zeros(3), 4.0, 30.0)
    a_pn = ppn_accel(r, v_m, np.zeros(3), 4.0, 30.0)
    assert float(np.linalg.norm(a_tpn)) < 1e-9, a_tpn
    assert float(np.linalg.norm(a_pn)) > 100.0, a_pn
    # ракета от цели убегает по той же ЛВ: V_c < 0 → ноль, а не «реверс» команды
    a_rec = tpn_accel(r, np.array([-700.0, 0.0, 0.0]), np.zeros(3), 4.0, 30.0)
    assert np.allclose(a_rec, 0.0, atol=1e-9)


def test_tpn_closed_loop_hits_all_geometries() -> None:
    """Замкнутый контур: на типовых аспектах встречи закон летает так же классически —
    попадание во сферу срабатывания на лобовой (в т.ч. со смещением), бортовой
    и догонной геометриях."""
    for aspect, off in (("head-on", 0.0), ("head-on", 400.0), ("beam", 400.0), ("tail-chase", 400.0)):
        res = collect(Scenario(**BASE, law="tpn", aspect=aspect, off_axis_m=off), stride=100_000)
        assert res.hit, (aspect, off, res.miss_m)


def test_tpn_registered_everywhere() -> None:
    """Реестр законов: id tpn присутствует в LAWS и LAW_LABEL одновременно —
    иначе фронт и сверка терминов (test_glossary) его не увидят."""
    assert "tpn" in LAWS
    assert "tpn" in LAW_LABEL
    assert set(LAWS) == set(LAW_LABEL)


def test_api_run_accepts_tpn() -> None:
    """Проводка API: /api/run с law="tpn" считает прогон и совпадает с in-process
    collect (тот же закон, а не молчаливый откат на базовый МПС)."""
    kw = {**BASE, "law": "tpn", "aspect": "head-on", "off_axis_m": 400.0}
    out = run_once(ScenarioIn(**kw))
    ref = collect(Scenario(**kw), stride=100_000)
    assert abs(out["miss_m"] - ref.miss_m) <= 0.01
    assert out["hit"] == ref.hit
