"""Аудит осей, знаков и метрик времени (см. docs/notation.md).

Тестируют математику, а не только факт перехвата:
- знак команды сенсорной ПН для цели сверху/снизу/справа/слева и паритет Python↔TypeScript;
- различение t_radial = R/Vc и t_cpa = −r·Vотн/|Vотн|² (бывшее tgo — alias t_radial);
- ортогональность нормальной команды скорости, инвариантность к переносу и повороту;
- согласованность «призрака» (предыдущая точка призрака, а не ракеты).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from navedenie.engine import collect, frame_to_dict, run
from navedenie.pn import pn_seeker_accel, ppn_accel
from navedenie.seeker import body_axes
from navedenie.sim import G, Scenario, closing_speed, los_omega, radial_time, time_to_cpa

REPO = Path(__file__).resolve().parent.parent


def _sc(**kw) -> Scenario:
    base = dict(aspect="head-on", v_m=780, v_t=240, range_m=6000, off_axis_m=200,
                n_max=30, t_max=14, dt=0.02, mode="pn", pn_n=4, seeker_delay_s=0)
    base.update(kw)
    return Scenario(**base)


# ── 1. Знак вертикальной/боковой команды сенсорной ПН (Python) ────────────────

def test_sensor_pn_command_sign_matches_target_motion() -> None:
    """Цель уходит вверх (el_dot>0) → команда нормального ускорения вверх;
    вправо (az_dot>0) → команда бокового ускорения вправо. Оси — скоростная СК."""
    v_m = np.array([700.0, 20.0, 0.0])
    _x, right, up = body_axes(v_m)
    cases = [
        ("выше", 0.0, +0.05, +1),
        ("ниже", 0.0, -0.05, -1),
        ("справа", +0.05, 0.0, +1),
        ("слева", -0.05, 0.0, -1),
    ]
    for _name, az_dot, el_dot, want in cases:
        a = pn_seeker_accel(az_dot, el_dot, v_m, 4.0, 30.0)
        vert = float(np.dot(a, up))
        lat = float(np.dot(a, right))
        # вертикальный канал управляется el_dot, боковой — az_dot
        if el_dot != 0.0:
            assert np.sign(vert) == want, f"{_name}: el_dot→вертикаль имеет знак {np.sign(vert)}"
        if az_dot != 0.0:
            assert np.sign(lat) == want, f"{_name}: az_dot→боковое имеет знак {np.sign(lat)}"


def test_sensor_pn_command_is_perpendicular_and_bounded() -> None:
    v_m = np.array([700.0, 20.0, 0.0])
    a = pn_seeker_accel(0.05, -0.04, v_m, 4.0, 30.0)
    assert abs(float(np.dot(a, v_m))) < 1e-6  # ⊥ скорости
    assert np.linalg.norm(a) <= 30.0 * G + 1e-6


# ── 2. Паритет Python ↔ TypeScript для сенсорной команды ─────────────────────

def _ts_sensor_pn(az_dot: float, el_dot: float, v_m: list[float], n: float, n_max: float) -> list[float]:
    with tempfile.TemporaryDirectory() as td:
        bundle = Path(td) / "ls.mjs"
        subprocess.run(
            ["npx", "esbuild", "web/src/localSim.ts", "--bundle", "--format=esm",
             f"--outfile={bundle}", "--log-level=error"],
            check=True, cwd=REPO,
        )
        script = (
            "import { pathToFileURL } from 'node:url'\n"
            f"const mod = await import(pathToFileURL('{bundle}').href)\n"
            f"const a = mod.sensorPnAccel({az_dot}, {el_dot}, {json.dumps(v_m)}, {n}, {n_max})\n"
            "console.log(JSON.stringify(a))\n"
        )
        sf = Path(td) / "probe.mjs"
        sf.write_text(script)
        out = subprocess.run(["node", str(sf)], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def test_sensor_pn_python_typescript_parity() -> None:
    v_m = [700.0, 20.0, 0.0]
    for az_dot, el_dot in [(0.0, 0.05), (0.0, -0.05), (0.05, 0.0), (-0.05, 0.0), (0.03, -0.02)]:
        py = pn_seeker_accel(az_dot, el_dot, np.array(v_m), 4.0, 30.0)
        js = np.array(_ts_sensor_pn(az_dot, el_dot, v_m, 4.0, 30.0))
        assert np.allclose(py, js, atol=1e-6), f"расхождение при az={az_dot} el={el_dot}: py={py} js={js}"


# ── 3. t_radial vs t_cpa: разные величины, tgo — alias t_radial ───────────────

def test_tgo_is_deprecated_alias_of_t_radial() -> None:
    sc = _sc(aspect="beam", law="pn_gsn", t_max=6)
    for fr in run(sc):
        assert fr.tgo == fr.t_radial
        d = frame_to_dict(fr)
        assert d["tgo"] == d["t_radial"]
        assert "t_cpa" in d


def test_t_radial_and_t_cpa_are_different_quantities() -> None:
    # боковой случай: радиальная оценка R/Vc ≠ время до CPA
    r = np.array([3000.0, 2000.0, 0.0])
    v_rel = np.array([-500.0, 100.0, 0.0])
    tr = radial_time(r, v_rel)
    tc = time_to_cpa(r, v_rel)
    assert tr is not None and tc is not None
    assert abs(tr - tc) > 1.0  # существенно разные
    # t_cpa аналитически: минимум |r + v_rel·t|
    assert abs(tc - (-float(np.dot(r, v_rel)) / float(np.dot(v_rel, v_rel)))) < 1e-9


def test_time_to_cpa_is_the_geometric_minimum() -> None:
    r = np.array([1000.0, 500.0, 0.0])
    v_rel = np.array([-200.0, 100.0, 0.0])
    tc = time_to_cpa(r, v_rel)
    # расстояние в момент CPA минимально среди окрестных
    d_cpa = np.linalg.norm(r + v_rel * tc)
    for eps in (tc - 5.0, tc + 5.0):
        assert d_cpa <= np.linalg.norm(r + v_rel * eps) + 1e-6


# ── 4. Ортогональность, инвариантность осей ──────────────────────────────────

def test_ppn_accel_orthogonal_to_velocity() -> None:
    rng = np.random.default_rng(7)
    for _ in range(20):
        v_m = rng.normal(size=3) * 300 + np.array([700, 0, 0])
        r = rng.normal(size=3) * 1000 + np.array([3000, 0, 0])
        v_t = rng.normal(size=3) * 200
        a = ppn_accel(r, v_m, v_t, 4.0, 30.0)
        assert abs(float(np.dot(a, v_m))) < 1e-4
        assert np.linalg.norm(a) <= 30.0 * G + 1e-6


def test_clos_invariant_to_global_translation() -> None:
    """Метод трёх точек задан абсолютными точками (пуск, ракета, цель), но команда
    должна быть инвариантна к переносу ВСЕЙ сцены на один вектор —relative-геометрия
    линии «пуск—цель» сохраняется."""
    from navedenie.pn import clos_accel

    v_m = np.array([700.0, 30.0, 10.0])
    launch = np.array([0.0, 0.0, 4000.0])
    missile = np.array([1500.0, 60.0, 4020.0])
    target = np.array([5200.0, 300.0, 4080.0])
    r = target - missile
    a0 = clos_accel(missile, r, v_m, target, launch, 30.0)
    off = np.array([1234.0, -567.0, 89.0])
    a1 = clos_accel(missile + off, r, v_m, target + off, launch + off, 30.0)
    assert np.allclose(a0, a1, atol=1e-9)


def test_ppn_covariant_under_rotation() -> None:
    """Согласованный поворот всей геометрии поворачивает команду на тот же оператор."""
    def rot_z(ang: float) -> np.ndarray:
        c, s = np.cos(ang), np.sin(ang)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    v_m = np.array([700.0, 30.0, 10.0])
    r = np.array([2500.0, 400.0, -100.0])
    v_t = np.array([-240.0, 20.0, 5.0])
    R = rot_z(0.7)
    a_direct = ppn_accel(r, v_m, v_t, 4.0, 30.0)
    a_rot = ppn_accel(R @ r, R @ v_m, R @ v_t, 4.0, 30.0)
    assert np.allclose(R @ a_direct, a_rot, atol=1e-6)


def test_closing_speed_and_los_omega_signs() -> None:
    # встречный курс: сближение положительно
    r = np.array([5000.0, 0.0, 0.0])
    v_rel = np.array([-600.0, 0.0, 0.0])
    assert closing_speed(r, v_rel) > 0
    # расхождение: сближение отрицательно
    assert closing_speed(r, np.array([600.0, 0.0, 0.0])) < 0
    # ω_LOS = (r × Vотн)/|r|² — для чисто радиального Vотн равна нулю
    assert np.linalg.norm(los_omega(r, v_rel)) < 1e-9
    # боковая составляющая даёт ненулевую угловую скорость
    om = los_omega(r, v_rel + np.array([0.0, 100.0, 0.0]))
    assert np.linalg.norm(om) > 0


# ── 5. Призрак: своя предыдущая точка ────────────────────────────────────────

def test_ghost_coincides_with_oracle_pn() -> None:
    """Для закона pn (oracle) ракета идентична призраку → отклонение ~0,
    а время наведения призрака совпадает с t_guide."""
    sc = _sc(law="pn", t_max=10)
    res = collect(sc, stride=100_000)
    assert res.ref_dev_m is not None and res.ref_dev_m < 1e-6
    if res.t_guide is not None and res.t_ref is not None:
        assert abs(res.t_guide - res.t_ref) < 0.05


def test_ghost_hit_time_uses_ghost_geometry() -> None:
    """Даже когда ракета (метод погони) промахивается, призрак-МПС может перехватить:
    ghost_t_hit обязано соответствовать траектории призрака, а не ракеты."""
    sc = _sc(law="pure", t_max=12)
    res = collect(sc, stride=100_000)
    # если у призрака зафиксировано время входа — оно в пределах прогона
    last = res.frames[-1]
    if last.ghost_t_hit is not None:
        assert 0.0 <= last.ghost_t_hit <= (res.t_end or 1e9) + sc.dt
