"""Режим «Свободная расстановка» (aspect="free"): ручной сценарий — стартовая
точка цели и начальные курсы/подъёмы обеих сторон задаются числами, типовые
аспекты не задействованы. Тесты: точность спавна, аналитика вектора скорости по
азимуту/подъёму, независимость от range_m, дальность из первой рамки, замкнутый
перехват на лобовой и боковой геометриях, регресс типовых аспектов, паритет API."""

from __future__ import annotations

import math

import numpy as np

from navedenie.app import ScenarioIn, run_once
from navedenie.engine import collect
from navedenie.sim import Scenario, spawn, vel_from_angles

FREE = dict(mode="pn", law="pn", v_m=780, v_t=260, n_max=30, pn_n=4, dt=0.02,
            t_max=30, kill_radius_m=45, aspect="free")


def test_free_spawn_exact_geometry() -> None:
    """Цель встает ровно в (free_tx, free_ty, free_talt) от точки пуска, модули
    скоростей равны v_m/v_t, направление цели — по азимуту 180° (строго на −X)."""
    sc = Scenario(alt_m=4000, free_tx=4000, free_ty=3000, free_talt=4100,
                  free_mhdg=0, free_mclimb=0, free_thdg=180, free_tclimb=0,
                  **{k: v for k, v in FREE.items() if k not in ("mode", "law")})
    m, t = spawn(sc)
    assert np.allclose(m.p, [0, 0, 4000], atol=1e-12)
    assert np.allclose(t.p, [4000, 3000, 4100], atol=1e-12)
    assert abs(float(np.linalg.norm(m.v)) - 780) < 1e-9
    assert np.allclose(t.v, [-260.0, 0.0, 0.0], atol=1e-9)


def test_vel_from_angles_analytics() -> None:
    """V·(cosγ·cosψ, cosγ·sinψ, sinγ): азимут 90° — чистый +Y; подъём 30° —
    v_z = V·sin30 и горизонталь V·cos30; модуль сохраняется на любых углах."""
    assert np.allclose(vel_from_angles(700, 90, 0), [0, 700, 0], atol=1e-9)
    v = vel_from_angles(700, 0, 30)
    assert abs(v[2] - 350.0) < 1e-9 and abs(v[0] - 700 * math.cos(math.radians(30))) < 1e-9
    rng = np.random.default_rng(3)
    for _ in range(30):
        speed, hdg, climb = 100 + rng.random() * 900, rng.uniform(-180, 180), rng.uniform(-45, 45)
        assert abs(float(np.linalg.norm(vel_from_angles(speed, hdg, climb))) - speed) < 1e-9


def test_free_ignores_range_m_and_engine_uses_true_range() -> None:
    """range_m в ветке free не участвует: та же расстановка с любым range_m
    даёт тот же спавн, а стартовая дальность в первом кадре — истинная
    наклонная из пифагоровой тройки (5000.9999 м при 4000/3000/Δ100)."""
    base = dict(alt_m=4000, free_tx=4000, free_ty=3000, free_talt=4100, free_thdg=180)
    a = spawn(Scenario(**base, range_m=8000, **{k: v for k, v in FREE.items() if k not in ("mode", "law")}))
    b = spawn(Scenario(**base, range_m=12345, **{k: v for k, v in FREE.items() if k not in ("mode", "law")}))
    assert np.allclose(a[0].p, b[0].p) and np.allclose(a[1].p, b[1].p)
    res = collect(Scenario(**base, t_max=5, dt=0.02, v_m=780, v_t=260, mode="pn", law="pn", aspect="free"), stride=100_000)
    expected = math.sqrt(4000**2 + 3000**2 + 100**2)
    assert abs(res.frames[0].range_m - expected) < 1e-4, res.frames[0].range_m


def test_free_head_on_intercepts() -> None:
    """Лобовая ручная геометрия (цель на +X на той же высоте, курс 180°) —
    классический встречный перехват: попадание, η = 180°."""
    res = collect(Scenario(free_tx=8000, free_ty=0, free_talt=4000, free_thdg=180, **FREE))
    assert res.hit, res.miss_m
    assert res.impact_angle_deg is not None and res.impact_angle_deg > 179.0


def test_free_side_geometry_intercepts() -> None:
    """Боковой ручной сценарий: цель в (2000, 3000), курс 140° — ракета с
    начальным азимутом 55° догоняет по упреждению: попадание в сферу БЧ."""
    res = collect(Scenario(free_tx=2000, free_ty=3000, free_talt=4000,
                           free_mhdg=55, free_thdg=140, **FREE))
    assert res.hit, res.miss_m


def test_aspect_presets_unchanged() -> None:
    """Регресс типовых аспектов: ветка free не сдвинула формулы head-on/beam/
    tail-chase (аналитика по исходной формуле raw·range/|raw|)."""
    sc = Scenario(aspect="head-on", range_m=8000, off_axis_m=400, alt_m=4000, v_m=780, v_t=260)
    m, t = spawn(sc)
    raw = np.array([1.0, 400 / 8000, 80 / 8000])
    p_t = np.array([0, 0, 4000]) + raw * (8000 / float(np.linalg.norm(raw)))
    assert np.allclose(t.p, p_t, atol=1e-9)
    assert np.allclose(t.v, [-260, 0, 0], atol=1e-12)
    beam = spawn(Scenario(aspect="beam", range_m=8000, alt_m=4000, v_t=260))
    assert np.allclose(beam[1].v, [0, 260, 0], atol=1e-12)
    assert abs(float(np.linalg.norm(beam[1].p - beam[0].p)) - 8000) < 1e-6


def test_api_run_free_matches_collect() -> None:
    """Проводка API: плоское тело /api/run с aspect=free и free_* полями
    считает ровно тот же промах, что in-process collect (поля не теряются)."""
    kw = {**FREE, "free_tx": 6000, "free_ty": 1500, "free_talt": 4000,
          "free_mhdg": 10, "free_thdg": 170}
    out = run_once(ScenarioIn(**kw))
    ref = collect(Scenario(**kw), stride=100_000)
    assert abs(out["miss_m"] - ref.miss_m) <= 0.01
    assert out["hit"] == ref.hit
