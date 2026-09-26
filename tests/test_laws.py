import numpy as np

from navedenie.engine import collect
from navedenie.pn import apn_accel, clos_accel, ppn_accel, pure_pursuit_accel
from navedenie.sim import G, Scenario


def _sc(**kw) -> Scenario:
    base = dict(aspect="head-on", v_m=780, v_t=240, range_m=6000, off_axis_m=200,
                n_max=30, t_max=14, dt=0.02, mode="pn", pn_n=4, seeker_delay_s=0)
    base.update(kw)
    return Scenario(**base)  # type: ignore[arg-type]


def test_apn_beats_pn_on_maneuvering_target() -> None:
    # низкая постоянная N против сильной змейки: компенсация ускорения цели решает
    sc_pn = _sc(maneuver="weave", n_target=5.0, pn_n=2.0)
    sc_apn = _sc(maneuver="weave", n_target=5.0, pn_n=2.0, law="apn")
    miss_pn = collect(sc_pn, stride=100_000).miss_m
    miss_apn = collect(sc_apn, stride=100_000).miss_m
    assert miss_apn < miss_pn  # компенсация манёвра цели уменьшает промах


def test_pure_pursuit_misses_head_on() -> None:
    res = collect(_sc(law="pure"), stride=100_000)
    assert np.isfinite(res.miss_m)
    assert not res.hit  # на встречных курсах метод погони не успевает


def test_pure_pursuit_aims_velocity_at_target() -> None:
    """Метод погони [Гусев §5.4, ур. (5.16)]: вектор скорости разворачивается на цель.
    Команда доворачивает Vк к линии визирования (ψ→φ) и нулится, когда они совпали."""
    v_m = np.array([700.0, 0.0, 0.0])
    r_off = np.array([4000.0, 300.0, 0.0])  # цель смещена в +y — доворот вверх по y
    a = pure_pursuit_accel(r_off, v_m, 30.0)
    assert a[1] > 0.0  # команда тянет скорость к линии визирования
    assert abs(float(np.dot(a, v_m))) < 1e-6  # ⊥ скорости
    r_aligned = np.array([5000.0, 0.0, 0.0])  # ψ = φ: скорость уже на цель
    assert np.allclose(pure_pursuit_accel(r_aligned, v_m, 30.0), 0.0, atol=1e-9)


def test_clos_holds_three_point_line() -> None:
    """Метод трёх точек [Гусев §5.1]: ракета держится на линии «пуск—цель».
    При боковом смещении команда гасит перпендикулярное отклонение h (ур. 5.6)."""
    launch = np.zeros(3)
    target_p = np.array([4000.0, 0.0, 0.0])  # луч — ось X
    missile_p = np.array([1500.0, 200.0, 0.0])  # смещена в +y от луча
    v_m = np.array([700.0, 0.0, 0.0])
    r = target_p - missile_p
    a = clos_accel(missile_p, r, v_m, target_p, launch, 30.0)
    deviation = missile_p - np.array([1500.0, 0.0, 0.0])  # вектор отклонения от луча (+y)
    assert float(np.dot(a, deviation)) < 0.0  # команда возвращает ракету на луч
    assert abs(float(np.dot(a, v_m))) < 1e-6  # ⊥ скорости


def test_clos_intercepts_tail_chase() -> None:
    res = collect(_sc(aspect="tail-chase", law="clos", t_max=20), stride=100_000)
    assert np.isfinite(res.miss_m)
    assert res.hit


def test_all_laws_bounded_and_perpendicular() -> None:
    v_m = np.array([700.0, 20.0, 0.0])
    target_p = np.array([4000.0, 300.0, 50.0])
    missile_p = np.array([1500.0, 50.0, 0.0])
    r = target_p - missile_p
    v_t = np.array([-240.0, 0.0, 0.0])
    a_t = np.array([0.0, 0.0, 3.0 * 9.81])
    launch = np.zeros(3)
    laws = [
        ppn_accel(r, v_m, v_t, 4.0, 30.0),
        apn_accel(r, v_m, v_t, a_t, 4.0, 30.0),
        pure_pursuit_accel(r, v_m, 30.0),
        clos_accel(missile_p, r, v_m, target_p, launch, 30.0),
    ]
    for a in laws:
        assert np.isfinite(a).all()
        assert np.linalg.norm(a) <= 30.0 * 9.81 + 1e-6
        assert abs(float(np.dot(a, v_m))) < 1e-6  # команда ⊥ скорости


def test_scenario_law_field_flows_to_engine() -> None:
    res = collect(_sc(law="apn", maneuver="weave", n_target=3.0), stride=100_000)
    assert np.isfinite(res.miss_m)
