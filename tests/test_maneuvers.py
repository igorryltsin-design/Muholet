import numpy as np
import pytest

from navedenie.sim import G, Body, Scenario, target_accel
from navedenie.swarm import fly_from_json, rollout

NEW = ("weave_var", "break", "scissors", "dive", "combo")


def _acc(maneuver: str, t: float, n_target: float = 3.0) -> np.ndarray:
    body = Body(p=np.zeros(3), v=np.array([260.0, 0.0, 0.0]))
    sc = Scenario(maneuver=maneuver, n_target=n_target)  # type: ignore[arg-type]
    return target_accel(body, sc, t)


def test_new_maneuvers_registered_and_rollouts_finish() -> None:
    fly = fly_from_json({"kind": "pn", "pn_n": 4.0})
    for m in NEW:
        res = rollout(Scenario(aspect="head-on", range_m=5000, maneuver=m, n_target=3.0), fly)  # type: ignore[arg-type]
        assert np.isfinite(res["miss_m"]), m
        assert res["miss_m"] > 0, m


def test_break_is_flat_then_hard() -> None:
    assert np.linalg.norm(_acc("break", 1.0)) < 1.0  # первые секунды — прямо
    assert np.linalg.norm(_acc("break", 4.0)) > 2.5 * G  # затем форсированный вираж


def test_scissors_alternates_direction() -> None:
    left = _acc("scissors", 0.5)[1]
    right = _acc("scissors", 2.0)[1]
    assert left * right < 0  # знаки противоположны
    assert abs(left) > 0.5 * G and abs(right) > 0.5 * G


def test_dive_is_vertical() -> None:
    a_up = _acc("dive", 0.75)
    assert abs(a_up[2]) > 0.5 * G  # вертикальная составляющая
    assert abs(a_up[0]) < 1.0  # горизонтальной почти нет


def test_combo_has_both_components() -> None:
    a = _acc("combo", 3.27)  # фаза, где вертикаль и горизонталь одновременно сильны
    assert abs(a[2]) > 0.5 * G  # вертикаль
    assert np.linalg.norm(a[:2]) > 0.5 * G  # горизонталь


def test_weave_var_chirps_faster_than_weave() -> None:
    def sign_flips(m: str) -> int:
        flips = 0
        prev = 0
        for i in range(240):
            t = i * 0.05
            v = _acc(m, t)[1]  # боковая перегрузка (лифт вдоль y)
            if prev != 0 and v != 0 and (v > 0) != (prev > 0):
                flips += 1
            if v != 0:
                prev = v
        return flips

    assert sign_flips("weave_var") > sign_flips("weave")


@pytest.mark.parametrize("m", ["straight", "turn", "weave", *NEW])
def test_maneuver_force_is_perpendicular_and_bounded(m: str) -> None:
    body = Body(p=np.zeros(3), v=np.array([260.0, 0.0, 0.0]))
    sc = Scenario(maneuver=m, n_target=4.0)  # type: ignore[arg-type]
    bound = 4.6 * G if m == "combo" else 4.05 * G  # у combo сумма горизонтальной и вертикальной
    for t in np.linspace(0.1, 11.9, 40):
        a = target_accel(body, sc, float(t))
        assert np.isfinite(a).all()
        assert np.linalg.norm(a) <= bound + 1e-6
        assert abs(float(np.dot(a, body.v))) < 1e-6  # перегрузка ⊥ скорости
