from navedenie.engine import collect
from navedenie.sim import Scenario


def test_pn_head_on_intercepts_stationary_offset():
    sc = Scenario(
        aspect="head-on",
        v_m=700,
        v_t=0.01,
        range_m=4000,
        off_axis_m=250,
        n_max=25,
        pn_n=4,
        mode="pn",
        t_max=20,
        fov_deg=40,
        seeker_delay_s=0,
        kill_radius_m=25,
    )
    result = collect(sc, stride=20)
    assert result.hit or result.miss_m < 50, result.miss_m


def test_pn_head_on_closing_target():
    sc = Scenario(
        aspect="head-on",
        v_m=780,
        v_t=220,
        range_m=6000,
        off_axis_m=180,
        n_max=30,
        pn_n=4,
        mode="pn",
        t_max=20,
        fov_deg=30,
        seeker_delay_s=0,
        kill_radius_m=20,
    )
    result = collect(sc, stride=25)
    assert result.miss_m < 50, (result.miss_m, result.reason)


def test_bio_does_not_crash_and_uses_seeker_only():
    sc = Scenario(mode="bio", t_max=6, range_m=5000, off_axis_m=120, fov_deg=20)
    result = collect(sc, stride=30)
    assert result.frames
    assert "dn" in result.frames[2].circuit
    # Ideal ω_LOS есть в кадре для оператора, но контур хранит literature-stub
    assert result.frames[2].circuit["weights"] in {"не обучен", "обучен", "literature-stub"}
