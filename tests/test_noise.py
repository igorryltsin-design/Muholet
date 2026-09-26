import numpy as np

from navedenie.engine import collect
from navedenie.seeker import observe
from navedenie.sim import Scenario
from navedenie.swarm import fly_from_json, rollout


def _sc(**kw) -> Scenario:
    base = dict(aspect="head-on", v_m=780, v_t=240, range_m=6000, off_axis_m=200,
                n_max=30, t_max=14, dt=0.02, mode="pn", pn_n=4, seeker_delay_s=0)
    base.update(kw)
    return Scenario(**base)


def _obs(az=0.1, el=0.05, lock=True):
    return {"az": az, "el": el, "size": 0.1, "size_dot": 0.02, "lock": lock}


def test_zero_noise_is_deterministic() -> None:
    a = collect(_sc(), stride=100_000).miss_m
    b = collect(_sc(), stride=100_000).miss_m
    assert abs(a - b) < 1e-9


def test_observe_noise_perturbs_readings() -> None:
    import numpy as np

    from navedenie.seeker import FOVEA_K

    r = np.array([3000.0, 200.0, 0.0])
    v = np.array([700.0, 0.0, 0.0])
    half_fov = np.deg2rad(12.0) / 2
    clean = observe(r, v, None, half_fov=half_fov, dt_s=0.02)
    noisy = observe(r, v, None, half_fov=half_fov, dt_s=0.02, noise_az=0.05, noise_range=50.0, rng=np.random.default_rng(5))
    # шум теперь в самом изображении (до декодирования) — пеленг тоже гуляет
    assert abs(noisy["az"] - clean["az"]) > 1e-6
    assert abs(noisy["range"] - clean["range"]) > 1.0
    # детерминизм: то же rng-зерно — то же показание
    again = observe(r, v, None, half_fov=half_fov, dt_s=0.02, noise_az=0.05, noise_range=50.0, rng=np.random.default_rng(5))
    assert again["az"] == noisy["az"] and again["range"] == noisy["range"]
    _ = FOVEA_K


def test_lock_drop_forces_search() -> None:
    import numpy as np

    r = np.array([3000.0, 200.0, 0.0])
    v = np.array([700.0, 0.0, 0.0])
    half_fov = np.deg2rad(12.0) / 2
    always_drop = [observe(r, v, None, half_fov=half_fov, lock_drop_p=1.0, rng=np.random.default_rng(i)) for i in range(5)]
    assert not any(o["lock"] for o in always_drop)
    never_drop = observe(r, v, None, half_fov=half_fov, lock_drop_p=0.0)
    assert never_drop["lock"]


def test_noise_changes_trajectory_but_seed_repeats() -> None:
    sc = _sc(noise_az_deg=4.0, noise_range_m=100.0)
    a = collect(sc, stride=100_000).miss_m
    b = collect(sc, stride=100_000).miss_m
    assert abs(a - b) < 1e-9  # детерминизм при том же seed
    # эталонная ПН управляется по точной геометрии и НЕ зависит от шумов пеленга
    assert a == collect(_sc(), stride=100_000).miss_m
    # а вот биоконтур (читает кадр) — зависит
    bio_clean = collect(_sc(mode="bio"), stride=100_000).miss_m
    bio_noisy = collect(_sc(mode="bio", noise_az_deg=5.0), stride=100_000).miss_m
    assert bio_noisy != bio_clean


def test_lock_drop_kills_bio_guidance_not_pn() -> None:
    bio = collect(_sc(mode="bio", lock_drop_p=1.0), stride=100_000)
    pn = collect(_sc(mode="pn", lock_drop_p=1.0), stride=100_000)
    assert bio.miss_m > 200.0  # био без единого кадра захвата не рулит
    assert pn.hit  # ПН от захвата не зависит


def test_swarm_rollout_noise_changes_bio_path() -> None:
    fly = fly_from_json({"kind": "bio", "w": [0.5, 1.8, 0, 0.35, 0.25, 0, 0.4, 0, 1.8, 0, 0.35, 0, 0, 0.25, 0.4, 0], "gain": 1.15})
    clean = rollout(_sc(), fly)
    noisy = rollout(_sc(noise_az_deg=6.0, noise_range_m=200.0), fly)
    assert noisy["miss_m"] != clean["miss_m"]
    assert any(p["az"] != 0.0 for p in noisy["tel"])


def test_jitter_changes_bio_flight() -> None:
    fly = fly_from_json({"kind": "bio", "w": [0.5, 1.8, 0, 0.35, 0.25, 0, 0.4, 0, 1.8, 0, 0.35, 0, 0, 0.25, 0.4, 0], "gain": 1.15})
    plain = rollout(_sc(mode="bio"), fly)
    jitter = rollout(_sc(mode="bio", noise_az_deg=2.0, seeker_jitter_s=0.06), fly)
    assert jitter["miss_m"] != plain["miss_m"]
