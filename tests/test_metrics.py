"""Метрики призрака-ПН: время наведения, отклонение от оптимальной траектории."""

import numpy as np

from navedenie.circuit import FlyCircuit
from navedenie.engine import collect
from navedenie.sim import Scenario
from navedenie.train import _rollout_update, train_step


def test_pn_ghost_deviation_is_zero() -> None:
    """Чистая ПН летит вдоль призрака: отклонение ~0, времена совпадают."""
    sc = Scenario(aspect="head-on", mode="pn", law="pn", range_m=6000, t_max=12, dt=0.02)
    res = collect(sc, stride=1000)
    assert res.hit
    assert res.t_guide is not None and abs(res.t_guide - res.t_hit) < 1e-9
    assert res.ref_dev_m is not None and res.ref_dev_m < 1.0
    assert res.t_ref is not None and abs(res.t_ref - res.t_guide) < 0.5


def test_h0_equals_hcv_in_kinematic_but_differs_in_physics() -> None:
    """§6.2: h₀ (промах при нулевой команде) и h_cv (экстраполяция при неизменных
    скоростях) совпадают в кинематической модели и РАЗЛИЧАЮТСЯ в физической,
    где свободный полёт искривляется тяжестью. legacy terminal_zem_m ≡ h_cv_m."""
    kin = collect(Scenario(aspect="head-on", mode="pn", law="pn", v_m=700, v_t=250,
                           range_m=6000, off_axis_m=200, t_max=12, dt=0.01), stride=1000)
    assert kin.terminal_zem_m == kin.h_cv_m          # legacy-alias стабилен
    assert abs((kin.h0_m or 0) - (kin.h_cv_m or 0)) < 1e-6  # кинематика: h₀ ≡ h_cv

    phys = collect(Scenario(aspect="head-on", mode="pn", law="pn", v_m=700, v_t=250,
                            range_m=6000, off_axis_m=200, t_max=12, dt=0.01,
                            model="point_mass_3dof"), stride=1000)
    assert phys.h0_m is not None and phys.h_cv_m is not None
    assert abs(phys.h0_m - phys.h_cv_m) > 0.5        # тяжесть гнёт free-полёт → расходятся


def test_bio_deviates_and_t_guide_none_on_miss() -> None:
    """Необученная схема мимо БЧ 1 м: t_guide пуст, но призрак отклоняется заметно."""
    sc = Scenario(aspect="head-on", mode="bio", range_m=6000, t_max=8, dt=0.02, kill_radius_m=1.0, seed=5)
    res = collect(sc, stride=10_000, circuit=FlyCircuit(kind="stub"))
    assert not res.hit
    assert res.t_guide is None
    assert res.ref_dev_m is not None and res.ref_dev_m > 1.0
    assert res.t_end is not None  # метрики v2: t_hit — строго время входа в сферу; конец — t_end


def test_rollout_episode_metrics() -> None:
    """Эпизод обучения возвращает полный набор метрик, включая призрак."""
    circuit = FlyCircuit(kind="stub")
    sc = Scenario(aspect="beam", mode="bio", t_max=8, dt=0.02, seed=3, kill_radius_m=1.0)
    out = _rollout_update(circuit, sc, lr=0.0)
    assert out["hit"] is False and out["t_guide"] is None
    assert out["ref_dev"] > 1.0
    assert "t_ref" in out


def test_train_step_returns_metrics() -> None:
    d = train_step("stub", ep=0)
    assert {"ep", "miss", "hit", "t_guide", "ref_dev", "t_ref", "w", "tel"} <= set(d)


def test_compare_rows_carry_new_metrics() -> None:
    from navedenie.app import BrainCompareIn, ScenarioIn, brain_compare

    body = BrainCompareIn(scenario=ScenarioIn(range_m=6000, t_max=10), kinds=["stub"], laws=[])
    out = brain_compare(body)
    labels = [r["kind"] for r in out["results"]]
    assert "pn" in labels and "stub" in labels
    for row in out["results"]:
        for key in ("t_guide", "ref_dev_m", "t_ref", "t_end", "n_int"):
            assert key in row
    pn_row = next(r for r in out["results"] if r["kind"] == "pn")
    if pn_row["ref_dev_m"] is not None:
        assert pn_row["ref_dev_m"] < 1.0


def test_full_brain_npz_roundtrip(tmp_path) -> None:
    """Полный мозг сохраняется целиком: W_dn, W_k, W_pool, gain, tau."""
    c = FlyCircuit(kind="full", gain=1.3, tau_s=0.03)
    c.trained = True
    w_dn, w_k, w_pool = c.W_dn.copy(), c.W_k.copy(), c.W_pool.copy()
    path = c.save(path=tmp_path / "weights_full.npz")

    c2 = FlyCircuit(kind="full", gain=1.0, tau_s=0.025)
    assert c2.load(path=path)
    assert np.array_equal(c2.W_dn, w_dn)
    assert np.array_equal(c2.W_k, w_k)
    assert np.array_equal(c2.W_pool, w_pool)
    assert c2.gain == 1.3
    assert c2.tau == 0.03
    assert c2.trained


def test_brain_export_has_tau_and_saved_at() -> None:
    from navedenie.app import brain_export

    exported = brain_export("stub")
    assert "tau" in exported and "seed" in exported and "saved_at" in exported
    assert exported["tau"] > 0


def test_swarm_gen_stats_have_worst_and_hit_rate() -> None:
    from navedenie.app import ScenarioIn, SwarmFlyIn, SwarmGenIn, swarm_gen

    body = SwarmGenIn(
        scenario=ScenarioIn(range_m=5000, t_max=8),
        scenarios=[],
        population=[SwarmFlyIn(kind="pn", pn_n=4.0), SwarmFlyIn(kind="pn", pn_n=3.0)],
        seed=7,
    )
    out = swarm_gen(body)
    stats = out["stats"]
    assert "worst" in stats and "hit_rate" in stats
    assert 0.0 <= stats["hit_rate"] <= 1.0
    assert stats["worst"] >= stats["best"]


def test_experiments_persist_roundtrip(tmp_path, monkeypatch) -> None:
    """Эксперименты пишутся в data/experiments и читаются обратно."""
    from navedenie import app as app_mod

    monkeypatch.setattr(app_mod, "_EXP_DIR", tmp_path)
    from fastapi import HTTPException

    saved = app_mod.experiment_save(app_mod.ExperimentIn(name="рой-1", payload={"generations": [1, 2], "champion": 25}))
    assert saved["ok"] is True
    listed = app_mod.experiments_list()
    assert listed["experiments"][0]["name"] == "рой-1"
    loaded = app_mod.experiment_get("рой-1")
    assert loaded["champion"] == 25
    try:
        app_mod.experiment_get("../secret")
        raise AssertionError("ожидался 422")
    except HTTPException as exc:
        assert exc.status_code == 422
    try:
        app_mod.experiment_get("нет-такого")
        raise AssertionError("ожидался 404")
    except HTTPException as exc:
        assert exc.status_code == 404

    # удаление: снимок уходит из архива, повторное удаление и плохое имя — ошибки
    deleted = app_mod.experiment_delete("рой-1")
    assert deleted["ok"] is True
    assert not (tmp_path / "рой-1.json").exists()
    assert app_mod.experiments_list()["experiments"] == []
    try:
        app_mod.experiment_delete("рой-1")
        raise AssertionError("ожидался 404")
    except HTTPException as exc:
        assert exc.status_code == 404
    try:
        app_mod.experiment_delete("../secret")
        raise AssertionError("ожидался 422")
    except HTTPException as exc:
        assert exc.status_code == 422
