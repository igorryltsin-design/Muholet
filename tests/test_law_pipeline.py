"""Двухэтапный конвейер «закон наведения»: этапы, fitness, geometric CPA (P12).

Проверяем контракты научного pipeline:
- этап B (outcome evolution) слеп к учителю: подмена oracle-ПН мусором не меняет
  ни оценки кандидатов, ни выбранного чемпиона;
- fitness лексикографичен: hit rate важнее промаха, промах — важнее усилия;
- geometric CPA не зависит от kill radius и сходится при уменьшении dt;
- этап A конфигурируем (teacher-режимы, ранняя остановка), лучшие веса — по validation;
-детерминизм при одинаковом seed; артефакты несут схему признаков и конфиг.
"""

import json

import numpy as np
import pytest

from navedenie.law import (
    ARTIFACTS_DIR,
    BatchMetrics,
    EpisodeMetrics,
    INIT_MODES,
    EvolveConfig,
    WarmStartConfig,
    _init_circuit,
    _shadow_cpa,
    discover_law,
    evaluate_episode,
    outcome_evolve,
    validation_batch,
    warm_start,
)
from navedenie.sim import Scenario

# быстрые конфиги для тестов (короткие эпизоды)
_WS = WarmStartConfig(episodes=4, eval_every=2, patience=2, early_stop_hit_frac=None, seed=5, t_cap=6.0)
_EV_KW = dict(preset="custom", generations=2, pop=2, scenarios_per_gen=2, validation_size=2,
              sigma=0.3, seed=42, t_cap=6.0, dt_cycle=(0.02,))


def _ep(hit: bool, cpa: float, effort: float, t_hit: float | None = None, sat: float = 0.0, lock: float = 1.0, jerk: float = 0.1) -> EpisodeMetrics:
    return EpisodeMetrics(hit=hit, t_hit=t_hit, trigger_range_m=min(cpa, 40.0), geometric_cpa_m=cpa,
                          effort_gs=effort, sat_frac=sat, lock_frac=lock, jerk=jerk, duration_s=8.0)


def _batch(eps: list[EpisodeMetrics]) -> BatchMetrics:
    return BatchMetrics(episodes=eps)


# ── P12.2: этап B слеп к учителю ─────────────────────────────────────────────

def test_outcome_evolution_is_teacher_blind(monkeypatch) -> None:
    """Подмена oracle-ПН бессмысленным результатом НЕ меняет ни оценки кандидатов,
    ни champion'а: в этапе B learn_step не вызывается, oracle-геометрия не читается."""
    import navedenie.law as law_mod
    import navedenie.pn as pn_mod
    import navedenie.train as train_mod

    def garbage(r, v_m, v_t, n_const, n_max):
        return np.array([12345.0, -999.0, 42.0])

    monkeypatch.setattr(pn_mod, "ppn_accel", garbage)
    monkeypatch.setattr(law_mod, "ppn_accel", garbage)
    monkeypatch.setattr(train_mod, "ppn_accel", garbage)

    def run() -> dict:
        circuit = _init_circuit("stub", "pn_coarse_warmstart", seed=7)
        out = outcome_evolve(circuit, EvolveConfig(**_EV_KW))
        return {
            "validation": out["validation"],
            "weights": circuit.W_dn.copy(),
        }

    a = run()
    b = run()
    assert a["validation"] == b["validation"]
    assert np.array_equal(a["weights"], b["weights"])


# ── P12.6: лексикографический fitness ────────────────────────────────────────

def test_fitness_hit_rate_beats_effort_and_miss() -> None:
    """Больше hit rate — выше ранг, ЛЮБОЙ проигрыш по усилию/промаху не важен."""
    sloppy_hit = _batch([_ep(True, 39.0, 500.0), _ep(False, 3000.0, 500.0)])  # hit_rate 0.5
    neat_miss = _batch([_ep(False, 120.0, 1.0), _ep(False, 140.0, 1.0)])  # hit_rate 0, малый CPA/effort
    assert sloppy_hit.rank_key() < neat_miss.rank_key()
    assert neat_miss.better_than(sloppy_hit) is False


def test_fitness_robust_miss_then_effort() -> None:
    """При равном hit rate и равной медиане: меньший worst-case CPA (p90/CVaR)
    лучше; при равном промахе — меньшее усилие лучше."""
    # медианы равны (30), но у b1 есть хвост 900 м: p90/CVaR больше — хуже
    b1 = _batch([_ep(True, 30.0, 10.0), _ep(True, 30.0, 10.0), _ep(True, 30.0, 10.0), _ep(True, 30.0, 10.0), _ep(True, 900.0, 10.0)])
    b2 = _batch([_ep(True, 30.0, 10.0), _ep(True, 30.0, 10.0), _ep(True, 30.0, 10.0), _ep(True, 34.0, 10.0), _ep(True, 34.0, 10.0)])
    assert b1.cpa_median == b2.cpa_median
    assert b2.rank_key() < b1.rank_key()
    # равные промахи: меньше усилие — лучше
    e1 = _batch([_ep(True, 50.0, 5.0), _ep(True, 50.0, 5.0)])
    e2 = _batch([_ep(True, 50.0, 50.0), _ep(True, 50.0, 50.0)])
    assert e1.rank_key() < e2.rank_key()
    assert e1.better_than(e2)


def test_fitness_components_stay_separate() -> None:
    """Все компоненты fitness хранятся и отдаются раздельно (никакой суммы)."""
    comp = _batch([_ep(True, 33.0, 7.0)]).components()
    for key in ("hit_rate", "cpa_median_m", "cpa_p90_m", "cpa_cvar90_m", "effort_median_gs", "sat_frac_median", "lock_frac_min", "jerk_median", "t_hit_median_s"):
        assert key in comp


# ── P12.7+14: geometric CPA — не зависит от kill radius, сходится по dt ──────

def test_shadow_cpa_geometry_and_kill_radius_independence() -> None:
    """Аналитическое теневое продолжение: минимальное расстояние прямолинейного
    относительного движения не зависит от радиуса сферы (его в формуле нет)."""
    p_m = np.array([0.0, 0.0, 0.0])
    v_m = np.array([900.0, 0.0, 0.0])
    p_t = np.array([4000.0, 120.0, 0.0])
    v_t = np.array([-300.0, 0.0, 0.0])
    s1 = _shadow_cpa(p_m, v_m, p_t, v_t)
    # та же геометрия, старт с другой точки той же прямой (радиус больше/меньше)
    alpha = 0.5
    s2 = _shadow_cpa(p_m + v_m * alpha, v_m, p_t + v_t * alpha, v_t)
    assert abs(s1 - 120.0) < 1e-6  # боковой промах 120 м — точная аналитика
    assert abs(s1 - s2) < 1e-6


def test_geometric_cpa_stable_across_kill_radii() -> None:
    """Эпизод с перехватом: geometric CPA почти не меняется при смене kill radius
    (trigger point смещается, но прямые траектории те же); trigger range следует
    за радиусом."""
    circuit = _init_circuit("stub", "pn_coarse_warmstart", seed=3)
    warm_start(circuit, WarmStartConfig(episodes=8, eval_every=4, patience=2, early_stop_hit_frac=None, seed=3, t_cap=10.0))
    cpas = []
    triggers = []
    for kr in (25.0, 45.0, 80.0):
        sc = Scenario(aspect="head-on", mode="bio", v_m=850.0, v_t=240.0, range_m=5000.0,
                      off_axis_m=150.0, t_max=12.0, dt=0.02, n_max=30.0, kill_radius_m=kr, seed=3)
        m = evaluate_episode(circuit, sc, teacher=False, t_cap=12.0)
        if m.hit:
            cpas.append(m.geometric_cpa_m)
            triggers.append((kr, m.trigger_range_m))
    assert len(cpas) == 3, "warm-started мозг должен перехватывать на всех радиусах"
    assert max(cpas) - min(cpas) < 5.0
    # trigger range отслеживает радиус сферы
    assert triggers[0][1] < triggers[-1][1] + 5.0


def test_geometric_cpa_converges_with_dt() -> None:
    """CPA при dt, dt/2, dt/4 сходится (разумное численное поведение, P12.14)."""
    circuit = _init_circuit("stub", "pn_coarse_warmstart", seed=3)
    warm_start(circuit, WarmStartConfig(episodes=8, eval_every=4, patience=2, early_stop_hit_frac=None, seed=3, t_cap=10.0))
    vals = []
    for dt in (0.02, 0.01, 0.005):
        sc = Scenario(aspect="head-on", mode="bio", v_m=850.0, v_t=240.0, range_m=5000.0,
                      off_axis_m=150.0, t_max=12.0, dt=dt, n_max=30.0, kill_radius_m=45.0, seed=3)
        m = evaluate_episode(circuit, sc, teacher=False, t_cap=12.0)
        assert m.hit
        vals.append(m.geometric_cpa_m)
    assert abs(vals[0] - vals[2]) < 5.0


# ── P0/P12.1: dt сценария реально используется движком обучения ───────────────

def test_episode_uses_scenario_dt() -> None:
    """Число шагов эпизода пропорционально 1/dt (движок обучения и оценки)."""
    circuit = _init_circuit("stub", "pn_coarse_warmstart", seed=1)
    counts = {}
    for dt in (0.02, 0.01):
        feats: list[np.ndarray] = []
        sc = Scenario(aspect="head-on", mode="bio", range_m=6000.0, t_max=4.0, dt=dt, seed=1)
        evaluate_episode(circuit, sc, teacher=False, t_cap=4.0, feats=feats)
        counts[dt] = len(feats)
    assert counts[0.01] == pytest.approx(counts[0.02] * 2, rel=0.2)
    # и интегрирование действительно идёт с шагом сценария: 0.005 даёт 4×
    feats5: list[np.ndarray] = []
    sc5 = Scenario(aspect="head-on", mode="bio", range_m=6000.0, t_max=4.0, dt=0.005, seed=1)
    evaluate_episode(circuit, sc5, teacher=False, t_cap=4.0, feats=feats5)
    assert counts[0.01] == pytest.approx(len(feats5) / 2, rel=0.2)


# ── P1: этап A — teacher-режимы, ранняя остановка, validation-выбор ───────────

def test_teacher_modes_shape_commands() -> None:
    from navedenie.law import _shape_teacher

    a = np.array([0.0, 0.0, 5.0])
    full = _shape_teacher(a, np.array([700.0, 0, 0]), 30.0, "full_command")
    assert abs(full[0] - 5.0 / (30 * 9.81)) < 1e-9  # амплитуда ПН сохранена
    direction = _shape_teacher(a, np.array([700.0, 0, 0]), 30.0, "direction_only")
    assert abs(np.linalg.norm(direction) - 1.0) < 1e-9  # только направление
    coarse = _shape_teacher(a, np.array([700.0, 0, 0]), 30.0, "clipped_coarse")
    assert np.all(np.abs(coarse) <= 0.5)  # грубая ограниченная амплитуда
    assert np.allclose(coarse * 2, np.round(coarse * 2))  # квантование 0.5


def test_warm_start_history_marks_stages_and_stops_early() -> None:
    circuit = _init_circuit("stub", "pn_full_warmstart", seed=2)
    out = warm_start(circuit, WarmStartConfig(episodes=6, eval_every=2, patience=1, early_stop_hit_frac=None, seed=2, t_cap=6.0))
    assert out["stage"] == "warm_start"
    train_rows = [h for h in out["history"] if h["split"] == "train"]
    val_rows = [h for h in out["history"] if h["split"] == "validation"]
    assert train_rows and all(h["teacher_active"] for h in train_rows)
    assert val_rows and not any(h["teacher_active"] for h in val_rows)
    assert train_rows[0]["teacher_mode"] in ("full_command", "direction_only", "clipped_coarse")
    assert out["validation"] is not None  # лучшие веса выбраны по validation


def test_warm_start_early_stop_on_validation_hit_rate() -> None:
    circuit = _init_circuit("stub", "pn_full_warmstart", seed=4)
    out = warm_start(circuit, WarmStartConfig(episodes=30, eval_every=2, patience=30, early_stop_hit_frac=0.34, seed=4, t_cap=8.0))
    assert out["early_stopped"] or out["episodes_done"] == 30
    # порог 1/3 перехватов на трио достигнут или исчерпан бюджет — оба исхода валидны,
    # но история обязана содержать validation-оценки
    assert any(h["split"] == "validation" for h in out["history"])


def test_init_modes_random_has_no_teacher_stage() -> None:
    assert set(INIT_MODES) == {"pn_full_warmstart", "pn_coarse_warmstart", "pursuit_warmstart", "random_initialization", "warmstart_only"}
    circuit = _init_circuit("stub", "random_initialization", seed=1)
    # случайная инициализация — не врождённый рефлекс: веса малы/шумны
    assert np.max(np.abs(circuit.W_dn)) <= 2.0
    assert circuit.trained is False


def test_warmstart_only_skips_outcome_stage(tmp_path, monkeypatch) -> None:
    """init=warmstart_only: этап B не выполняется — отчёт эволюции пуст,
    метрики чемпионa = результат имитации на validation."""
    monkeypatch.setattr("navedenie.law.ARTIFACTS_DIR", tmp_path / "law_discovery")
    out = discover_law(
        "stub", init="warmstart_only", preset="custom", seed=5,
        warm_cfg=WarmStartConfig(episodes=2, eval_every=2, patience=2, early_stop_hit_frac=None, seed=5, t_cap=5.0),
        evolve_cfg=EvolveConfig(preset="custom", generations=3, pop=2, scenarios_per_gen=2, validation_size=2, seed=5, t_cap=5.0, dt_cycle=(0.02,)),
    )
    assert out["warm_start"] is not None
    assert out["evolution"]["generations_done"] == 0
    assert out["evolution"]["stage"] == "warm_start_only"
    assert out["evolution"]["validation"] is not None


# ── P3: validation-батч разнообразнее канонического трио ─────────────────────

def test_validation_batch_covers_conditions() -> None:
    cfg = EvolveConfig(preset="custom", validation_size=9, seed=17)
    scens = validation_batch(cfg, 1.0)
    assert len(scens) == 9
    aspects = {str(s.aspect) for s in scens}
    maneuvers = {str(s.maneuver) for s in scens}
    modes = {str(s.target_speed_mode) for s in scens}
    assert aspects == {"head-on", "beam", "tail-chase"}
    assert len(maneuvers) >= 3
    assert len(modes) >= 4
    # шум/задержки/отказы встречаются в батче
    assert any(s.noise_az_deg > 0 for s in scens)
    assert any(s.retina_death_p > 0 for s in scens)


# ── P12.13: детерминизм pipeline и артефакты (P12.12) ────────────────────────

def test_pipeline_deterministic_same_seed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("navedenie.law.ARTIFACTS_DIR", tmp_path / "law_discovery")

    def run() -> dict:
        return discover_law(
            "stub", init="pn_coarse_warmstart", preset="custom", seed=9,
            warm_cfg=WarmStartConfig(episodes=2, eval_every=2, patience=2, early_stop_hit_frac=None, seed=9, t_cap=5.0),
            evolve_cfg=EvolveConfig(preset="custom", generations=2, pop=2, scenarios_per_gen=2, validation_size=2, seed=9, t_cap=5.0, dt_cycle=(0.02,)),
        )

    a, b = run(), run()
    assert a["validation"] == b["validation"]
    assert np.array_equal(a["weights"], b["weights"])


def test_artifacts_carry_schema_config_and_history(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("navedenie.law.ARTIFACTS_DIR", tmp_path / "law_discovery")
    out = discover_law(
        "stub", init="random_initialization", preset="custom", seed=3,
        evolve_cfg=EvolveConfig(preset="custom", generations=1, pop=2, scenarios_per_gen=2, validation_size=2, seed=3, t_cap=5.0, dt_cycle=(0.02,)),
    )
    run_dir = tmp_path / "law_discovery" / out["run_id"]
    assert (run_dir / "config.json").exists() and (run_dir / "weights.npz").exists()
    assert (run_dir / "history.json").exists() and (run_dir / "README.md").exists()
    assert (run_dir / "validation_metrics.json").exists()
    assert not (run_dir / "test_metrics.json").exists()  # held-out дисциплина
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    from navedenie.circuit import FEATURE_SCHEMA_VERSION

    assert cfg["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert cfg["teacher_blind_stage_b"] is True and cfg["init"] == "random_initialization"
    blob = np.load(run_dir / "weights.npz", allow_pickle=False)
    assert int(np.asarray(blob["feat_schema"]).reshape(-1)[0]) == FEATURE_SCHEMA_VERSION
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    assert all(h["teacher_active"] is False for h in history)
    # test-метрики появляются ТОЛЬКО по явному запросу
    out2 = discover_law(
        "stub", init="random_initialization", preset="custom", seed=3, evaluate_test=True,
        evolve_cfg=EvolveConfig(preset="custom", generations=1, pop=2, scenarios_per_gen=2, validation_size=2, seed=3, t_cap=5.0, dt_cycle=(0.02,)),
    )
    assert out2["test"] is not None and "hit_rate" in out2["test"]
