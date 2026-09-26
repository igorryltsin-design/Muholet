"""Воспроизводимый эксперимент «закон наведения» (P4, P5, P9, P13 промта).

Протокол:
  1. P5  — v2-мозги (stub, 3 seed): warm start (coarse/full) + outcome evolution,
           retraining-абляции (без theta / без rho / без обоих), warmstart_only,
           random → outcome. Отдельно коннектом (компактный конфиг).
  2. P4  — одинаковый outcome evolution из 4 инициализаций × 3 seed.
  3. P9  — inference-абляции чемпиона + фотометрическая устойчивость.
  4. P13 — held-out таблица: 3 v_m × 3 v_t × 3 аспекта × 3 профиля скорости,
           81 эпизод на политику; oracle/сенсорные ПН, scheduled, BIO (этап A,
           этап B), deployable-суррогаты H1/H2/H5/полином, абляции.

Честность: чемпион — по validation; test-таблица строится один раз в самом
конце; ПН-сходство в fitness не входит; артефакты — artifacts/law_discovery/.

Запуск: .venv/bin/python tools/law_experiment.py [--fast]
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from navedenie.diagnostics import (  # noqa: E402
    NarxLaw,
    PolyReadout,
    SensorConstantPN,
    SensorScheduledPN,
    episode_trace,
    fit_hypotheses,
    neff_decomposition,
    photometric_robustness,
)
from navedenie.circuit import FlyCircuit  # noqa: E402
from navedenie.engine import collect  # noqa: E402
from navedenie.formula import _flight_fit  # noqa: E402
from navedenie.law import (  # noqa: E402
    EvolveConfig,
    WarmStartConfig,
    _init_circuit,
    discover_law,
    evaluate_batch,
    evaluate_episode,
    _shadow_cpa,
)
from navedenie.science import feature_ablation, get_live_circuit  # noqa: E402
from navedenie.sim import Scenario  # noqa: E402

OUT = REPO / "artifacts" / "law_experiment"
STAMP = time.strftime("%Y%m%d_%H%M%S")
FAST = "--fast" in sys.argv

WARM = WarmStartConfig(episodes=8 if FAST else 16, eval_every=4, patience=6, early_stop_hit_frac=0.95, seed=11, t_cap=14.0)
# средний исследовательский бюджет сессии (полный research-пресет: pop 32, gen 40,
# 10 сценариев/поколение — доступен в law.PRESETS для отдельных запусков)
EVO = dict(generations=5 if FAST else 12, pop=6 if FAST else 10, scenarios_per_gen=3 if FAST else 4,
           validation_size=5 if FAST else 6, sigma=0.3, t_cap=14.0)
CONN_EVO = dict(generations=4 if FAST else 8, pop=4 if FAST else 6, scenarios_per_gen=3,
                validation_size=4 if FAST else 6, sigma=0.3, t_cap=12.0, dt_cycle=(0.02, 0.01))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def evo_cfg(**kw) -> EvolveConfig:
    base = dict(EVO)
    base.update(kw)
    return EvolveConfig(preset="custom", **base)  # type: ignore[arg-type]


def warm_cfg_for(init: str, seed: int) -> WarmStartConfig:
    """Бюджет этапа A из WARM, но режим teacher — по инициализации:
    full_command / direction_only / pursuit (иначе init не имеет смысла)."""
    return replace(
        WARM,
        episodes=8 if init == "pursuit_warmstart" and FAST else WARM.episodes,
        teacher_mode=("full_command" if init == "pn_full_warmstart" else "direction_only"),
        teacher_kind=("pursuit" if init == "pursuit_warmstart" else "pn"),
        seed=seed + 11,
    )


# ── held-out сетка P13 ────────────────────────────────────────────────────────

VM_GRID = (650.0, 780.0, 950.0)
VT_GRID = (150.0, 260.0, 360.0)
ASPECTS = ("head-on", "beam", "tail-chase")
MODES = ("constant", "accelerate", "pulse")


def held_out_cell(i: int, vm: float, vt: float, aspect: str, mode: str) -> Scenario:
    return Scenario(
        aspect=aspect,  # type: ignore[arg-type]
        v_m=vm, v_t=vt, range_m=8000.0, off_axis_m=420.0,
        n_max=30.0, n_target=0.0, maneuver="straight",
        pn_n=4.0, dt=0.02, t_max=16.0, fov_deg=14.0, kill_radius_m=45.0,
        mode="bio",
        target_speed_mode=mode,  # type: ignore[arg-type]
        target_longitudinal_g=1.0,
        target_speed_min=float(max(120.0, vt * 0.55)),
        target_speed_max=float(min(520.0, vt * 1.5)),
        target_speed_period_s=6.0,
        target_speed_phase=1.5,
        circuit_gain=1.0,
        seed=9000 + i,
    )


def held_out_grid() -> list[Scenario]:
    cells = []
    i = 0
    for vm in VM_GRID:
        for vt in VT_GRID:
            for aspect in ASPECTS:
                for mode in MODES:
                    cells.append(held_out_cell(i, vm, vt, aspect, mode))
                    i += 1
    return cells


def wilson(hit: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = hit / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# ── PN-законы через движок + shadow CPA (таблица P13) ─────────────────────────


def pn_law_row(law: str, grid: list[Scenario]) -> dict:
    hits = 0
    cpas, triggers, efforts, sats, locks = [], [], [], [], []
    for sc in grid:
        from dataclasses import replace as _rep

        sc_l = _rep(sc, mode="pn", law=law)
        res = collect(sc_l, stride=100_000)
        hits += int(res.hit)
        cpa = float(res.cpa_m)
        if res.hit and res.frames:
            # для попавших тоже честный geometric CPA: теневое продолжение от кадра попадания
            last = res.frames[-1]
            cpa = _shadow_cpa(last.missile, last.missile_v, last.target, last.target_v)
        cpas.append(cpa)
        triggers.append(min(cpa, sc.kill_radius_m) if res.hit else cpa)
        efforts.append(float(res.n_int))
        sats.append(float(res.sat_frac))
        locks.append(float(res.lock_fraction))
    lo, hi = wilson(hits, len(grid))
    return {
        "policy": law,
        "hit_rate": round(hits / len(grid), 4),
        "hit_ci95": [round(lo, 3), round(hi, 3)],
        "cpa_median_m": round(float(np.median(cpas)), 1),
        "cpa_p90_m": round(float(np.percentile(cpas, 90)), 1),
        "effort_median_gs": round(float(np.median(efforts)), 1),
        "sat_median": round(float(np.median(sats)), 3),
        "lock_median": round(float(np.median(locks)), 3),
    }


def bio_row(policy: str, circuit, grid: list[Scenario]) -> dict:
    bm = evaluate_batch(circuit, grid, teacher=False, t_cap=16.0, scenario_split="test")
    comp = bm.components()
    hits = int(round(comp["hit_rate"] * len(grid)))
    lo, hi = wilson(hits, len(grid))
    return {
        "policy": policy,
        "hit_rate": comp["hit_rate"],
        "hit_ci95": [round(lo, 3), round(hi, 3)],
        "cpa_median_m": comp["cpa_median_m"],
        "cpa_p90_m": comp["cpa_p90_m"],
        "effort_median_gs": comp["effort_median_gs"],
        "sat_median": comp["sat_frac_median"],
        "lock_median": comp["lock_frac_min"],
    }


def controller_row(policy: str, law, grid: list[Scenario]) -> dict:
    def ctrl(obs, v_m, n_max, _l=law):
        return _l.command(obs, v_m, n_max)

    bm = evaluate_batch(None, grid, teacher=False, t_cap=16.0, scenario_split="test", controller=ctrl)
    comp = bm.components()
    hits = int(round(comp["hit_rate"] * len(grid)))
    lo, hi = wilson(hits, len(grid))
    return {
        "policy": policy,
        "hit_rate": comp["hit_rate"],
        "hit_ci95": [round(lo, 3), round(hi, 3)],
        "cpa_median_m": comp["cpa_median_m"],
        "cpa_p90_m": comp["cpa_p90_m"],
        "effort_median_gs": comp["effort_median_gs"],
        "sat_median": comp["sat_frac_median"],
        "lock_median": comp["lock_frac_min"],
    }


# ── основной протокол ─────────────────────────────────────────────────────────


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"stamp": STAMP, "fast": FAST}

    # ── P5: v2-мозги на схеме ──
    log("P5: обучаю v2-мозги (stub, 3 seed)")
    v2_runs: list[dict] = []
    configs = [
        ("v2_coarse_full_pipeline", "pn_coarse_warmstart", ()),
        ("v2_fullcmd_pipeline", "pn_full_warmstart", ()),
        ("v2_no_theta_retrain", "pn_coarse_warmstart", (4,)),
        ("v2_no_rho_retrain", "pn_coarse_warmstart", (6,)),
        ("v2_no_theta_rho_retrain", "pn_coarse_warmstart", (4, 6)),
        ("warmstart_only", "warmstart_only", ()),
        ("random_outcome", "random_initialization", ()),
    ]
    seeds = (0, 1, 2) if not FAST else (0,)
    champions: dict[str, object] = {}
    for name, init, zeros in configs:
        best = None
        for seed in seeds:
            out = discover_law(
                "stub", init=init, preset="custom", seed=seed,
                warm_cfg=warm_cfg_for(init, seed), evolve_cfg=evo_cfg(seed=seed + 101),
                feat_zero_columns=zeros,
            )
            out.pop("weights", None)
            v2_runs.append({"config": name, "init": init, "seed": seed, "validation": out["validation"], "run_id": out["run_id"]})
            key = (
                -out["validation"]["hit_rate"],
                out["validation"]["cpa_median_m"],
                out["validation"]["cpa_p90_m"],
            )
            if best is None or key < best[0]:
                best = (key, seed, out["run_id"])
        if best is not None:
            champions[name] = {"seed": best[1], "run_id": best[2]}
            log(f"  {name}: лучший по validation seed={best[1]} val={best[2]}")
    report["v2_runs"] = v2_runs
    report["v2_champions"] = champions

    # перезагружаем веса чемпионов из артефактов (не из живого стенда)
    def load_champion(run_id: str):
        c = FlyCircuit(kind="stub")
        blob = np.load(REPO / "artifacts" / "law_discovery" / run_id / "weights.npz", allow_pickle=False)
        c.W_dn = blob["W_dn"]
        c.trained = True
        c.reset()
        return c

    # ── коннектом: компактный полный pipeline ──
    log("P5: коннектом — полный pipeline (компактный конфиг)")
    conn_out = discover_law(
        "connectome", init="pn_coarse_warmstart", preset="custom", seed=0,
        warm_cfg=replace(warm_cfg_for("pn_coarse_warmstart", 0), episodes=8 if FAST else 16, t_cap=12.0),
        evolve_cfg=evo_cfg(seed=101, **CONN_EVO),
    )
    conn_out.pop("weights", None)
    report["connectome_run"] = {"run_id": conn_out["run_id"], "validation": conn_out["validation"]}
    log(f"  connectome val: {conn_out['validation']}")

    # ── P4: инициализации ──
    log("P4: сравнение инициализаций (stub, одинаковый outcome evolution)")
    inits = ["pn_full_warmstart", "pn_coarse_warmstart", "pursuit_warmstart", "random_initialization"]
    init_rows = []
    for init in inits:
        per_seed = []
        for seed in seeds:
            out = discover_law(
                "stub", init=init, preset="custom", seed=seed + 50,
                warm_cfg=warm_cfg_for(init, seed + 50), evolve_cfg=evo_cfg(seed=seed + 50 + 101),
            )
            out.pop("weights", None)
            per_seed.append(out["validation"])
        hit = [v["hit_rate"] for v in per_seed]
        cpa = [v["cpa_median_m"] for v in per_seed]
        init_rows.append({
            "init": init,
            "hit_rate_mean": round(float(np.mean(hit)), 4),
            "hit_rate_seeds": [round(float(x), 3) for x in hit],
            "cpa_median_mean_m": round(float(np.mean(cpa)), 1),
            "cpa_median_seeds": [round(float(x), 1) for x in cpa],
        })
        log(f"  {init}: hit={np.mean(hit):.2f} cpa={np.mean(cpa):.1f} м")
    report["initializations"] = init_rows

    # чемпион всего P5 — по validation среди stub-конфигов (НЕ по test):
    # лучший validation hit_rate, затем CPA
    def _champ_key(n: str) -> tuple:
        v = next(r["validation"] for r in v2_runs if r["config"] == n and r["seed"] == champions[n]["seed"])
        return (-v["hit_rate"], v["cpa_median_m"], v["cpa_p90_m"])

    champ_name = min(list(champions), key=_champ_key)
    log(f"P5: чемпион по validation — {champ_name}")
    champ = load_champion(champions[champ_name]["run_id"])
    report["champion"] = {"config": champ_name, **champions[champ_name]}

    # ── P9: inference-абляции и фотометрия на чемпионе ──
    log("P9: inference-абляции + фотометрия чемпиона")
    abl_grid = [
        held_out_cell(700 + i, 780.0, vt, aspect, "constant")
        for i, (vt, aspect) in enumerate([(150.0, "head-on"), (260.0, "beam"), (360.0, "tail-chase")])
    ]
    from navedenie.science import _apply_rho_ablation, _max_steps
    from navedenie.train import _rollout_update

    # donor-ряды rho с БАЗОВОГО чемпиона (полночастотные) — для честных перестановок
    rho_donors: list[list[float]] = []
    for sc in abl_grid:
        circuit = load_champion(champions[champ_name]["run_id"])
        trace: list[float] = []
        _rollout_update(circuit, sc, lr=0.0, rho_trace=trace)
        rho_donors.append(trace)

    from navedenie.law import BatchMetrics

    abl_rows = {}
    for mode in ("none", "no_theta", "no_rho", "rho_fixed", "rho_episode_perm", "rho_reverse"):
        circuit = load_champion(champions[champ_name]["run_id"])
        episodes = []
        for cell_i, sc in enumerate(abl_grid):
            _apply_rho_ablation(circuit, mode, rho_donors, cell_i, sc.seed, _max_steps(sc))
            episodes.append(evaluate_episode(circuit, sc, teacher=False, t_cap=14.0, scenario_split="test"))
        bmetrics = BatchMetrics(episodes=episodes)
        abl_rows[mode] = {"hit_rate": round(bmetrics.hit_rate, 3), "cpa_median_m": round(bmetrics.cpa_median, 1)}
        log(f"  абляция {mode}: hit={bmetrics.hit_rate:.2f} cpa={bmetrics.cpa_median:.1f} м")
    report["inference_ablation"] = abl_rows

    photo = photometric_robustness(champ, abl_grid, factors=(0.7, 1.0, 1.4), t_cap=14.0)
    report["photometric"] = photo
    log(f"  фотометрия: {photo['verdict']}")

    # ── P13: held-out таблица ──
    log("P13: held-out таблица (81 эпизод × политика)")
    grid = held_out_grid()
    rows: list[dict] = []
    rows.append(pn_law_row("pn", grid))              # oracle ПН N=4
    rows.append(pn_law_row("pn_gsn", grid))          # сенсорная ПН
    rows.append(pn_law_row("pn_sched_oracle", grid))
    rows.append(pn_law_row("pn_sched_sensor", grid))

    warm_circuit = _init_circuit("stub", "pn_coarse_warmstart", 0)
    from navedenie.law import warm_start

    warm_start(warm_circuit, replace(WARM, seed=11), progress=None)
    rows.append(bio_row("BIO после coarse warm-start (этап A)", warm_circuit, grid))
    rows.append(bio_row(f"BIO после outcome evolution ({champ_name})", champ, grid))

    # deployable-суррогаты: параметры из фитинга на ОТДЕЛЬНЫХ train-эпизодах
    log("P13: фитинг H1/H2/H5 на train-эпизодах")
    from navedenie.train import _episode_scenario

    fit_traces = [episode_trace(champ, _episode_scenario(200 + i, champ.gain)) for i in range(5)]
    fits = fit_hypotheses(fit_traces, val_frac=0.4, seed=3)
    report["hypotheses_fit"] = {k: {kk: vv for kk, vv in v.items()} for k, v in fits["hypotheses"].items()}
    for k, v in fits["hypotheses"].items():
        log(f"  {k}: R2={v['r2_val']} params={v['n_params']}")
    laws = []
    try:
        laws.append(("H1 сенсорная ПН (N из H1)", SensorConstantPN(float(fits["hypotheses"]["H1_const_pn"]["coef_summary"]["N"]))))
    except Exception:  # noqa: BLE001
        pass
    try:
        h2 = fits["hypotheses"]["H2_sched_pn"]["coef_summary"]
        laws.append(("H2 scheduled PN (N0,k из H2)", SensorScheduledPN(float(h2["N0"]), float(h2["k_rho_s"]))))
    except Exception:  # noqa: BLE001
        pass
    flight = _flight_fit("stub", champ, 2)
    if flight is not None:
        laws.append(("полином-readout (суррогат dn)", PolyReadout(flight["fits"]["pitch"]["coef"], flight["fits"]["yaw"]["coef"], 2)))
    laws.append(("H5 NARX (лаг 1)", NarxLaw(fits["coef_narx"])))
    for label, law in laws:
        law.reset()
        rows.append(controller_row(label, law, grid))
        log(f"  closed-loop {label}: готово")

    # retraining-абляции в таблице (если чемпионы существуют)
    for abl_name in ("v2_no_theta_retrain", "v2_no_rho_retrain", "v2_no_theta_rho_retrain"):
        if abl_name in champions:
            rows.append(bio_row(f"BIO без фазовых признаков ({abl_name})", load_champion(champions[abl_name]["run_id"]), grid))

    report["held_out_table"] = rows

    # разложение N_eff чемпиона на каноническом трио + 3 ячейках
    log("P6: разложение N_eff чемпиона")
    from navedenie.law import canonical_trio

    dec = neff_decomposition(champ, canonical_trio(1.0) + abl_grid, n_boot=100)
    report["neff_decomposition"] = dec
    log(f"  вердикт: {dec['verdict']} (residual {dec['medians']['residual_median']}, alignment {dec['medians']['alignment_median']})")

    # ── сохранение отчёта ──
    def _json_safe(o):
        if isinstance(o, dict):
            return {k: _json_safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_json_safe(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    (OUT / f"report_{STAMP}.json").write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# Эксперимент «закон наведения» — {STAMP}", ""]
    lines += ["## P13 held-out таблица (81 эпизод: 3 v_m × 3 v_t × 3 аспекта × 3 профиля)", "",
              "| Политика | Перехваты | CI95 | CPA мед., м | CPA p90, м | Усилие, g·с | Насыщ. | Захват |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['policy']} | {round(r['hit_rate']*100)}% | {r['hit_ci95'][0]}–{r['hit_ci95'][1]} | {r['cpa_median_m']} | {r['cpa_p90_m']} | {r['effort_median_gs']} | {r['sat_median']} | {r['lock_median']} |")
    lines += ["", "## Гипотезы (фит на train-эпизодах)", "",
              "| Гипотеза | R² (val) | MAE | Ошибка напр., ° | Параметров |", "|---|---|---|---|---|"]
    for k, v in fits["hypotheses"].items():
        lines.append(f"| {v['label']} | {v['r2_val']} | {v['mae_val']} | {v['direction_err_deg']} | {v['n_params']} |")
    lines += ["", f"## Разложение N_eff чемпиона: вердикт **{dec['verdict']}**", "",
              f"- медианы: {dec['medians']}", f"- CI95: {dec['bootstrap_ci95']}", "",
              "## Инициализации (P4)", "", "| Инициализация | hit rate (среднее) | по seed | CPA мед. |", "|---|---|---|---|"]
    for r in init_rows:
        lines.append(f"| {r['init']} | {r['hit_rate_mean']} | {r['hit_rate_seeds']} | {r['cpa_median_mean_m']} |")
    lines += ["", f"## Фотометрия: {photo['verdict']}", "", "## Inference-абляции чемпиона", "", "| Режим | hit | CPA мед., м |", "|---|---|---|"]
    for m, v in abl_rows.items():
        lines.append(f"| {m} | {v['hit_rate']} | {v['cpa_median_m']} |")
    (OUT / f"report_{STAMP}.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"ГОТОВО: {OUT / f'report_{STAMP}.md'}")


if __name__ == "__main__":
    main()
