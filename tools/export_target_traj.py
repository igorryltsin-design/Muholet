"""Экспорт эталонов для сверки фолбэка localSim (JS) с python: траектории ЦЕЛИ
по манёврам + траектория РАКЕТЫ (ПН) в свободной расстановке с виражем цели.

Использование: .venv/bin/python tools/export_target_traj.py > /tmp/ref.json
"""

from __future__ import annotations

import json
import sys

import numpy as np

from navedenie.sim import Scenario, spawn, target_accel, integrate

MANEUVERS = ("weave_var", "break", "scissors", "dive", "combo")
DT = 0.02
T_MAX = 5.0
STRIDE = 10

# ракета в свободной расстановке с виражем цели: сквозной прогон engine.collect —
# именно он ловит расхождения лока фолбэка с стендом (прежний харнес сверил только цель)
MISSILE_SC = dict(
    mode="pn", law="pn", aspect="free", v_m=780, v_t=260, n_max=30, pn_n=4,
    dt=DT, t_max=12.0, kill_radius_m=45, alt_m=4000,
    free_tx=6400.0, free_ty=4800.0, free_talt=4080.0,
    free_mhdg=0.0, free_mclimb=0.0, free_thdg=180.0, free_tclimb=0.0,
    maneuver="turn", n_target=3.0,
)


def main() -> None:
    from navedenie.engine import collect

    out: dict[str, dict] = {}
    ref = collect(Scenario(**MISSILE_SC), stride=1)
    out["missile"] = {
        "scenario": MISSILE_SC,
        "dt": DT,
        "pos": [[float(x) for x in fr.missile] for fr in ref.frames],
        "miss": ref.miss_m,
    }
    for m in MANEUVERS:
        sc = Scenario(aspect="head-on", range_m=6000, v_t=240.0, off_axis_m=200.0, maneuver=m, n_target=3.0, t_max=T_MAX, dt=DT)
        _missile, target = spawn(sc)
        pos: list[list[float]] = [[float(target.p[0]), float(target.p[1]), float(target.p[2])]]
        t = 0.0
        step = 0
        while t < T_MAX:
            integrate(target, target_accel(target, sc, t), DT)
            t += DT
            step += 1
            if step % STRIDE == 0:
                pos.append([float(target.p[0]), float(target.p[1]), float(target.p[2])])
        out[m] = {"dt": DT * STRIDE, "pos": pos}
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
