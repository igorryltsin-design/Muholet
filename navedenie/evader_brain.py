"""Мозг-уклонист: тот же контур FlyCircuit, но цель смотрит на ракету.

Четвёртый закон уклонения — обучаемая схема.sensor строится честно: цель —
такое же «зрение», как у ракеты (фовеальная сетчатка, декодирование из
изображения, задержка и шум), но в наблюдаемой роли стоит ракета. Врождённый
рефлекс — разворот к пеленгу (бабочка на огонь): ракета берёт такую цель
быстрее слепой; весам W_dn нужно научиться разворачивать ОТ пеленга, выкручивая
ЛВ и жгуя перегрузку наводчика. Тренирует схему evader_train (эволюция популяции
по «школе уклониста»), веса живут в data/weights_evader.npz (brain_store).
"""

from __future__ import annotations

import numpy as np

from navedenie.circuit import FlyCircuit
from navedenie.seeker import ObservationBuffer, observe
from navedenie.sim import Scenario, Vec, clip_accel

EVADER_BRAIN_LAW = "brain"


class EvaderBrainSensor:
    """Сенсор + контур цели на время одного прогона (свои rng/буфер/сетчатка).

    Один экземпляр — один прогон: состояние сетчатки, очередь задержки и
    постоянная карта отказов ретины живут здесь, а не в процессном реестре."""

    def __init__(self, sc: Scenario, circuit: FlyCircuit, seed: int = 1) -> None:
        self.sc = sc
        self.circuit = circuit
        self.rng = np.random.default_rng(int(seed) * 31 + 7)
        self.half_fov = np.deg2rad(max(sc.bio_fov_deg, 2.0)) / 2.0
        self.buffer = ObservationBuffer(sc.seeker_delay_s, 0.0, sc.dt, self.rng)
        self.prev: dict | None = None
        self.lock_seen = False

    def accel(self, target, missile, t: float, dt: float) -> Vec:
        """Нормальная команда цели ⊥ V_ц, клип n_target·g; при потере «вижу
        ракету» — ноль (уклоняться вслепую не от чего)."""
        sc = self.sc
        if sc.n_target <= 0.0:
            return np.zeros(3)
        r = missile.p - target.p
        obs = observe(
            r,
            target.v,
            self.prev,
            half_fov=self.half_fov,
            dt_s=dt,
            noise_az=np.deg2rad(sc.noise_az_deg),
            noise_range=sc.noise_range_m,
            lock_drop_p=sc.lock_drop_p,
            rng=self.rng,
            brightness=getattr(sc, "target_brightness", 1.0),
        )
        obs["retina_alive"] = 1.0
        self.prev = obs
        self.buffer.push(t, obs)
        delayed = self.buffer.sample(t)
        if not delayed.get("lock"):
            return np.zeros(3)
        self.lock_seen = True
        self.circuit.step(delayed, dt)
        a = self.circuit.accel_cmd(target.v, sc.n_target)
        return clip_accel(a, sc.n_target)

    def snapshot(self) -> dict:
        """Компактный снимок контура цели для кадра: те же слои энергии, DN и
        корзина активности act_b64, но без картинки сетчатки (её дублирует
        снимок ракеты — рамка и так тяжёлая)."""
        s = self.circuit.state.snapshot(
            kind="evader", n_cells=self.circuit.n_cells, trained=self.circuit.trained
        )
        s.pop("photo", None)
        s["lock"] = self.lock_seen
        return s
