"""Контур дрозофилы: признаки из ГСН → скрытые слои → DN (readout).

«Коннектом» — это CONNECTOME-INFORMED RATE MODEL: размеры регионов и проводка
МЕЖДУ НИМИ взяты из таблиц FlyWire FAFB (реальные синапсы), но моторный выход
выполняет синтетический обучаемый readout: 10 признаков → W_fx (каналы) → W_dn →
2 команды. Число ~139 тыс нейронов — масштаб внутренней активности, а НЕ число
независимо обучаемых управляющих нейронов (обучаются только W_dn и W_fx).

Формулу ПН не копируем. Обучается только readout. Знаки скрытых связей фиксированы.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from navedenie.sim import G, Vec, _perp, clip_accel

LAYERS = ("VISION", "FLOW", "LOOM", "DECISION", "MOTOR")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEAT_DIM = 10  # схема признаков v2: + theta (угловой размер) и rho = θ̇/θ (фаза сближения)
LEGACY_FEAT_DIM = 8  # схема v1: без theta/rho (старые мозги мигрируют нулями на новых столбцах)
FEATURE_SCHEMA_VERSION = 2
# старые столбцы v1 (az, el, ωβ, ωλ, loom, Φx, Φy, захват) сохраняют смысл:
LEGACY_COLUMN_MAP = (0, 1, 2, 3, 5, 7, 8, 9)
# новые столбцы v2: 4 = theta, 6 = rho — при миграции заполняются нулями
NEW_FEATURE_COLUMNS = (4, 6)
# нормировка фазовых признаков по фактическим диапазонам сенсора (измерено на
# реальных сближениях): theta ∈ 0…0.45 рад в операционной зоне, rho ∈ 0…8 1/с
K_THETA = 4.0  # рад⁻¹
K_RHO = 0.4   # (1/с)⁻¹
PHOTO_N = 16  # сетчатка: PHOTO_N×PHOTO_N «омматидиев» (см. seeker.SEEKER_N)
PHOTO_CELLS = PHOTO_N * PHOTO_N
ACT_BINS = 2048
FLYWIRE_NEURONS = 139_000
FLYWIRE_SYNAPSES = 175_000_000


def migrate_feature_columns(w_in: np.ndarray, axis: int = 1, from_dim: int = LEGACY_FEAT_DIM) -> np.ndarray:
    """Расширить матрицу весов со старой схемы признаков на текущую.

    Старые столбцы сохраняют прежний смысл (LEGACY_COLUMN_MAP), новые
    (theta, rho) заполняются нулями — выученная политика не меняется до
    дообучения. axis — ось, по которой идут признаки."""
    w = np.asarray(w_in)
    cur = w.shape[axis]
    if cur == FEAT_DIM:
        return w.copy()
    if cur != from_dim:
        raise ValueError(
            f"несовместимая схема признаков: ожидается {FEAT_DIM} или {LEGACY_FEAT_DIM} столбцов, получено {cur}"
        )
    w = np.moveaxis(w, axis, -1)
    out = np.zeros(w.shape[:-1] + (FEAT_DIM,), dtype=w.dtype)
    for new_idx, old_idx in enumerate(LEGACY_COLUMN_MAP):
        out[..., old_idx] = w[..., new_idx]
    return np.moveaxis(out, -1, axis)


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _hash_blocks(blocks: list[np.ndarray]) -> str:
    h = hashlib.sha256()
    for w in blocks:
        h.update(np.ascontiguousarray(w).tobytes())
    return h.hexdigest()[:16]


def _features(obs: dict, t4: np.ndarray, loom: float) -> np.ndarray:
    """Схема признаков v2 (порядок фиксирован, см. FEATURE_SCHEMA_VERSION):

    0 az, 1 el, 2 ω_az, 3 ω_el, 4 theta (угловой размер, рад), 5 loom,
    6 rho = θ̇/θ (фаза сближения, 1/с), 7 Φx, 8 Φy, 9 захват.
    theta/rho приходят ИЗ obs — то есть декодированы из изображения после
    фовеи, шумов, отказов и задержки; истинные дальность/Vc/t_go не входят."""
    lock = 1.0 if obs.get("lock") else 0.0
    az = float(obs.get("az") or 0.0) * lock
    el = float(obs.get("el") or 0.0) * lock
    az_dot = float(obs.get("az_dot") or 0.0) * lock
    el_dot = float(obs.get("el_dot") or 0.0) * lock
    theta = float(obs.get("theta") or 0.0) * lock
    rho = float(obs.get("rho") or 0.0) * lock
    flow_x = float(t4[1] - t4[0]) if t4.size >= 4 else 0.0
    flow_y = float(t4[3] - t4[2]) if t4.size >= 4 else 0.0
    return np.tanh(
        np.array(
            [
                az * 3, el * 3, az_dot * 0.4, el_dot * 0.4,
                theta * K_THETA, loom, rho * K_RHO,
                flow_x * 4, flow_y * 4, lock,
            ],
            dtype=np.float64,
        )
    )


def default_W() -> np.ndarray:
    """Старт: поворот к пеленгу + чуть потока. Не ПН. Новые столбцы (theta, rho)
    равны нулю — врождённый рефлекс в точности как в схеме v1."""
    w = np.zeros((2, FEAT_DIM))
    w[0, 1] = 1.8   # el
    w[0, 3] = 0.35  # el_dot
    w[0, 5] = 0.25  # loom
    w[0, 8] = 0.4   # flow_y
    w[1, 0] = 1.8   # az
    w[1, 2] = 0.35  # az_dot
    w[1, 5] = 0.25  # loom
    w[1, 7] = 0.4   # flow_x
    return w


def _pack_activity(vecs: list[tuple[str, np.ndarray]], bins: int = ACT_BINS) -> dict:
    """Реальная активность всех нейронов → регионы + проекция в корзины (base64 uint8).

    Порядок склейки фиксирован: нейрон i попадает в корзину floor(i/(N/bins)),
    клиент раскладывает точки в том же порядке.
    """
    parts = [np.asarray(v, dtype=np.float64).reshape(-1) for _, v in vecs]
    sizes = [p.size for p in parts]
    total = int(sum(sizes))
    cat = np.concatenate(parts) if parts else np.zeros(1)
    bpb = max(total / bins, 1.0)
    idx = np.minimum((np.arange(total) / bpb).astype(np.int64), bins - 1)
    bins_val = np.zeros(bins, dtype=np.uint8)
    if total:
        m = np.abs(cat)
        np.maximum.at(bins_val, idx, np.clip(m * 255.0, 0, 255).astype(np.uint8))
    regions = []
    off = 0
    for (name, _v), p in zip(vecs, parts):
        regions.append(
            {
                "name": name,
                "start": int(off),
                "n": int(p.size),
                "mean": float(np.mean(np.abs(p))) if p.size else 0.0,
            }
        )
        off += p.size
    return {
        "n_neurons": total,
        "regions": regions,
        "act_b64": base64.b64encode(bins_val.tobytes()).decode("ascii"),
    }


@dataclass
class CircuitState:
    photo: np.ndarray = field(default_factory=lambda: np.zeros(PHOTO_CELLS))
    t4: np.ndarray = field(default_factory=lambda: np.zeros(4))
    t5: np.ndarray = field(default_factory=lambda: np.zeros(4))
    lc11: np.ndarray = field(default_factory=lambda: np.zeros(4))
    lc10: np.ndarray = field(default_factory=lambda: np.zeros(2))
    lplc2: float = 0.0
    aotu: np.ndarray = field(default_factory=lambda: np.zeros(6))
    kenyon: np.ndarray = field(default_factory=lambda: np.zeros(1))
    pool: np.ndarray = field(default_factory=lambda: np.zeros(32))
    dn: np.ndarray = field(default_factory=lambda: np.zeros(2))
    feat: np.ndarray = field(default_factory=lambda: np.zeros(FEAT_DIM))

    def neuron_vectors(self) -> list[tuple[str, np.ndarray]]:
        """Все нейроны контура в каноническом порядке (для карты активности)."""
        return [
            ("глаз", self.photo),
            ("T4", self.t4),
            ("T5", self.t5),
            ("LC11", self.lc11),
            ("LC10", self.lc10),
            ("LPLC2", np.atleast_1d(self.lplc2)),
            ("AOTU", self.aotu),
            ("грибовидное тело", self.kenyon),
            ("пул", self.pool),
            ("DN", self.dn),
        ]

    def snapshot(self, *, kind: str, n_cells: int, trained: bool) -> dict:
        def mean_abs(x: np.ndarray | float) -> float:
            arr = np.atleast_1d(x)
            return float(np.mean(np.abs(arr)))

        energy = {
            "VISION": mean_abs(self.photo),
            "FLOW": 0.5 * (mean_abs(self.t4) + mean_abs(self.t5)),
            "LOOM": abs(self.lplc2),
            "DECISION": 0.5 * (mean_abs(self.lc11) + mean_abs(self.aotu) + mean_abs(self.kenyon)),
            "MOTOR": mean_abs(self.dn),
        }
        return {
            "photo": np.asarray(self.photo[:PHOTO_CELLS]).reshape(PHOTO_N, PHOTO_N).tolist(),
            "t4": self.t4.tolist(),
            "t5": self.t5.tolist(),
            "lc11": self.lc11.tolist(),
            "lc10": self.lc10.tolist(),
            "lplc2": self.lplc2,
            "aotu": self.aotu.tolist(),
            "dn": {"pitch": float(self.dn[0]), "yaw": float(self.dn[1])},
            "layers": energy,
            "weights": "обучен" if trained else "не обучен",
            "n_cells": n_cells,
            "kind": kind,
            **_pack_activity(self.neuron_vectors()),
        }


class FlyCircuit:
    def __init__(self, tau_s: float = 0.025, gain: float = 1.0, kind: str = "stub", pool_size: int = 32, expand: int = 4096) -> None:
        self.tau = tau_s
        self.gain = gain
        self.kind = "full" if kind == "full" else "stub"
        self.seed = 23  # зерно случайной проводки: рецепт воспроизводимости
        self.pool_size = int(pool_size)
        self.n_expand = int(expand) if self.kind == "full" else 0
        self.n_cells = PHOTO_CELLS + 23 + (self.n_expand + self.pool_size + self.pool_size if self.kind == "full" else 0)
        self.state = CircuitState()
        if self.n_expand == 0:
            self.state.pool = self.state.pool[:0]
        self.W_dn = default_W()
        self.trained = False
        # абляции признаков: {индекс: значение | последовательность по шагам};
        #_SEQUENCE значений индексируется номером шага эпизода (перестановка/фиксация rho)
        self.feat_patch: dict[int, float | np.ndarray] | None = None
        self._step_i = 0
        rng = np.random.default_rng(23)
        if self.n_expand:
            self.W_k = rng.standard_normal((self.n_expand, FEAT_DIM)) * 0.35
            self.W_k *= rng.random(self.W_k.shape) < 0.12
            self.W_pool = rng.standard_normal((self.pool_size, self.n_expand)) * (1.0 / np.sqrt(max(self.n_expand * 0.12, 1)))
            self.W_dn = rng.standard_normal((2, self.pool_size)) * 0.05
            self.W_dn[0, 1] = 0.4
            self.W_dn[1, 0] = 0.4
        else:
            self.W_k = None
            self.W_pool = None

    def reset(self) -> None:
        self.state = CircuitState()
        self._step_i = 0

    def _patched(self, feat: np.ndarray) -> np.ndarray:
        if not self.feat_patch:
            return feat
        out = feat.copy()
        for idx, val in self.feat_patch.items():
            i = int(idx)
            if np.ndim(val) == 0:
                out[i] = float(val)  # type: ignore[arg-type]
            else:
                seq = np.asarray(val).reshape(-1)
                out[i] = float(seq[min(self._step_i, seq.size - 1)])
        return out

    def step(self, obs: dict, dt: float) -> CircuitState:
        s = self.state
        img = np.asarray(obs.get("image", np.zeros((PHOTO_N, PHOTO_N))), dtype=np.float64).reshape(-1)
        if img.size < PHOTO_CELLS:
            img = np.pad(img, (0, PHOTO_CELLS - img.size))
        img = img[:PHOTO_CELLS]
        s.photo = (1 - dt / self.tau) * s.photo[:PHOTO_CELLS] + (dt / self.tau) * img
        if s.photo.size != PHOTO_CELLS:
            s.photo = s.photo[:PHOTO_CELLS]

        grid = s.photo.reshape(PHOTO_N, PHOTO_N)
        h = PHOTO_N // 2
        gx = grid[:, h:].mean() - grid[:, :h].mean()
        gy = grid[h:, :].mean() - grid[:h, :].mean()
        flow = np.array([max(-gx, 0), max(gx, 0), max(-gy, 0), max(gy, 0)])
        s.t4 = (1 - dt / self.tau) * s.t4 + (dt / self.tau) * flow
        s.t5 = (1 - dt / (self.tau * 1.2)) * s.t5 + (dt / (self.tau * 1.2)) * flow[::-1] * 0.4

        lock = bool(obs.get("lock"))
        az = float(obs.get("az") or 0.0) if lock else 0.0
        el = float(obs.get("el") or 0.0) if lock else 0.0
        loom = max(0.0, float(obs.get("size_dot") or 0.0) * 40.0) if lock else 0.0
        quad = np.array(
            [
                max(0.0, -az) * max(0.0, -el),
                max(0.0, az) * max(0.0, -el),
                max(0.0, -az) * max(0.0, el),
                max(0.0, az) * max(0.0, el),
            ]
        )
        s.lc11 = (1 - dt / self.tau) * s.lc11 + (dt / self.tau) * np.tanh(quad * 8.0)
        s.lc10 = (1 - dt / self.tau) * s.lc10 + (dt / self.tau) * np.tanh(np.array([el, az]) * 6.0)
        s.lplc2 = (1 - dt / self.tau) * s.lplc2 + (dt / self.tau) * np.tanh(loom)
        s.aotu = np.array(
            [s.t4[1] - s.t4[0], s.t4[3] - s.t4[2], s.lc10[1], s.lc10[0], float(s.lc11.mean()), s.lplc2]
        )
        s.feat = self._patched(_features(obs, s.t4, s.lplc2))
        self._step_i += 1
        if self.W_k is not None and self.W_pool is not None:
            k = np.maximum(self.W_k @ s.feat, 0.0)
            s.kenyon = k / (np.linalg.norm(k) + 1e-6)
            hidden = np.tanh(self.W_pool @ s.kenyon)
            s.pool = hidden
            target_dn = np.tanh(self.W_dn @ hidden)
        else:
            s.kenyon = s.feat
            s.pool = s.pool[:0]
            target_dn = np.tanh(self.W_dn @ s.feat)
        s.dn = (1 - dt / self.tau) * s.dn + (dt / self.tau) * target_dn
        return s

    def hidden(self) -> np.ndarray:
        s = self.state
        if self.W_k is not None and self.W_pool is not None:
            k = np.maximum(self.W_k @ s.feat, 0.0)
            return np.tanh(self.W_pool @ (k / (np.linalg.norm(k) + 1e-6)))
        return s.feat

    def learn_step(self, teacher: np.ndarray, lr: float, lr_hidden: float = 0.0) -> np.ndarray:
        """Шаг обучения учителем: дельта-правило на выходе.

        «Полный» дополнительно проводит градиент в скрытый слой W_pool
        (пластичность грибовидного тела); W_k остаётся врождённой проводкой.
        """
        h = self.hidden()
        pred = np.tanh(self.W_dn @ h)
        err = teacher - pred
        if lr_hidden > 0.0 and self.W_pool is not None:
            dp = np.clip(np.dot(self.W_dn.T, err) * (1.0 - self.state.pool ** 2), -1.0, 1.0)
            self.W_pool += lr_hidden * np.outer(dp, self.state.kenyon)  # kenyon уже нормализован
            np.clip(self.W_pool, -2.0, 2.0, out=self.W_pool)
        self.W_dn += lr * np.outer(err, h)
        self.W_dn = np.clip(self.W_dn, -4.0, 4.0)
        return pred

    def wiring(self) -> dict:
        """Рецепт проводки: зерно + скрытые блоки (у полного — целиком) + хеш целостности."""
        blocks: dict[str, str] = {}
        hashes: list[np.ndarray] = []
        if self.W_k is not None:
            blocks["W_k"] = _b64(self.W_k)
            hashes.append(self.W_k)
        if self.W_pool is not None:
            blocks["W_pool"] = _b64(self.W_pool)
            hashes.append(self.W_pool)
        out = {"algo": "v1", "seed": self.seed}
        if blocks:
            out["blocks"] = blocks
            out["shapes"] = {k: list(v.shape) for k, v in zip(blocks, hashes)}
            out["hash"] = _hash_blocks(hashes)
        return out

    def restore_wiring(self, wiring: dict) -> bool:
        """Восстановить скрытые блоки из файла. False — хеш не совпал."""
        blocks = wiring.get("blocks") or {}
        restored: list[np.ndarray] = []
        for name, target in (("W_k", self.W_k), ("W_pool", self.W_pool)):
            if name not in blocks or target is None:
                continue
            arr = np.frombuffer(base64.b64decode(blocks[name]), dtype=np.float64).reshape(target.shape).copy()
            restored.append(arr)
        if wiring.get("hash") and _hash_blocks(restored) != wiring["hash"]:
            return False
        for name, arr in zip(("W_k", "W_pool"), restored):
            if name == "W_k" and self.W_k is not None:
                self.W_k = arr
            if name == "W_pool" and self.W_pool is not None:
                self.W_pool = arr
        return True

    def accel_cmd(self, v_m: Vec, n_max: float) -> Vec:
        from navedenie.seeker import body_axes

        _x, y, z = body_axes(v_m)
        n_pitch = float(self.state.dn[0]) * n_max * G * self.gain
        n_yaw = float(self.state.dn[1]) * n_max * G * self.gain
        a = z * n_pitch + y * n_yaw
        return clip_accel(_perp(a, v_m), n_max)

    def save(self, path: Path | None = None) -> Path:
        """Полный снимок: выход + скрытые блоки + усиление/тау/зерно проводки +
        версия схемы признаков (совместимость сохранённых мозгов)."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = path or DATA_DIR / f"weights_{self.kind}.npz"
        payload: dict = {
            "W_dn": self.W_dn,
            "trained": np.array([1 if self.trained else 0], dtype=np.int8),
            "gain": np.float64(self.gain),
            "tau": np.float64(self.tau),
            "seed": np.int64(self.seed),
            "feat_schema": np.int64(FEATURE_SCHEMA_VERSION),
        }
        if self.W_k is not None:
            payload["W_k"] = self.W_k
        if self.W_pool is not None:
            payload["W_pool"] = self.W_pool
        np.savez(path, **payload)
        return path

    def load(self, path: Path | None = None) -> bool:
        """Загрузка с МИГРАЦИЕЙ схем признаков: мозг v1 (8 признаков) расширяется
        нулями на столбцах theta/rho — поведение сохраняется до дообучения.
        False — файл несовместим (чужая форма, схема новее стенда)."""
        path = path or DATA_DIR / f"weights_{self.kind}.npz"
        if not path.exists():
            return False
        blob = np.load(path, allow_pickle=False)
        stored_schema = int(np.asarray(blob["feat_schema"]).reshape(-1)[0]) if "feat_schema" in blob else 1
        if stored_schema > FEATURE_SCHEMA_VERSION:
            return False  # мозг из более новой версии стенда
        w = blob["W_dn"]
        if w.ndim == 2 and w.shape[1] != self.W_dn.shape[1] and w.shape[1] in (LEGACY_FEAT_DIM, FEAT_DIM):
            w = migrate_feature_columns(w, axis=1, from_dim=w.shape[1])
        if w.shape != self.W_dn.shape:
            return False
        self.W_dn = w
        self.trained = bool(blob["trained"][0]) if "trained" in blob else True
        if "gain" in blob:
            self.gain = float(blob["gain"])
        if "tau" in blob:
            self.tau = float(blob["tau"])
        if "seed" in blob:
            self.seed = int(blob["seed"])
        if "W_k" in blob and self.W_k is not None:
            wk = blob["W_k"]
            if wk.ndim == 2 and wk.shape[1] != self.W_k.shape[1] and wk.shape[1] in (LEGACY_FEAT_DIM, FEAT_DIM):
                wk = migrate_feature_columns(wk, axis=1, from_dim=wk.shape[1])
            if wk.shape == self.W_k.shape:
                self.W_k = wk
        if "W_pool" in blob and self.W_pool is not None and blob["W_pool"].shape == self.W_pool.shape:
            self.W_pool = blob["W_pool"]
        return True


def neuron_cloud(state: CircuitState, n: int = 180, kind: str = "stub") -> list[dict]:
    rng = np.random.default_rng(7)
    count = 420 if kind == "full" else n
    cloud: list[dict] = []
    k_act = float(np.mean(state.kenyon)) if state.kenyon.size else 0.0
    regions = [
        ("lamina", state.photo.mean(), 0.22 if kind == "full" else 0.18, (-1.1, 0.2, 0.4)),
        ("medulla", 0.5 * (state.t4.mean() + state.t5.mean()), 0.2, (-0.5, 0.1, 0.2)),
        ("lobula", state.lc11.mean(), 0.16, (0.05, 0.0, 0.1)),
        ("kenyon", k_act, 0.22 if kind == "full" else 0.08, (0.45, -0.05, 0.15)),
        ("dn", float(np.mean(np.abs(state.dn))), 0.12, (0.95, 0.0, -0.15)),
    ]
    remain = count
    for i, (name, rate, frac, center) in enumerate(regions):
        k = int(count * frac) if i < len(regions) - 1 else remain
        remain -= k
        pts = rng.normal(loc=center, scale=0.16, size=(max(k, 1), 3))
        for p in pts:
            cloud.append(
                {
                    "x": float(p[0]),
                    "y": float(p[1]),
                    "z": float(p[2]),
                    "r": float(np.clip(abs(rate) * 1.8 + rng.random() * 0.12, 0, 1)),
                    "region": name,
                }
            )
    return cloud


# ─── Мозг масштаба коннектома ────────────────────────────────────────────────
#FlyWire (Dorkenwald et al., Nature 2024): ~139 тыс нейронов, ~175 млн синапсов.
#Полная проводка в репозитории отсутствует (см. NOTICE), поэтому связи между
#регионами структурно правдоподобные: плотные блоки через узкие «базисы».
#Число нейронов и синапсов в модели считается честно и показывается в UI.

CONNECTOME_REGIONS: list[tuple[str, int, str]] = [
    # имя, размер, описание (для подсказок UI)
    ("ламина", 1600, "Первичная оптика: контраст из 64 каналов глаза."),
    ("медулла", 75000, "Основная масса оптической доли: элементы движения и потока."),
    ("T4/T5", 6400, "Детекторы направления: куда ползёт образ цели."),
    ("лобула", 44000, "Опишет положение и признаки цели: LC-нейроны."),
    ("LPLC2", 1500, "Радар приближения: пятно растёт — сигнал тревоги."),
    ("центральный комплекс", 2000, "Интеграция и «решение» о манёвре."),
    ("грибовидное тело", 8000, "Ассоциативный слой: контекст и обучение."),
    ("DN", 200, "Даунины: populations выходят на рули — тангаж и рыскание."),
]

# Зоны СИНТЕТИЧЕСКИХ каналов readout 10→N — банки по мотивам детекторных банков
# зрительной доли (НЕ проводка FlyWire: каналы создаются случайно, seed+1).
# Сумма должна равняться числу каналов по умолчанию (128).
CHANNEL_ZONES: list[tuple[str, int]] = [
    ("прямой", 10),         # точная копия одного признака — врождённый канал (все 10 признаков)
    ("пеленг", 28),         # смеси β, λ, ωβ, ωλ
    ("поток", 28),          # смеси θ, θ̇(loom), rho, Φx, Φy, [захв]
    ("ассоциативный", 62),  # разреженные случайные комбинации всех признаков
]

_CONN_STEP = 0.02  # внутренний шаг сети, 50 Гц

# блоки скрытой проводки: структурные (_build) или реальные (circuit_v1.npz)
_REAL_BLOCKS = ("W_lam", "W_bA", "W_med", "W_bB", "W_t4", "W_bC", "W_lob", "W_lob2", "W_lp", "W_c", "W_mb")


class _ConnectomeState:
    """Векторы активности регионов + совместимый со CircuitState снимок."""

    def __init__(self, sizes: list[tuple[str, int, str]]) -> None:
        self.vecs = {name: np.zeros(n, dtype=np.float32) for name, n, _ in sizes}

    def neuron_vectors(self) -> list[tuple[str, np.ndarray]]:
        return [(name, self.vecs[name]) for name, _, _ in CONNECTOME_REGIONS]

    def snapshot(self, *, kind: str, n_cells: int, trained: bool) -> dict:
        def m(name: str) -> float:
            v = self.vecs[name]
            return float(np.mean(np.abs(v))) if v.size else 0.0

        energy = {
            "VISION": m("ламина"),
            "FLOW": m("T4/T5"),
            "LOOM": m("LPLC2"),
            "DECISION": m("центральный комплекс"),
            "MOTOR": m("DN"),
        }
        dn = self.vecs["DN"]
        cmd = float(np.mean(dn[: dn.size // 2])) if dn.size else 0.0
        yaw = float(np.mean(dn[dn.size // 2 :])) if dn.size else 0.0
        return {
            "photo": np.zeros((PHOTO_N, PHOTO_N)).tolist(),
            "t4": self.vecs["T4/T5"][:4].tolist(),
            "t5": self.vecs["T4/T5"][4:8].tolist(),
            "lc11": self.vecs["лобула"][:4].tolist(),
            "lc10": self.vecs["лобула"][4:6].tolist(),
            "lplc2": m("LPLC2"),
            "aotu": self.vecs["центральный комплекс"][:6].tolist(),
            "dn": {"pitch": cmd, "yaw": yaw},
            "layers": energy,
            "weights": "обучен" if trained else "не обучен",
            "n_cells": n_cells,
            "kind": kind,
            **_pack_activity(self.neuron_vectors()),
        }


class ConnectomeCircuit:
    """Connectome-informed rate model: ~139 тыс rate-нейронов с проводкой регионов
    по FlyWire + синтетический обучаемый readout 10→каналы→2 команды."""

    def __init__(self, tau_s: float = 0.025, gain: float = 1.0, kind: str = "connectome", seed: int = 42, basis: int = 192, channels: int = 128) -> None:
        self.tau = tau_s
        self.gain = gain
        self.kind = "connectome"
        self.trained = False
        self.seed = seed
        self.basis = basis
        self.channels = int(channels)
        self.rng = np.random.default_rng(seed)
        self.acc = 0.0
        self.state = _ConnectomeState(CONNECTOME_REGIONS)
        self.n_cells = int(sum(n for _, n, _ in CONNECTOME_REGIONS))
        self.real_wiring = False
        self.real_meta: dict = {}
        # синаптическое расширение признаков 10→N — врождённая проводка (зерно seed+1),
        # каналы сгруппированы в ЗОНЫ-банки, как детекторные банки зрительной доли:
        # прямые (копия одного признака), пеленговые (β/λ/ωβ/ωλ), потоковые
        # (θ̇/Φ/захват), ассоциативные (разреженные комбинации всех признаков).
        rng2 = np.random.default_rng(seed + 1)
        self.W_fx = np.zeros((self.channels, FEAT_DIM))
        self.channel_zones: list[str] = []
        i0 = 0
        for zname, zcount in CHANNEL_ZONES:
            take = min(zcount, self.channels - i0)
            if take <= 0:
                break
            if zname == "прямой":
                for i in range(take):
                    self.W_fx[i0 + i, i % FEAT_DIM] = 1.0
            elif zname == "пеленг":
                sup = np.zeros(FEAT_DIM, dtype=bool)
                sup[[0, 1, 2, 3]] = True  # beta3, lam3, wb, wl
                mask = rng2.random((take, FEAT_DIM)) < 0.5
                self.W_fx[i0 : i0 + take] = (rng2.standard_normal((take, FEAT_DIM)) * 0.4) * mask * sup
            elif zname == "поток":
                sup = np.zeros(FEAT_DIM, dtype=bool)
                sup[[4, 5, 6, 7, 8, 9]] = True  # theta, loom, rho, flow_x, flow_y, lock
                mask = rng2.random((take, FEAT_DIM)) < 0.5
                self.W_fx[i0 : i0 + take] = (rng2.standard_normal((take, FEAT_DIM)) * 0.4) * mask * sup
            else:  # ассоциативные
                mask = rng2.random((take, FEAT_DIM)) < 0.25
                self.W_fx[i0 : i0 + take] = (rng2.standard_normal((take, FEAT_DIM)) * 0.3) * mask
            self.channel_zones.extend([zname] * take)
            i0 += take
        self.b_fx = np.full(self.channels, 0.08)  # смещение: каналы не «умирают» при слабом сигнале
        # врождённый рефлекс выхода — как у схемы (default_W): тангаж ← высотный пеленг,
        # рыскание ← боковой; случайные каналы подключены слабо
        self.W_dn = rng2.standard_normal((2, self.channels)) * 0.02
        self.W_dn[0, 1] = 1.8
        self.W_dn[0, 3] = 0.35
        self.W_dn[0, 4] = 0.25
        self.W_dn[0, 6] = 0.4
        self.W_dn[1, 0] = 1.8
        self.W_dn[1, 2] = 0.35
        self.W_dn[1, 4] = 0.25
        self.W_dn[1, 5] = 0.4
        self.feat = np.zeros(FEAT_DIM)
        self._dn = np.zeros(2)
        self.feat_patch: dict[int, float | np.ndarray] | None = None  # абляции признаков
        self._step_i = 0
        # реальная проводка FlyWire, если собрана tools/extract_subgraph.py
        if not self._load_real_wiring():
            self._build()

    REAL_WIRING_FILE = DATA_DIR / "circuit_v1.npz"

    def _load_real_wiring(self) -> bool:
        """Настоящая проводка FlyWire из data/circuit_v1.npz (см. tools/extract_subgraph.py)."""
        if not self.REAL_WIRING_FILE.exists():
            return False
        try:
            blob = np.load(self.REAL_WIRING_FILE, allow_pickle=False)
            self.real_meta = json.loads(str(blob["meta"]))
            for name in _REAL_BLOCKS:
                setattr(self, name, blob[name].astype(np.float32))
            self.n_synapses = int(self.real_meta.get("n_synapses", 0))
            self.real_wiring = True
            return True
        except Exception:  # noqa: BLE001 — повреждённый файл не должен валить стенд
            self.real_wiring = False
            self.real_meta = {}
            return False

    def _build(self) -> None:
        rng = self.rng
        B = self.basis

        def block(r: int, c: float, scale: float) -> np.ndarray:
            w = rng.standard_normal((r, c)).astype(np.float32)
            return (w * np.float32(scale)).astype(np.float32)  # type: ignore[return-value]

        self.W_lam = block(1600, 64, 0.35)
        self.W_bA = block(B, 1600, 1.0 / np.sqrt(1600))
        self.W_med = block(75000, B, 1.0 / np.sqrt(B))
        self.W_bB = block(B, 75000, 1.0 / np.sqrt(75000))
        self.W_t4 = block(6400, B, 1.0 / np.sqrt(B))
        self.W_bC = block(64, 6400, 1.0 / np.sqrt(6400))
        self.W_lob = block(44000, B, 1.0 / np.sqrt(B))
        self.W_lob2 = block(44000, 64, 1.0 / np.sqrt(64))
        self.W_lp = block(1500, 64, 1.0 / np.sqrt(64))
        self.W_c = block(2000, B, 1.0 / np.sqrt(B))
        self.W_mb = block(8000, B, 1.0 / np.sqrt(B))
        self.n_synapses = int(
            self.W_lam.size + self.W_bA.size + self.W_med.size + self.W_bB.size
            + self.W_t4.size + self.W_bC.size + self.W_lob.size + self.W_lob2.size
            + self.W_lp.size + self.W_c.size + self.W_mb.size
        )
        self._blocks = [
            self.W_lam, self.W_bA, self.W_med, self.W_bB, self.W_t4,
            self.W_bC, self.W_lob, self.W_lob2, self.W_lp, self.W_c, self.W_mb,
        ]

    def install_real_wiring(self, blob_bytes: bytes) -> bool:
        """Установить реальную проводку FlyWire из .npz (импорт полного мозга)."""
        import io

        try:
            blob = np.load(io.BytesIO(blob_bytes), allow_pickle=False)
            meta = json.loads(str(blob["meta"]))
            for name in _REAL_BLOCKS:
                arr = blob[name]
                if arr.dtype != np.float16:
                    return False
                setattr(self, name, arr.astype(np.float32))
            self.real_meta = meta
            self.n_synapses = int(meta.get("n_synapses", 0))
            self.real_wiring = True
            return True
        except Exception:  # noqa: BLE001
            self.real_wiring = False
            self.real_meta = {}
            return False

    def wiring(self) -> dict:
        """Рецепт проводки: зерно + структура + хеш всех синапсов.

        Структурная проводка восстанавливается из зерна; реальная (FlyWire) —
        хешируется по файлу circuit_v1.npz, рецепт ссылается на источник.
        """
        out: dict = {
            "algo": "v1-real" if self.real_wiring else "v1",
            "seed": self.seed,
            "basis": self.basis,
            "regions": [[name, n] for name, n, _ in CONNECTOME_REGIONS],
            "n_synapses": int(self.n_synapses),
        }
        if self.real_wiring:
            out["source"] = self.real_meta.get("source", "FlyWire FAFB v783")
            out["data_version"] = self.real_meta.get("data_version", "783")
            out["license"] = self.real_meta.get("license", "CC BY-NC-SA 4.0")
        if self.real_wiring:
            h = hashlib.sha256()
            for name in _REAL_BLOCKS:
                h.update(np.ascontiguousarray(getattr(self, name).astype(np.float16)).tobytes())
            out["hash"] = h.hexdigest()[:16]
        else:
            out["hash"] = _hash_blocks(self._blocks)
        return out

    def restore_wiring(self, wiring: dict) -> bool:
        expected = self.wiring()
        if wiring.get("hash") != expected["hash"] or wiring.get("seed") != expected["seed"]:
            return False
        if self.real_wiring and wiring.get("algo") != "v1-real":
            return False
        return True

    def reset(self) -> None:
        for name in self.state.vecs:
            self.state.vecs[name][:] = 0.0
        self.acc = 0.0
        self._dn[:] = 0.0
        self.feat[:] = 0.0
        self._step_i = 0

    def hidden(self) -> np.ndarray:
        return np.tanh(self.W_fx @ self.feat + self.b_fx)

    def learn_step(self, teacher: np.ndarray, lr: float, lr_hidden: float = 0.0) -> np.ndarray:
        """Обучение учителем: выход 2×64 дельта-правилом, синаптические каналы W_fx — градиентом."""
        h = self.hidden()
        pred = np.tanh(self.W_dn @ h)
        err = teacher - pred
        self.W_dn += lr * np.outer(err, h)
        self.W_dn = np.clip(self.W_dn, -4.0, 4.0)
        if lr_hidden > 0.0:
            dpred = err * (1.0 - pred ** 2)
            dh = np.dot(self.W_dn.T, dpred) * (1.0 - h ** 2)
            self.W_fx += lr_hidden * np.outer(dh, self.feat)
            np.clip(self.W_fx, -2.0, 2.0, out=self.W_fx)
        return pred

    def step(self, obs: dict, dt: float) -> None:
        self.feat = _features(obs, self.state.vecs["T4/T5"][:4], float(np.mean(np.abs(self.state.vecs["LPLC2"]))))
        if self.feat_patch:
            out = self.feat.copy()
            for idx, val in self.feat_patch.items():
                i = int(idx)
                if np.ndim(val) == 0:
                    out[i] = float(val)  # type: ignore[arg-type]
                else:
                    seq = np.asarray(val).reshape(-1)
                    out[i] = float(seq[min(self._step_i, seq.size - 1)])
            self.feat = out
        self._step_i += 1
        self.acc += dt
        while self.acc >= _CONN_STEP:
            self.acc -= _CONN_STEP
            self._integrate(obs)

    def _integrate(self, obs: dict) -> None:
        s = self.state.vecs
        k = float(np.clip(_CONN_STEP / self.tau, 0.0, 1.0))
        img = np.asarray(obs.get("image", np.zeros((PHOTO_N, PHOTO_N))), dtype=np.float32).reshape(-1)[:PHOTO_CELLS]
        drive = float(np.mean(img)) if img.size else 0.0
        if img.size < PHOTO_CELLS:
            img = np.pad(img, (0, PHOTO_CELLS - img.size))
        # конвергенция 2×2 омматидиев → один канал лампины (проводка FlyWire 1600×64)
        eye = img.reshape(PHOTO_N, PHOTO_N).reshape(PHOTO_N // 2, 2, PHOTO_N // 2, 2).mean(axis=(1, 3)).astype(np.float32).reshape(-1)

        lam = np.tanh(self.W_lam @ eye + 0.1 * drive)
        bA = np.tanh(self.W_bA @ lam)
        med = np.maximum(self.state.vecs["медулла"] + (np.tanh(self.W_med @ bA) - self.state.vecs["медулла"]) * k, 0.0)
        bB = np.tanh(self.W_bB @ med)

        az = float(obs.get("az") or 0.0) if obs.get("lock") else 0.0
        el = float(obs.get("el") or 0.0) if obs.get("lock") else 0.0
        dir_drive = np.array([az, -az, el, -el], dtype=np.float32) * float(obs.get("retina_alive", 1.0))
        t4 = np.maximum(self.state.vecs["T4/T5"] + (np.tanh(self.W_t4 @ bB + 0.4 * np.tile(dir_drive, 1600)) - self.state.vecs["T4/T5"]) * k, 0.0)
        bC = np.tanh(self.W_bC @ t4)

        lob = np.maximum(self.state.vecs["лобула"] + (np.tanh(self.W_lob @ bB + self.W_lob2 @ bC) - self.state.vecs["лобула"]) * k, 0.0)
        loom = float(np.tanh((obs.get("size_dot") or 0.0) * 40.0)) if obs.get("lock") else 0.0
        lp = np.maximum(self.state.vecs["LPLC2"] + (np.tanh(self.W_lp @ bC + 0.6 * loom) - self.state.vecs["LPLC2"]) * k, 0.0)
        cen = np.tanh(self.state.vecs["центральный комплекс"] + (np.tanh(self.W_c @ bB) - self.state.vecs["центральный комплекс"]) * k)
        mb = np.maximum(self.state.vecs["грибовидное тело"] + (np.tanh(self.W_mb @ bB) - self.state.vecs["грибовидное тело"]) * k, 0.0)

        target_dn = np.tanh(self.W_dn @ self.hidden())
        self._dn = (1 - k) * self._dn + k * target_dn
        dn_pop = np.empty(200, dtype=np.float32)
        n_half = 100
        dn_pop[:n_half] = np.clip(self._dn[0] + self.rng.standard_normal(n_half).astype(np.float32) * 0.02, -1, 1)
        dn_pop[n_half:] = np.clip(self._dn[1] + self.rng.standard_normal(n_half).astype(np.float32) * 0.02, -1, 1)

        s["ламина"] = lam
        s["медулла"] = med
        s["T4/T5"] = t4
        s["лобула"] = lob
        s["LPLC2"] = lp
        s["центральный комплекс"] = cen
        s["грибовидное тело"] = mb
        s["DN"] = dn_pop

    def accel_cmd(self, v_m: Vec, n_max: float) -> Vec:
        from navedenie.seeker import body_axes

        _x, y, z = body_axes(v_m)
        n_pitch = float(self._dn[0]) * n_max * G * self.gain
        n_yaw = float(self._dn[1]) * n_max * G * self.gain
        a = z * n_pitch + y * n_yaw
        return clip_accel(_perp(a, v_m), n_max)

    def save(self, path: Path | None = None) -> Path:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = path or DATA_DIR / "weights_connectome.npz"
        np.savez(
            path,
            W_dn=self.W_dn,
            W_fx=self.W_fx,
            trained=np.array([1 if self.trained else 0], dtype=np.int8),
            gain=np.float64(self.gain),
            tau=np.float64(self.tau),
            seed=np.int64(self.seed),
            feat_schema=np.int64(FEATURE_SCHEMA_VERSION),
        )
        return path

    def load(self, path: Path | None = None) -> bool:
        """Загрузка с миграцией: схема v1 (W_fx 8 признаков) расширяется нулями
        на столбцах theta/rho; несовместимый файл → False с ясной семантикой."""
        path = path or DATA_DIR / "weights_connectome.npz"
        if not path.exists():
            path = DATA_DIR / "weights_stub.npz"
            if not path.exists():
                return False
        blob = np.load(path, allow_pickle=False)
        stored_schema = int(np.asarray(blob["feat_schema"]).reshape(-1)[0]) if "feat_schema" in blob else 1
        if stored_schema > FEATURE_SCHEMA_VERSION:
            return False  # мозг из более новой версии стенда
        w = blob["W_dn"]
        if w.shape != self.W_dn.shape:
            return False
        self.W_dn = w
        self.trained = bool(blob["trained"][0]) if "trained" in blob else True
        if "W_fx" in blob:
            wfx = blob["W_fx"]
            if wfx.ndim == 2 and wfx.shape[1] != self.W_fx.shape[1] and wfx.shape[1] in (LEGACY_FEAT_DIM, FEAT_DIM):
                wfx = migrate_feature_columns(wfx, axis=1, from_dim=wfx.shape[1])
            if wfx.shape == self.W_fx.shape:
                self.W_fx = wfx
        if "gain" in blob:
            self.gain = float(blob["gain"])
        if "tau" in blob:
            self.tau = float(blob["tau"])
        return True


class EnsembleBrain:
    """Ансамбль «три мозга»: команда управления — поэлементная МЕДИАНА команд
    членов (схема, полный, коннектом). Отказ или промах одного мозга не валит
    систему: медиана устойчива к одному выпадающему участнику."""

    def __init__(self, members: list) -> None:
        if not members:
            raise ValueError("ансамбль без мозгов")
        self.members = members
        self.kind = "ensemble"
        self.n_cells = int(sum(getattr(m, "n_cells", 0) for m in members))
        self.trained = all(bool(getattr(m, "trained", False)) for m in members)
        self.tau = float(getattr(members[0], "tau", 0.025))
        self.gain = float(getattr(members[0], "gain", 1.0))

    def reset(self) -> None:
        for m in self.members:
            m.reset()

    def step(self, obs: dict, dt: float) -> None:
        for m in self.members:
            m.step(obs, dt)

    @property
    def state(self):
        return self.members[0].state

    def accel_cmd(self, v_m: Vec, n_max: float) -> Vec:
        cmds = np.stack([np.asarray(m.accel_cmd(v_m, n_max), dtype=np.float64) for m in self.members])
        return np.median(cmds, axis=0)
