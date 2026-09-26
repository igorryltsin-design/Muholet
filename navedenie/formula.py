"""Восстановление формулы закона наведения из обученного мозга.

Управляющий путь мухи аналитичен: признаки (пеленги, их скорости, угловой
размер, фаза сближения, поток, захват) → скрытый слой → W_dn → тангаж/рыскание.
Модуль даёт:

1. Точную аналитическую форму для КАЖДОГО мозга — сложность формулы растёт
   вместе с мозгом: схема — линейный выход по 10 признакам; полный — перцептрон
   10→4096→32→2; коннектом — 128-канальный readout (синтетические каналы,
   проводка регионов — FlyWire).
2. ПОЛИНОМИАЛЬНЫЙ СУРРОГАТ МГНОВЕННОГО READOUT по текущим сенсорным признакам.
   Это НЕ полная динамическая формула мозга: она не включает состояния нейронов
   (фильтры tau), задержки ГСН, историю изображений и динамику захвата — только
   статический отклик dn(feat) в моменте. Фит: primary — flight-domain (признаки
   из реальных прогонов, разбиение ПО ЭПИЗОДАМ), global-сетка [-1,1]^d —
   дополнительный stress-test.
3. Локальные усиления K_az = ∂yaw/∂ω_az и K_el = ∂pitch/∂ω_el (в физических
   единицах на рад/с) и оценку N_poly_eff ≈ K_ω·(n_max·g)/V_m — при ЯВНО
   выбранной скорости ракеты: скорость не входит в полином, поэтому
   универсального N из суррогата не следует.
4. Терминологию русскоязычной литературы по самонаведению.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from navedenie.brain_store import get_circuit
from navedenie.circuit import CHANNEL_ZONES, FEAT_DIM, FEATURE_SCHEMA_VERSION, K_RHO, K_THETA
from navedenie.sim import G

# Признаки контура v2 в терминах русскоязычной литературы по самонаведению:
# key — латинский идентификатор для выгружаемого кода, symbol — обозначение в формулах.
FEATURE_SPEC = [
    {"key": "beta3", "symbol": "3β", "ru": "пеленг цели по азимуту, ×3",
     "term": "утроенный угол (пеленг) линии визирования в горизонтальной плоскости"},
    {"key": "lam3", "symbol": "3λ", "ru": "пеленг цели по углу места, ×3",
     "term": "утроенный угол места линии визирования (вертикальная плоскость)"},
    {"key": "wb", "symbol": "0.4·ωβ", "ru": "скорость пеленга по азимуту, ×0.4",
     "term": "угловая скорость вращения линии визирования по азимуту (ωβ)"},
    {"key": "wl", "symbol": "0.4·ωλ", "ru": "скорость пеленга по углу места, ×0.4",
     "term": "угловая скорость вращения линии визирования по углу места (ωλ)"},
    {"key": "theta", "symbol": f"θ·{K_THETA:g}", "ru": "измеренный угловой размер цели, ×" + f"{K_THETA:g}",
     "term": "угловой размер пятна цели (рад), декодирован из изображения калибровкой PSF сетчатки; θ→0, пока цель не разрешима"},
    {"key": "loom", "symbol": "loom", "ru": "скорость роста яркости пятна (лоом)",
     "term": "«ломб» — скорость увеличения видимого размера/яркости цели, признак сближения"},
    {"key": "rho", "symbol": f"ρ·{K_RHO:g}", "ru": "фаза сближения ρ = θ̇/θ, ×" + f"{K_RHO:g}",
     "term": "нормированная скорость расширения изображения — оптическая оценка Vc/R ≈ 1/t_go"},
    {"key": "flow_x", "symbol": "4Φx", "ru": "оптический поток по горизонтали, ×4",
     "term": "ретинальный (зрительный) поток по горизонтальной оси сетчатки"},
    {"key": "flow_y", "symbol": "4Φy", "ru": "оптический поток по вертикали, ×4",
     "term": "ретинальный (зрительный) поток по вертикальной оси сетчатки"},
    {"key": "lock", "symbol": "[захв]", "ru": "признак сопровождения цели",
     "term": "1 — цель в кадре ГСН и сопровождается, 0 — захват отсутствует"},
]
FEATURE_KEYS = [f["key"] for f in FEATURE_SPEC]
FEATURE_RU = [f["ru"] for f in FEATURE_SPEC]
FEATURE_SYMBOLS = [f["symbol"] for f in FEATURE_SPEC]
IDX_WB = FEATURE_KEYS.index("wb")
IDX_WL = FEATURE_KEYS.index("wl")

# степень полинома-суррогата растёт со сложностью мозга
FORMULA_DEGREE = {"stub": 1, "full": 2, "connectome": 3}

SURROGATE_TITLE = "Полиномиальный суррогат мгновенного readout по текущим сенсорным признакам"
SURROGATE_NOT_INCLUDED = (
    "суррогат НЕ является полной динамической формулой мозга: в него не входят "
    "состояния нейронов (фильтры с tau), задержка и джиттер ГСН, история "
    "изображений и динамика захвата"
)


def _poly_design(X: np.ndarray, degree: int) -> tuple[np.ndarray, list[tuple[str, str, str]]]:
    """Дизайн-матрица ПОЛНОГО полинома заданной степени: мономы с повторяющимися
    множителями (x²y, x³, xyz…). Число столбцов = C(d+degree, degree)."""
    from itertools import combinations_with_replacement

    sup = {2: "²", 3: "³", 4: "⁴"}
    cols: list[np.ndarray] = [np.ones(X.shape[0])]
    terms: list[tuple[str, str, str]] = [("1", "1", "свободный член")]
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(X.shape[1]), d):
            col = np.ones(X.shape[0])
            for j in combo:
                col = col * X[:, j]
            cols.append(col)
            counts: dict[int, int] = {}
            for j in combo:
                counts[j] = counts.get(j, 0) + 1
            key_parts = [
                "·".join([FEATURE_KEYS[j]] * e) for j, e in counts.items()
            ]
            sym_parts = [
                FEATURE_SYMBOLS[j] + (sup[e] if 1 < e <= 4 else f"^{e}" if e > 4 else "")
                for j, e in counts.items()
            ]
            ru_parts = [
                FEATURE_RU[j] + (sup[e] if 1 < e <= 4 else f"^{e}" if e > 4 else "")
                for j, e in counts.items()
            ]
            terms.append(("·".join(key_parts), "·".join(sym_parts), " × ".join(ru_parts)))
    return np.stack(cols, axis=1), terms


def _fit_metrics(coef: np.ndarray, A: np.ndarray, y: np.ndarray) -> dict[str, float]:
    err = A @ coef - y
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) or 1.0
    return {
        "r2": round(1.0 - ss_res / ss_tot, 4),
        "err_mean": round(float(np.mean(np.abs(err))), 4),
        "err_max": round(float(np.max(np.abs(err))), 4),
    }


def _fit_pruned(X: np.ndarray, y: np.ndarray, prune: float = 0.04) -> tuple[np.ndarray, dict[str, float]]:
    """МНК-фит с прореживанием мелких коэффициентов (hold-out по хвосту выборки).
    Возвращает (веса, метрики погрешности на отложенной части)."""
    split = int(X.shape[0] * 0.75)
    coef, *_ = np.linalg.lstsq(X[:split], y[:split], rcond=None)
    thresh = prune * float(np.max(np.abs(coef))) if coef.size else 0.0
    coef = np.where(np.abs(coef) < thresh, 0.0, coef)
    return coef, _fit_metrics(coef, X[split:], y[split:])


def _term_table(coef: np.ndarray, terms: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
    order = np.argsort(-np.abs(coef))
    return [
        {"key": terms[i][0], "symbol": terms[i][1], "ru": terms[i][2], "coef": round(float(coef[i]), 4)}
        for i in order
        if abs(coef[i]) > 1e-6
    ]


def _formula_ru(coef: np.ndarray, terms: list[tuple[str, str, str]]) -> str:
    parts = [f"{c:+.3f}·{sym}" for c, (_k, sym, _r) in zip(coef, terms) if abs(c) > 1e-6]
    inner = " + ".join(parts).replace("+-", "− ")
    return inner or "0"


def _formula_code(coef: np.ndarray, terms: list[tuple[str, str, str]]) -> str:
    parts = [f"{c:+.4f}*{key}" for c, (key, _s, _r) in zip(coef, terms) if abs(c) > 1e-6]
    return " + ".join(parts) or "0.0"


# ── flight-domain: признаки из реальных прогонов, разбиение по эпизодам ──────

FLIGHT_SEED = 77
FLIGHT_EPISODES = 8
FLIGHT_OFFSET = 3  # смещение train-жребия: эпизоды не совпадают с валидацией


def collect_flight_features(circuit, episodes: int = FLIGHT_EPISODES) -> list[np.ndarray]:
    """Признаки контура из РЕАЛЬНЫХ прогонов на train-сценариях протокола
    (одна запись на шаг; эпизоды не смешиваются — разбиение целиком по ним)."""
    from navedenie.train import _episode_scenario, _rollout_update

    per_ep: list[np.ndarray] = []
    for i in range(episodes):
        ep = FLIGHT_OFFSET + i * 2
        sc = _episode_scenario(ep, getattr(circuit, "gain", 1.0))
        feats: list[np.ndarray] = []
        _rollout_update(circuit, sc, lr=0.0, feats=feats)
        if feats:
            per_ep.append(np.stack(feats))
    return per_ep


def _zone_errors(coef: np.ndarray, A: np.ndarray, X: np.ndarray, dn_ch: np.ndarray) -> dict[str, Any]:
    """Ошибка суррогата по зонам: центр/периферия сетчатки, малое/большое rho."""
    err = np.abs(A @ coef - dn_ch)
    az_feat = X[:, 0]
    el_feat = X[:, 1]
    central = (np.abs(az_feat) <= 0.6) & (np.abs(el_feat) <= 0.6)  # |az|,|el| ≲ 12°
    rho_feat = X[:, FEATURE_KEYS.index("rho")]
    rho_med = float(np.median(rho_feat))
    low_rho = rho_feat < rho_med
    out: dict[str, Any] = {
        "central_mae": round(float(np.mean(err[central])), 4) if central.any() else None,
        "peripheral_mae": round(float(np.mean(err[~central])), 4) if (~central).any() else None,
        "low_rho_mae": round(float(np.mean(err[low_rho])), 4) if low_rho.any() else None,
        "high_rho_mae": round(float(np.mean(err[~low_rho])), 4) if (~low_rho).any() else None,
        "rho_split": round(rho_med, 4),
    }
    return out


def _flight_fit(kind: str, circuit, degree: int, per_ep: list[np.ndarray] | None = None) -> dict[str, Any] | None:
    """Flight-domain фит: эпизоды целиком в train/validation, МНК без прореживания
    (реальные признаки сосредоточены в узкой области), честные метрики на
    held-out эпизодах + ошибки по зонам и диапазоны признаков."""
    if per_ep is None:
        per_ep = collect_flight_features(circuit)
    if len(per_ep) < 4:
        return None
    rng = np.random.default_rng(FLIGHT_SEED)
    order = rng.permutation(len(per_ep))
    n_val = max(2, len(per_ep) // 4)
    val_idx = set(order[:n_val].tolist())
    X_train = np.vstack([per_ep[i] for i in range(len(per_ep)) if i not in val_idx])
    X_val = np.vstack([per_ep[i] for i in sorted(val_idx)])
    A_tr, _terms = _poly_design(X_train, degree)
    A_val, _ = _poly_design(X_val, degree)
    dn_tr = brain_dn(kind, circuit, X_train)
    dn_val = brain_dn(kind, circuit, X_val)
    fits: dict[str, dict[str, Any]] = {}
    for k, ch in enumerate(("pitch", "yaw")):
        coef, *_ = np.linalg.lstsq(A_tr, dn_tr[:, k], rcond=None)
        fits[ch] = {
            "coef": coef,
            "metrics": _fit_metrics(coef, A_val, dn_val[:, k]),
            "zones": _zone_errors(coef, A_val, X_val, dn_val[:, k]),
        }
    return {
        "sample": "flight",
        "n_episodes": len(per_ep),
        "n_episodes_train": len(per_ep) - len(val_idx),
        "n_episodes_val": len(val_idx),
        "seed": FLIGHT_SEED,
        "n_points_train": int(X_train.shape[0]),
        "n_points_val": int(X_val.shape[0]),
        "feature_ranges": [
            [round(float(X_train[:, j].min()), 4), round(float(X_train[:, j].max()), 4)]
            for j in range(X_train.shape[1])
        ],
        "fits": fits,
    }


def _grid_surrogate(kind: str, circuit, degree: int, per_ep: list[np.ndarray] | None = None) -> tuple[int, list[tuple[str, str, str]], dict[str, tuple[np.ndarray, dict[str, float]]], dict[str, Any] | None]:
    """Глобальная сетка [-1,1]^d — прежний стресс-тест (физически невозможные
    комбинации признаков включены). Возвращает степень, мономы, фиты и
    flight-домен (если собран)."""
    rng = np.random.default_rng(7)
    X = rng.uniform(-1.0, 1.0, size=(6000, FEAT_DIM))
    deg = degree
    dn = brain_dn(kind, circuit, X)
    A, terms = _poly_design(X, deg)
    split = int(A.shape[0] * 0.75)
    fits: dict[str, tuple[np.ndarray, dict[str, float]]] = {}
    for k, ch in enumerate(("pitch", "yaw")):
        fits[ch] = _fit_pruned(A[:split], dn[:split, k], prune=0.04)
    flight = None
    if per_ep and len(per_ep) >= 4:
        flight = _flight_fit(kind, circuit, deg, per_ep=per_ep)
    return deg, terms, fits, flight


def _surrogate(kind: str, circuit, degree: int | None = None):
    """Единый суррогат «мозг → полином»: flight-domain как основной фит +
    глобальная сетка как stress-test. Возвращает (degree, terms, fits, flight)."""
    per_ep = None
    try:
        per_ep = collect_flight_features(circuit)
    except Exception:  # noqa: BLE001 — без прогонов остаётся глобальная сетка
        per_ep = None
    deg = max(1, min(int(degree), 4)) if degree else FORMULA_DEGREE.get(kind, 2)
    return _grid_surrogate(kind, circuit, deg, per_ep=per_ep)


def local_gains(kind: str, circuit, v_m_values: tuple[float, ...] = (650.0, 780.0, 950.0), n_max: float = 30.0, gain: float = 1.0) -> dict[str, Any]:
    """Локальные усиления суррогата по угловым скоростям в опорной точке
    (захват есть, пеленг/поток ≈ 0 — малый сигнал):

    K_az_feat = ∂yaw/∂feat[ωβ] (в единицах признака на единицу признака);
    пересчёт в физику: feat_ω = tanh(0.4·ω) ≈ 0.4·ω, команда (до клипа)
    a = out·n_max·g·gain ⇒ K_ω [м/с² на рад/с] ≈ ∂out/∂feat·0.4·n_max·g·gain;
    N_poly_eff(V_m) = K_ω / V_m — размерностно как коэффициент ПН.

    Скорость ракеты НЕ входит в полином, поэтому N_poly_eff показывается
    ТОЛЬКО при явно выбранной V_m (таблицей), а не как универсальное N."""
    d = FEAT_DIM
    base = np.zeros(d)
    base[FEATURE_KEYS.index("lock")] = 1.0
    eps = 1e-4
    out0 = brain_dn(kind, circuit, base[None, :])[0]

    def _deriv(idx: int, ch: int) -> float:
        xp = base.copy()
        xp[idx] += eps
        xm = base.copy()
        xm[idx] -= eps
        # признак ограничен танхом — на малом сигнале производная честная
        return float((brain_dn(kind, circuit, xp[None, :])[0][ch] - brain_dn(kind, circuit, xm[None, :])[0][ch]) / (2 * eps))

    k_wb_feat = _deriv(IDX_WB, 1)  # ∂yaw/∂feat[ωβ]
    k_wl_feat = _deriv(IDX_WL, 0)  # ∂pitch/∂feat[ωλ]
    scale = 0.4 * n_max * G * gain  # признак→рад/с и доля→м/с²
    k_wb_phys = k_wb_feat * scale
    k_wl_phys = k_wl_feat * scale
    return {
        "point": "lock=1, остальные признаки 0 (малый сигнал)",
        "k_az_feat": round(k_wb_feat, 5),
        "k_el_feat": round(k_wl_feat, 5),
        "k_az_phys": round(k_wb_phys, 4),  # м/с² на рад/с
        "k_el_phys": round(k_wl_phys, 4),
        "n_poly_eff": [
            {"v_m": vm, "n_az": round(k_wb_phys / vm, 4), "n_el": round(k_wl_phys / vm, 4)}
            for vm in v_m_values
        ],
        "note": (
            "N_poly_eff = K_ω/V_m — корректен только при указанной V_m и вне "
            "насыщения; это свойство суррогата, не универсальная постоянная ПН"
        ),
    }


def _term_exponents(key: str) -> tuple[int, ...]:
    """Вектор степеней признаков для монома по его ключу («beta3·beta3·flow_y» → (2,0,0,0,0,0,0,1,0,0))."""
    if key == "1":
        return (0,) * len(FEATURE_KEYS)
    exps = [0] * len(FEATURE_KEYS)
    for factor in key.split("·"):
        exps[FEATURE_KEYS.index(factor)] += 1
    return tuple(exps)


def _approx_python_block(kind: str, circuit, degree: int | None = None) -> str:
    """Хвост выгружаемого файла: приближение мозга полиномом (исполняемое) + его погрешность."""
    degree, terms, fits, _flight = _surrogate(kind, circuit, degree)
    lines = [
        "",
        "",
        f"# ── Часть 2: {SURROGATE_TITLE} ──",
        f"# Полином степени {degree}, пригнанный к отклику dn(feat); МНК с прореживанием",
        "# (|коэффициент| < 4% максимума обнулён). Это мгновенный readout: состояния",
        "# нейронов (tau), задержки ГСН и история кадров в формулу НЕ входят.",
        "# Отклонение от точного dn(feat) на отложенной сетке (единицы команды −1…1):",
    ]
    for ch, ru in (("pitch", "тангаж"), ("yaw", "рыскание")):
        _coef, m = fits[ch]
        lines.append(
            f"#   {ru:<8} R² {m['r2']:.4f} · средняя |ошибка| {m['err_mean']:.4f} · максимальная {m['err_max']:.4f}"
        )
    lines.append("#")
    for ch, ru in (("pitch", "тангаж "), ("yaw", "рыскание")):
        coef, _m = fits[ch]
        lines.append(f"# {ru} ≈ {_formula_ru(coef, terms)}")
    lines += [
        "#",
        "# Член списка: (коэффициент, степени признаков в порядке " + ", ".join(FEATURE_KEYS) + ")",
        f"APPROX_DEGREE = {degree}",
    ]
    for ch, name in (("pitch", "APPROX_PITCH"), ("yaw", "APPROX_YAW")):
        coef, _m = fits[ch]
        lines.append(f"{name} = [")
        for c, (key, sym, _r) in zip(coef, terms):
            if abs(c) < 1e-6:
                continue
            es = ", ".join(str(e) for e in _term_exponents(key))
            lines.append(f"    ({c:+.6f}, ({es})),  # {key} ({sym})")
        lines.append("]")
    lines += [
        "",
        "",
        "def dn_approx(feat):",
        "    'Суррогат мгновенного readout: (тангаж, рыскание); может слегка выйти за −1…1.'",
        "    def _sum(trms):",
        "        s = 0.0",
        "        for c, es in trms:",
        "            m = c",
        "            for x, e in zip(feat, es):",
        "                if e:",
        "                    m *= x ** e",
        "            s += m",
        "        return s",
        "    return _sum(APPROX_PITCH), _sum(APPROX_YAW)",
        "",
        "",
        "def dn_local_gain(feat, v_m=None, n_max=30.0, gain=1.0):",
        "    'Локальный коэффициент при угловой скорости: d(yaw)/d(wb), d(pitch)/d(wl)'",
        "    'и, если задана v_m, оценка N_poly_eff = K_ω/V_m (корректна только при'",
        "    'явно выбранной V_m — скорость ракеты в полином не входит).'",
        "    eps = 1e-5",
        "    def _out(f):",
        "        p, y = dn_approx(f)",
        "        return p, y",
        "    fp = list(feat); fp[2] += eps",
        "    fm = list(feat); fm[2] -= eps",
        "    k_az = (_out(fm)[1] - _out(fp)[1]) / (2 * eps)  # d(yaw)/d(feat_wb)",
        "    fq_p = list(feat); fq_p[3] += eps",
        "    fq_m = list(feat); fq_m[3] -= eps",
        "    k_el = (_out(fq_p)[0] - _out(fq_m)[0]) / (2 * eps)  # d(pitch)/d(feat_wl)",
        "    scale = 0.4 * n_max * gain",
        "    out = {'k_az_feat': k_az, 'k_el_feat': k_el,",
        "           'k_az_phys': k_az * scale, 'k_el_phys': k_el * scale}",
        "    if v_m:",
        "        out['n_poly_eff'] = {'v_m': v_m, 'n_az': k_az * scale / v_m,",
        "                             'n_el': k_el * scale / v_m}",
        "    return out",
        "",
    ]
    return "\n".join(lines)


def brain_dn(kind: str, circuit, X: np.ndarray) -> np.ndarray:
    """Точный отклик мозга (N×2: тангаж, рыскание) на сетке признаков X (N×10)."""
    if kind == "connectome":
        H = np.tanh(circuit.W_fx @ X.T + circuit.b_fx[:, None])
        return np.tanh(circuit.W_dn @ H).T
    if kind == "full":
        KN = np.maximum(circuit.W_k @ X.T, 0.0)
        KN = KN / (np.linalg.norm(KN, axis=0, keepdims=True) + 1e-6)
        HID = np.tanh(circuit.W_pool @ KN)
        return np.tanh(circuit.W_dn @ HID).T
    return np.tanh(circuit.W_dn @ X.T).T


def brain_formula(kind: str, degree: int | None = None) -> dict[str, Any]:
    """Формула закона наведения обученного мозга: точный вид, сложность по мозгу,
    полиномиальный суррогат мгновенного readout и его погрешность. Основной
    фит — flight-domain (признаки реальных прогонов, разбиение по эпизодам);
    глобальная сетка [-1,1]^d — дополнительный стресс-тест."""
    circuit = get_circuit(kind)
    degree, terms, fits, flight = _surrogate(kind, circuit, degree)

    poly_out: dict[str, Any] = {
        "title": SURROGATE_TITLE,
        "not_included": SURROGATE_NOT_INCLUDED,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "degree": degree,
        "r2_pitch": None,
        "r2_yaw": None,
        "formula_ru_pitch": "",
        "formula_ru_yaw": "",
        "formula_code_pitch": "",
        "formula_code_yaw": "",
        "terms_pitch": [],
        "terms_yaw": [],
        "err_mean_pitch": None,
        "err_max_pitch": None,
        "err_mean_yaw": None,
        "err_max_yaw": None,
        "domain": "flight",
        "flight": None,
        "grid_stress": None,
    }
    for ch in ("pitch", "yaw"):
        coef, metr = fits[ch]
        poly_out[f"r2_{ch}"] = metr["r2"]
        poly_out[f"err_mean_{ch}"] = metr["err_mean"]
        poly_out[f"err_max_{ch}"] = metr["err_max"]
        poly_out[f"formula_ru_{ch}"] = _formula_ru(coef, terms)
        poly_out[f"formula_code_{ch}"] = _formula_code(coef, terms)
        poly_out[f"terms_{ch}"] = _term_table(coef, terms)[:10]
    if flight is not None:
        poly_out["domain"] = "flight"
        poly_out["flight"] = {
            "sample": "признаки реальных прогонов на train-сценариях протокола; эпизод целиком в train или validation",
            "n_episodes": flight["n_episodes"],
            "n_episodes_train": flight["n_episodes_train"],
            "n_episodes_val": flight["n_episodes_val"],
            "n_points_train": flight["n_points_train"],
            "n_points_val": flight["n_points_val"],
            "seed": flight["seed"],
            "feature_ranges": flight["feature_ranges"],
            "metrics": {
                ch: {
                    "r2": flight["fits"][ch]["metrics"]["r2"],
                    "mae": flight["fits"][ch]["metrics"]["err_mean"],
                    "err_max": flight["fits"][ch]["metrics"]["err_max"],
                    "zones": flight["fits"][ch]["zones"],
                }
                for ch in ("pitch", "yaw")
            },
        }
        # R² и ошибки в шапке — теперь от flight-домена (основной фит)
        for ch in ("pitch", "yaw"):
            m = flight["fits"][ch]["metrics"]
            poly_out[f"r2_{ch}"] = m["r2"]
            poly_out[f"err_mean_{ch}"] = m["err_mean"]
            poly_out[f"err_max_{ch}"] = m["err_max"]
        coef_yaw = flight["fits"]["yaw"]["coef"]
        coef_pitch = flight["fits"]["pitch"]["coef"]
        poly_out["formula_ru_pitch"] = _formula_ru(coef_pitch, terms)
        poly_out["formula_ru_yaw"] = _formula_ru(coef_yaw, terms)
        poly_out["formula_code_pitch"] = _formula_code(coef_pitch, terms)
        poly_out["formula_code_yaw"] = _formula_code(coef_yaw, terms)
        poly_out["terms_pitch"] = _term_table(coef_pitch, terms)[:10]
        poly_out["terms_yaw"] = _term_table(coef_yaw, terms)[:10]
    else:
        poly_out["domain"] = "grid"
    poly_out["grid_stress"] = {
        "sample": "глобальная сетка [-1,1]^10 — стресс-тест, физически невозможные комбинации включены",
        "r2_pitch": fits["pitch"][1]["r2"],
        "r2_yaw": fits["yaw"][1]["r2"],
    }
    poly_out["local_gains"] = local_gains(kind, circuit, gain=float(getattr(circuit, "gain", 1.0)))

    exact_stub_formula = [
        f"тангаж = tanh({_formula_ru(circuit.W_dn[0], list(zip(FEATURE_KEYS, FEATURE_SYMBOLS, FEATURE_RU)))})",
        f"рыскание = tanh({_formula_ru(circuit.W_dn[1], list(zip(FEATURE_KEYS, FEATURE_SYMBOLS, FEATURE_RU)))})",
    ]
    return {
        "kind": kind,
        "n_cells": circuit.n_cells,
        "trained": bool(circuit.trained),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_spec": FEATURE_SPEC,
        "degree": degree,
        "exact": {
            "stub": {
                "вид": f"линейный выход по {FEAT_DIM} признакам (через тангенс)",
                "dn_формула": exact_stub_formula,
            },
            "full": {
                "вид": f"перцептрон Розенблатта: {FEAT_DIM} → relu(W_k·x), нормировка Кеньона → 4096 → tanh(W_pool··) → 32 → W_dn → 2",
                "dn_формула": "тангаж/рыскание = tanh(W_dn · tanh(W_pool · norm(relu(W_k · признаки))))",
            },
            "connectome": {
                "вид": (
                    f"{getattr(circuit, 'channels', '?')}-канальный синтетический readout (проводка регионов — FlyWire): "
                    "tanh(W_dn · tanh(W_fx · признаки + b_fx)); зоны каналов: "
                    + ", ".join(
                        f"{name} {sum(1 for z in getattr(circuit, 'channel_zones', []) if z == name)}"
                        for name, _n in CHANNEL_ZONES
                    )
                ),
                "dn_формула": "тангаж/рыскание = tanh(W_dn · tanh(W_fx · признаки + b_fx))",
                "зоны_каналов": [
                    {"зона": name, "каналов": sum(1 for z in getattr(circuit, "channel_zones", []) if z == name)}
                    for name, _n in CHANNEL_ZONES
                ],
            },
        }[kind],
        "python_url": f"/api/brain/formula/python?kind={kind}",
        "c_url": f"/api/brain/formula/c?kind={kind}",
        "export_python": brain_formula_python(kind),
        "poly": poly_out,
    }


def brain_formula_python(kind: str, degree: int | None = None) -> str:
    """Исполняемый Python-файл: формула мгновенного readout (часть 1) + полиномиальный
    суррогат мгновенного readout (часть 2).

    Readout — это отклик сети на 10 признаков ГСН; состояния нейронов, задержки
    и история кадров в экспорт не входят. Для коннектома W_fx — синтетические
    каналы, а не проводка FlyWire."""
    circuit = get_circuit(kind)
    legend = "\n".join(
        f"#   {f['key']:<7} = {f['symbol']:<8} — {f['ru']} ({f['term']})" for f in FEATURE_SPEC
    )
    kind_title = {"full": "полный", "connectome": "коннектом"}.get(kind, "схема")
    head = (
        f'"""Закон наведения мухи (мозг «{kind_title}», МУХОЛЁТ).\n\n'
        "Часть 1 — ТОЧНАЯ формула: dn(feat) — та же сеть, что летает в стенде\n"
        "           (мгновенный readout: без состояний нейронов, задержек и истории).\n"
        "Часть 2 — ПОЛИНОМИАЛЬНЫЙ СУРРОГАТ мгновенного readout: dn_approx(feat) —\n"
        "           полином, пригнанный к отклику dn (МНК с прореживанием мелких\n"
        "           членов); погрешность — в комментарии рядом. Это НЕ полная\n"
        "           динамическая формула мозга.\n\n"
        f"Признаки (входы, все нормированы в −1…1; схема признаков v{FEATURE_SCHEMA_VERSION}, {len(FEATURE_KEYS)} шт.):\n" + legend + "\n"
        "Команды тангаж/рыскание — тоже в −1…1.\n"
        '"""\n'
        "import math\n\n\n"
    )
    if kind == "full":
        wk_rows = "\n".join("  [" + ", ".join(f"{v:.6f}" for v in row) + "]," for row in circuit.W_k)
        wp_rows = "\n".join("  [" + ", ".join(f"{v:.6f}" for v in row) + "]," for row in circuit.W_pool)
        wd_rows = "\n".join("  [" + ", ".join(f"{v:.6f}" for v in row) + "]," for row in circuit.W_dn)
        body = (
            f"W_K = [\n{wk_rows}\n]  # случайная фикс-проводка 4096×{FEAT_DIM} (не обучается)\n"
            f"W_POOL = [\n{wp_rows}\n]  # скрытый слой 32×4096 (не обучается)\n"
            f"W_DN = [\n{wd_rows}\n]  # обучаемый выход 2×32\n\n\n"
            "def dn(feat):\n"
            f"    'feat — {FEAT_DIM} признаков. Возврат: (тангаж, рыскание) в −1…1.'\n"
            "    k = [max(0.0, sum(w * x for w, x in zip(row, feat))) for row in W_K]\n"
            "    norm = math.sqrt(sum(v * v for v in k)) + 1e-6\n"
            "    kn = [v / norm for v in k]\n"
            "    hidden = [math.tanh(sum(w * v for w, v in zip(row, kn))) for row in W_POOL]\n"
            "    pitch = math.tanh(sum(w * h for w, h in zip(W_DN[0], hidden)))\n"
            "    yaw = math.tanh(sum(w * h for w, h in zip(W_DN[1], hidden)))\n"
            "    return pitch, yaw\n"
        )
    elif kind == "connectome":
        wfx_rows = "\n".join("  [" + ", ".join(f"{v:.6f}" for v in row) + "]," for row in circuit.W_fx)
        bfx_row = ", ".join(f"{float(v):.6f}" for v in circuit.b_fx)
        wd_rows = "\n".join("  [" + ", ".join(f"{v:.6f}" for v in row) + "]," for row in circuit.W_dn)
        body = (
            f"W_FX = [\n{wfx_rows}\n]  # синтетические каналы readout {FEAT_DIM}→{circuit.W_fx.shape[0]}\n"
            f"B_FX = [{bfx_row}]\n"
            f"W_DN = [\n{wd_rows}\n]  # обучаемый выход 2×{circuit.W_dn.shape[1]}\n\n\n"
            "def dn(feat):\n"
            f"    'feat — {FEAT_DIM} признаков. Возврат: (тангаж, рыскание) в −1…1.'\n"
            "    hidden = [math.tanh(sum(w * x for w, x in zip(row, feat)) + b) for row, b in zip(W_FX, B_FX)]\n"
            "    pitch = math.tanh(sum(w * h for w, h in zip(W_DN[0], hidden)))\n"
            "    yaw = math.tanh(sum(w * h for w, h in zip(W_DN[1], hidden)))\n"
            "    return pitch, yaw\n"
        )
    else:
        w_rows = "\n".join(
            "  [" + ", ".join(f"{v:.6f}" for v in row) + "]," for row in circuit.W_dn
        )
        body = (
            f"W_DN = [\n{w_rows}\n]  # обучаемый линейный выход 2×{FEAT_DIM}\n\n\n"
            "def dn(feat):\n"
            f"    'feat — {FEAT_DIM} признаков. Возврат: (тангаж, рыскание) в −1…1.'\n"
            "    pitch = math.tanh(sum(w * x for w, x in zip(W_DN[0], feat)))\n"
            "    yaw = math.tanh(sum(w * x for w, x in zip(W_DN[1], feat)))\n"
            "    return pitch, yaw\n"
        )
    return head + body + _approx_python_block(kind, circuit, degree)


def _c_matrix(name: str, arr: np.ndarray, comment: str) -> str:
    """Двумерный массив float в C: статический const, по строкам."""
    rows = "\n".join(
        "  {" + ", ".join(f"{v:.6f}f" for v in row) + "}," for row in arr
    )
    return f"static const float {name}[{arr.shape[0]}][{arr.shape[1]}] = {{\n{rows}\n}};  // {comment}\n"


def brain_formula_c(kind: str, degree: int | None = None) -> str:
    """Чистый C99 (только math.h): формула мгновенного readout dn() + полиномиальный
    суррогат dn_approx().

    Файл самодостаточен для внешнего симулятора/железа: без numpy, без API стенда.
    Блок main под макросом MUHOLET_MAIN — для проверки: читает 10 признаков, печатает команды."""
    circuit = get_circuit(kind)
    _deg, terms, fits, _flight = _surrogate(kind, circuit, degree)
    kind_title = {"full": "полный", "connectome": "коннектом"}.get(kind, "схема")
    legend = "\n".join(
        f"   {f['key']:<7} = {f['symbol']:<8} — {f['ru']}" for f in FEATURE_SPEC
    )
    head = (
        f"/* Закон наведения мухи — readout мозга «{kind_title}» (МУХОЛЁТ) — чистый C99.\n"
        " * dn(feat, out)      — точная сеть, та же, что летает в стенде (мгновенный\n"
        " *                       readout: без состояний нейронов/задержек/истории);\n"
        " * dn_approx(feat, out) — ПОЛИНОМИАЛЬНЫЙ СУРРОГАТ мгновенного readout\n"
        f" *                       (МНК, степень {_deg}; НЕ полная динамика мозга);\n"
        " *   погрешность на отложенной сетке (единицы команды −1…1):\n"
    )
    for ch, ru in (("pitch", "тангаж"), ("yaw", "рыскание")):
        _coef, m = fits[ch]
        head += f" *     {ru:<7} R2 {m['r2']:.4f} · средняя |ошибка| {m['err_mean']:.4f} · макс {m['err_max']:.4f}\n"
    head += (
        " *\n"
        f" * Признаки (входы, нормированы в −1…1; схема v{FEATURE_SCHEMA_VERSION}, {FEAT_DIM} шт.):\n" + legend + "\n"
        " * out[0] — тангаж, out[1] — рыскание, оба в −1…1.\n"
        " */\n"
        "#include <math.h>\n\n"
    )

    if kind == "connectome":
        n = circuit.W_fx.shape[0]
        body = (
            _c_matrix("W_FX", circuit.W_fx, f"синаптические каналы {FEAT_DIM}→{n}")
            + "static const float B_FX[%d] = {%s};\n\n"
            % (n, ", ".join(f"{float(v):.6f}f" for v in circuit.b_fx))
            + _c_matrix("W_DN", circuit.W_dn, "обучаемый выход")
            + """
void dn(const float feat[%d], float out[2]) {
    float hidden[%d];
    for (int i = 0; i < %d; i++) {
        float s = B_FX[i];
        for (int j = 0; j < %d; j++) s += W_FX[i][j] * feat[j];
        hidden[i] = tanhf(s);
    }
    for (int ch = 0; ch < 2; ch++) {
        float s = 0.0f;
        for (int i = 0; i < %d; i++) s += W_DN[ch][i] * hidden[i];
        out[ch] = tanhf(s);
    }
}
""" % (FEAT_DIM, n, n, FEAT_DIM, circuit.W_dn.shape[1])
        )
    elif kind == "full":
        nk, npool = circuit.W_k.shape[0], circuit.W_pool.shape[0]
        body = (
            _c_matrix("W_K", circuit.W_k, "случайная фикс-проводка (не обучается)")
            + _c_matrix("W_POOL", circuit.W_pool, "скрытый слой (не обучается)")
            + _c_matrix("W_DN", circuit.W_dn, "обучаемый выход")
            + """
void dn(const float feat[%d], float out[2]) {
    float k[%d], kn[%d], hidden[%d];
    float norm = 1e-6f;
    for (int i = 0; i < %d; i++) {
        float s = 0.0f;
        for (int j = 0; j < %d; j++) s += W_K[i][j] * feat[j];
        k[i] = s > 0.0f ? s : 0.0f;
        norm += k[i] * k[i];
    }
    norm = sqrtf(norm);
    for (int i = 0; i < %d; i++) kn[i] = k[i] / norm;
    for (int i = 0; i < %d; i++) {
        float s = 0.0f;
        for (int j = 0; j < %d; j++) s += W_POOL[i][j] * kn[j];
        hidden[i] = tanhf(s);
    }
    for (int ch = 0; ch < 2; ch++) {
        float s = 0.0f;
        for (int i = 0; i < %d; i++) s += W_DN[ch][i] * hidden[i];
        out[ch] = tanhf(s);
    }
}
""" % (FEAT_DIM, nk, nk, npool, nk, FEAT_DIM, nk, npool, nk, circuit.W_dn.shape[1])
        )
    else:
        body = _c_matrix("W_DN", circuit.W_dn, f"обучаемый линейный выход 2×{FEAT_DIM}") + """
void dn(const float feat[%d], float out[2]) {
    for (int ch = 0; ch < 2; ch++) {
        float s = 0.0f;
        for (int j = 0; j < %d; j++) s += W_DN[ch][j] * feat[j];
        out[ch] = tanhf(s);
    }
}
""" % (FEAT_DIM, FEAT_DIM)

    # полином-суррогат: мономы (коэффициент + степени признаков)
    approx = (
        "\n/* ── Полиномиальный суррогат мгновенного readout, степень %d (исполняемый; погрешность в шапке) ── */\n"
        "typedef struct { float c; unsigned char e[%d]; } Mono;\n" % (_deg, FEAT_DIM)
    )
    for ch, name in (("pitch", "APPROX_PITCH"), ("yaw", "APPROX_YAW")):
        coef, _m = fits[ch]
        monos = []
        for c, (key, _s, _r) in zip(coef, terms):
            if abs(c) < 1e-6:
                continue
            es = ", ".join(str(e) for e in _term_exponents(key))
            monos.append("    {%+.6ff, {%s}}," % (c, es))
        approx += "static const Mono %s[] = {\n%s\n};\n" % (name, "\n".join(monos))
    approx += """
static float poly_sum(const Mono* t, int n, const float* x) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        float m = t[i].c;
        for (int j = 0; j < %d; j++)
            for (int e = 0; e < t[i].e[j]; e++) m *= x[j];
        s += m;
    }
    return s;
}

void dn_approx(const float feat[%d], float out[2]) {
    out[0] = poly_sum(APPROX_PITCH, (int)(sizeof(APPROX_PITCH) / sizeof(Mono)), feat);
    out[1] = poly_sum(APPROX_YAW, (int)(sizeof(APPROX_YAW) / sizeof(Mono)), feat);
}

/* Локальный коэффициент при угловой скорости (численная производная суррогата):
 *   K_az = d(yaw)/d(feat_wb), K_el = d(pitch)/d(feat_wl) в единицах признака;
 * пересчёт в физику: a ≈ K·0.4·n_max·g·ω, N_poly_eff = K·0.4·n_max·g/V_m —
 * корректен ТОЛЬКО при явно выбранной V_m (скорость ракеты в полином не входит). */
void dn_local_gain(const float feat[%d], float* k_az, float* k_el) {
    const float eps = 1e-5f;
    float fp[%d], fm[%d], out[2];
    for (int j = 0; j < %d; j++) { fp[j] = feat[j]; fm[j] = feat[j]; }
    fp[2] += eps; fm[2] -= eps;
    dn_approx(fp, out); *k_az = out[1];
    dn_approx(fm, out); *k_az = (*k_az - out[1]) / (2 * eps);
    for (int j = 0; j < %d; j++) { fp[j] = feat[j]; fm[j] = feat[j]; }
    fp[3] += eps; fm[3] -= eps;
    dn_approx(fp, out); *k_el = out[0];
    dn_approx(fm, out); *k_el = (*k_el - out[0]) / (2 * eps);
}

#ifdef MUHOLET_MAIN
#include <stdio.h>
int main(void) {
    float feat[%d], out[2];
    for (int i = 0; i < %d; i++)
        if (scanf("%%f", &feat[i]) != 1) return 1;
    dn(feat, out);
    printf("%%.6f %%.6f\\n", out[0], out[1]);
    dn_approx(feat, out);
    printf("%%.6f %%.6f\\n", out[0], out[1]);
    float kaz, kel;
    dn_local_gain(feat, &kaz, &kel);
    printf("%%.6f %%.6f\\n", kaz, kel);
    return 0;
}
#endif
""" % (
        FEAT_DIM,  # poly_sum inner dim
        FEAT_DIM,  # dn_approx feat
        FEAT_DIM,  # dn_local_gain feat
        FEAT_DIM,  # fp
        FEAT_DIM,  # fm
        FEAT_DIM,  # reset loop 1
        FEAT_DIM,  # reset loop 2
        FEAT_DIM,  # main feat
        FEAT_DIM,  # main loop
    )
    return head + body + approx
