"""ГСН: фовеальная широкоугольная сетчатка, прицел, честное декодирование признаков.

BIO — сенсорный агент: он НЕ получает истинных az/el/ω_LOS. Всё, что видит
контур (пеленги, их скорости, лоом, захват), декодируется ИЗ ИЗОБРАЖЕНИЯ
16×16 после применения шумов, мёртвых рецепторов и задержки:

1. Фовеальная геометрия: угловые центры ячеек распределены нелинейно —
   angle(q) = half_fov·sinh(k·q)/sinh(k), q ∈ [−1,1], k ≈ 3.5. Центр плотный
   (разрешение как у узкого кадра 12–14°), периферия редкая, покрытие —
   полное поле bio_fov_deg (по умолчанию 165°).
2. Захват (lock) определяется физическим полем и ОБНАРУЖЕНИЕМ сигнала в кадре,
   а не скрытым точным азимутом.
3. Пеленги decodируются как центр масс (центроид) изображения в угловых
   координатах; угловые скорости — конечными разностями декодированных
   пеленгов; лоом — из суммарной яркости пятна.
4. Шум и дропаут применяются К ИЗОБРАЖЕНИЮ до декодирования; постоянная карта
   «мёртвых» рецепторов создаётся один раз на прогон (в движке) и передаётся
   сюда готовой; доп. случайная задержка выбирается из U[0, seeker_jitter_s].
5. Истинные углы пишутся рядом (truth_az/truth_el) ТОЛЬКО для операторской
   телеметрии — контур их не получает.
"""

from __future__ import annotations

import numpy as np

from navedenie.sim import Vec, _unit

FOVEA_K = 3.5  # нелинейность фовеи: чем больше k, тем плотнее центр
THETA_WIN = 6  # окно оценки θ̇, кадров (0.12 с при dt=0.02)


def body_axes(v_m: Vec) -> tuple[Vec, Vec, Vec]:
    """Оси скоростной (траекторной) системы координат: x — по вектору скорости,
    y — вправо, z — вверх относительно текущего курса.

    Название `body_axes` оставлено ради совместимости, но это НЕ связанная система
    координат корпуса: продольная ось строится непосредственно по скорости,
    угловое положение корпуса моделью не задаётся. Это скоростная система OXкYкZк
    (кнemатика точки-массы). См. docs/notation.md."""
    x = _unit(v_m)
    if float(np.linalg.norm(x)) < 1e-8:
        x = np.array([1.0, 0.0, 0.0])
    world_up = np.array([0.0, 0.0, 1.0])
    y = np.cross(world_up, x)
    if float(np.linalg.norm(y)) < 0.05:
        y = np.cross(np.array([0.0, 1.0, 0.0]), x)
    y = _unit(y)
    z = _unit(np.cross(x, y))
    return x, y, z


def project(r: Vec, v_m: Vec) -> dict:
    """Истинные углы цели в осях корпуса (для телеметрии и рендера кадра)."""
    x, y, z = body_axes(v_m)
    rng = float(np.linalg.norm(r)) + 1e-9
    u = r / rng
    ahead = float(np.dot(u, x))
    az = float(np.arctan2(np.dot(u, y), ahead if abs(ahead) > 1e-9 else 1e-9))
    el = float(np.arctan2(np.dot(u, z), max(abs(ahead), 1e-6)))
    return {
        "range": rng,
        "az": az,
        "el": el,
        "ahead": ahead,
        "size": min(0.35, 12.0 / rng),
    }


SEEKER_N = 16  # сетчатка ГСН: 16×16 «омматидиев»


def fovea_angles(half_fov: float, n: int = SEEKER_N, k: float = FOVEA_K) -> np.ndarray:
    """Нелинейные угловые центры n ячеек: плотно у оси, редко на периферии.
    angle(q) = half_fov·sinh(k·q)/sinh(k), q = −1 + 2(i+0.5)/n. Возвращает радианы."""
    q = -1.0 + 2.0 * (np.arange(n, dtype=np.float64) + 0.5) / n
    return half_fov * np.sinh(k * q) / np.sinh(k)


def fovea_cell_width(half_fov: float, n: int = SEEKER_N, k: float = FOVEA_K) -> np.ndarray:
    """Локальный угловой размер ячейки (рад): |d angle / dq|·Δq."""
    q = -1.0 + 2.0 * (np.arange(n, dtype=np.float64) + 0.5) / n
    dq = 2.0 / n
    dang = half_fov * k * np.cosh(k * q) / np.sinh(k)
    return dang * dq


def fovea_inverse(angle: float, half_fov: float, k: float = FOVEA_K) -> float:
    """Угол → нормированная координата q ∈ [−1, 1] (обратная к fovea_angles)."""
    s = np.sinh(k) * (angle / max(half_fov, 1e-9))
    # asinh может переполниться на огромных углах — клипуем
    return float(np.arcsinh(np.clip(s, -1e6, 1e6)) / k)


def raster_foveal(
    az: float,
    el: float,
    size: float,
    half_fov: float,
    n: int = SEEKER_N,
    k: float = FOVEA_K,
    range_m: float | None = None,
) -> np.ndarray:
    """Изображение цели на фовеальной сетчатке.

    Пятно рисуется в УГЛОВЫХ координатах: ширина гауссианы на каждой оси —
    максимум из локальной ширины ячейки (функция рассения точки, PSF собственной
    оптики: пятно не уже ~0.7 ячейки, чтобы цель не терялась между рецепторами)
    и углового размера цели. Планка ЛОКАЛЬНАЯ: в фовеа пятно узкое (угловой
    размер цели измерим задолго до контакта), на редкой периферии — широкое.
    size — прежняя безразмерная мера видимого размера."""
    ax = fovea_angles(half_fov, n, k)
    wx = fovea_cell_width(half_fov, n, k)
    sig = np.maximum(0.7 * wx, size * max(half_fov, 1e-9) * 0.9)
    az_c = np.clip(az, -ax[-1], ax[-1])
    el_c = np.clip(el, -ax[-1], ax[-1])
    gx = np.exp(-0.5 * ((ax - az_c) / sig) ** 2)
    gy = np.exp(-0.5 * ((ax - el_c) / sig) ** 2)
    return np.outer(gy, gx)


def psf_reference(az: float, el: float, half_fov: float, n: int = SEEKER_N, k: float = FOVEA_K) -> dict:
    """Калибровка СОБСТВЕННОЙ оптики в точке (az, el): пятно от точечной цели.

    Это врождённое знание о сетчатке (карта ячеек), а не информация о цели:
    тот же raster с size=0. Возвращает суммарную яркость и локальную σ пятна PSF."""
    psf = raster_foveal(az, el, 0.0, half_fov, n, k)
    ax = fovea_angles(half_fov, n, k)
    wx = fovea_cell_width(half_fov, n, k)
    ic = int(np.argmin(np.abs(ax - np.clip(az, -ax[-1], ax[-1]))))
    sig_loc = 0.7 * float(wx[ic])
    return {"total": float(np.sum(psf)), "sigma": sig_loc}


def _blob_total(az: float, el: float, sigma: float, half_fov: float, n: int, k: float, ax: np.ndarray, wx: np.ndarray) -> float:
    """Суммарная яркость пятна с ядром σ на этой сетчатке (та же модель, что raster)."""
    sig = np.maximum(0.7 * wx, sigma)
    az_c = np.clip(az, -ax[-1], ax[-1])
    el_c = np.clip(el, -ax[-1], ax[-1])
    gx = np.exp(-0.5 * ((ax - az_c) / sig) ** 2)
    gy = np.exp(-0.5 * ((ax - el_c) / sig) ** 2)
    return float(np.outer(gy, gx).sum())


def measure_theta(total: float, az: float, el: float, half_fov: float, n: int = SEEKER_N, k: float = FOVEA_K) -> float:
    """Измеренный угловой размер цели (рад) из декодированного кадра.

    Обратная задача оптики: суммарная яркость пятна T(σ) монотонна по угловому
    размеру σ ядра; решаем T(σ) = измеренная яркость бисекцией в ИЗВЕСТНОЙ
    модели собственной сетчатки (врождённая калибровка, не информация о цели).
    Цель много меньше PSF → T = T_PSF → theta = 0: размер не измерим — честный
    ответ сенсора. В линейной зоне theta ∝ 1/дальность, поэтому θ̇/θ ≈ Vc/R."""
    if total <= 0.0:
        return 0.0
    ax = fovea_angles(half_fov, n, k)
    wx = fovea_cell_width(half_fov, n, k)
    ref = psf_reference(az, el, half_fov, n, k)
    t_lo = _blob_total(az, el, 0.0, half_fov, n, k, ax, wx)
    if total <= t_lo * 1.01:  # в пределах PSF и ниже — размер не разрешим
        return 0.0
    sig_lo, sig_hi = 0.0, float(max(half_fov, ref["sigma"]))
    for _ in range(14):
        mid = 0.5 * (sig_lo + sig_hi)
        if _blob_total(az, el, mid, half_fov, n, k, ax, wx) < total:
            sig_lo = mid
        else:
            sig_hi = mid
    return float(0.5 * (sig_lo + sig_hi))


def decode_image(
    img: np.ndarray,
    half_fov: float,
    n: int = SEEKER_N,
    k: float = FOVEA_K,
) -> dict:
    """Декодирование признаков из изображения (единственный источник углов агента).

    Возвращает az/el (центроид пятна в угловых координатах, рад), size —
    суммарную яркость (прокси углового размера), peak — максимальную яркость и
    theta — ИЗМЕРЕННЫЙ угловой размер цели (рад, калибровка по собственной PSF)."""
    ax = fovea_angles(half_fov, n, k)
    total = float(np.sum(img))
    if total < 0.15:  # сигнал ниже порога обнаружения — пеленг не выдаётся
        return {"az": 0.0, "el": 0.0, "size": 0.0, "peak": 0.0, "theta": 0.0}
    wx = img.sum(axis=0)
    wy = img.sum(axis=1)
    az = float(np.dot(wx, ax) / max(wx.sum(), 1e-9))
    el = float(np.dot(wy, ax) / max(wy.sum(), 1e-9))
    # размер: суммарная яркость растёт с угловым размером пятна; нормируем в 0…0.35
    size = float(np.clip(total / (n * n * 0.08), 0.0, 0.35))
    return {
        "az": az,
        "el": el,
        "size": size,
        "peak": float(np.max(img)),
        "theta": measure_theta(total, az, el, half_fov, n, k),
    }


def dead_mask_for(fraction: float, rng: np.random.Generator, n: int = SEEKER_N) -> np.ndarray | None:
    """ПОСТОЯННАЯ карта отказов: создаётся РОВНО ОДИН раз на прогон (seed-стабильно)."""
    if fraction <= 0.0:
        return None
    return rng.random((n, n)) < fraction


class ObservationBuffer:
    """Очередь наблюдений с временными метками (задержка + джиттер).

    - seeker_delay_s — штатная задержка;
    - seeker_jitter_s — верхняя граница ДОПОЛНИТЕЛЬНОЙ случайной задержки,
      разыгрываемой как U[0, jitter] на каждый кадр (семантика задокументирована);
    - выдаётся самое СВЕЖЕЕ наблюдение с меткой t ≤ t_now − delay; поддерживает
      задержки больше одного dt.
    """

    def __init__(self, delay_s: float, jitter_s: float, dt: float, rng: np.random.Generator) -> None:
        self.delay = max(float(delay_s), 0.0)
        self.jitter = max(float(jitter_s), 0.0)
        self.dt = max(float(dt), 1e-6)
        self.rng = rng
        self.buf: list[tuple[float, dict]] = []
        # ёмкость: максимум задержки + запас
        self.cap = int((self.delay + self.jitter) / self.dt) + 8

    def push(self, t: float, obs: dict) -> None:
        self.buf.append((t, obs))
        if len(self.buf) > self.cap:
            self.buf.pop(0)

    def sample(self, t_now: float) -> dict:
        extra = float(self.rng.uniform(0.0, self.jitter)) if self.jitter > 0.0 else 0.0
        t_target = t_now - self.delay - extra
        chosen = self.buf[0]
        for tt, obs in self.buf:
            if tt <= t_target + 1e-9:
                chosen = (tt, obs)
            else:
                break
        return chosen[1]


def observe(
    r: Vec,
    v_m: Vec,
    prev: dict | None,
    *,
    half_fov: float,
    dt_s: float = 0.005,
    noise_az: float = 0.0,
    noise_range: float = 0.0,
    lock_drop_p: float = 0.0,
    dead_mask: np.ndarray | None = None,
    dropout_p: float = 0.0,
    rng: np.random.Generator | None = None,
    brightness: float = 1.0,
) -> dict:
    """Кадр ГСН: изображение → (яркость, шум, дропаут) → декодирование признаков.

    half_fov — ПОЛОВИНА физического поля зрения сетчатки, рад (для BIO это
    bio_fov_deg/2 ≈ 82.5°; oracle-ПН работает с узким кадром sc.fov_deg/2 и
    истинной геометрией — она помечена как oracle). Шум пеленга здесь
    превращается в шум ИЗОБРАЖЕНИЯ (пятнышко гуляет по сетчатке), пеленги
    декодируются уже из зашумлённого кадра. rng — из движка, детерминизм.
    brightness — фотометрия цели (масштаб яркости пятна до декодирования):
    меняет ИЗМЕРЕННУЮ theta/rho (они инвертируются из суммарной яркости) и
    используется для проверки «скрытого дальномера» (P9), в управление не
    входит сам по себе."""
    raw = project(r, v_m)
    see_los = 2.0 * half_fov  # физическое поле по каждой оси
    los = float(np.arccos(np.clip(raw["ahead"], -1.0, 1.0)))

    in_fov = abs(raw["az"]) <= half_fov and abs(raw["el"]) <= half_fov
    # изображение: цель в поле — полное пятно; чуть за краем — приглушённое у края
    margin = 2.0 * half_fov * 0.02
    if in_fov or los < 2.0 * half_fov * 1.05:
        img = raster_foveal(raw["az"], raw["el"], raw["size"], half_fov)
        if not in_fov:
            img *= 0.35
    else:
        img = np.zeros((SEEKER_N, SEEKER_N))

    # фотометрия цели до декодирования: яркость пятна меняется — контур видит
    # только изображение, никаких компенсаций
    if brightness != 1.0:
        img = img * float(brightness)

    # ПОСТОЯННАЯ маска мёртвых рецепторов приходит снаружи (одна на прогон);
    # здесь — только независимый дропаут этого кадра
    if dropout_p > 0.0 and rng is not None:
        img = img * (rng.random(img.shape) >= dropout_p)
    if dead_mask is not None:
        img = img * (~dead_mask)
    # сенсорный шум: амплитуда пятна гуляет по ячейкам до декодирования
    if noise_az > 0.0 and rng is not None:
        img = np.clip(img + rng.standard_normal(img.shape) * (noise_az / max(half_fov, 1e-9)) * 0.5, 0.0, None)

    # декодирование — единственный источник углов агента
    dec = decode_image(img, half_fov)
    detected = dec["peak"] > 0.25 and in_fov
    lock = bool(detected)
    if lock and lock_drop_p > 0.0 and rng is not None and rng.random() < lock_drop_p:
        lock = False

    rng_m = raw["range"]
    if rng is not None and noise_range > 0.0:
        rng_m += float(rng.standard_normal()) * noise_range

    # производные — из последовательности ДЕКОДИРОВАННЫХ измерений
    dt_prev = max(dt_s, 1e-6)
    if prev is None or not lock:
        az_dot = el_dot = size_dot = theta_dot = 0.0
    else:
        az_dot = (dec["az"] - prev["az"]) / dt_prev
        el_dot = (dec["el"] - prev["el"]) / dt_prev
        size_dot = (dec["size"] - prev["size"]) / dt_prev

    # фаза сближения — ТОЛЬКО из декодированных theta/theta_dot:
    # rho ≈ θ̇/θ — оптическая оценка Vc/R ≈ 1/t_go (без истинной дальности);
    # tau_contact = θ/θ̇ — ограниченная оценка времени до контакта, с.
    # Декодированный размер сглаживается экспонентой (тау 60 мс — динамика
    # фоторецептора); θ̇ берётся на окне THETA_WIN кадров (сглаживает ступеньку
    # при пересечении цели с PSF сетчатки). Окно едет ВНУТРИ кадра через буфер
    # задержки: оценка доступна в момент съёма кадра.
    theta_tau = 0.06
    raw_theta = float(dec["theta"]) if lock else 0.0
    prev_theta_s = float(prev.get("theta") or 0.0) if prev is not None else 0.0
    theta = raw_theta + (prev_theta_s - raw_theta) * float(np.exp(-dt_prev / theta_tau))
    if not lock and theta < 1e-6:
        theta = 0.0
    win = [float(v) for v in ((prev or {}).get("theta_win") or [])][-THETA_WIN + 1 :] + [theta]
    win_dt = max((len(win) - 1) * dt_prev, dt_prev)
    theta_dot = float(np.clip((win[-1] - win[0]) / win_dt, -5.0, 5.0))
    rho = float(np.clip(theta_dot / max(theta, 1e-4), -10.0, 30.0))
    tau_contact = float(np.clip(theta / theta_dot, 0.0, 30.0)) if theta_dot > 1e-3 else 30.0

    # сглаженные угловые скорости — ОТДЕЛЬНЫЕ каналы для углоскоростных законов:
    # сырые az_dot/el_dot остаются признаками контура (мигрировавшие мозги не
    # меняют поведение), а законам наведения нужна фильтрация квантования фовеи
    tau_rate = 0.1
    k_rate = 1.0 - float(np.exp(-dt_prev / tau_rate))
    prev_az_s = float((prev or {}).get("az_dot_s") or 0.0)
    prev_el_s = float((prev or {}).get("el_dot_s") or 0.0)
    az_dot_s = prev_az_s + (az_dot - prev_az_s) * k_rate if lock else prev_az_s * (1.0 - k_rate)
    el_dot_s = prev_el_s + (el_dot - prev_el_s) * k_rate if lock else prev_el_s * (1.0 - k_rate)
    if not lock and abs(az_dot_s) < 1e-6 and abs(el_dot_s) < 1e-6:
        az_dot_s = el_dot_s = 0.0

    return {
        # измеренное (декодированное из изображения) — единственное, что видит контур
        "az": dec["az"] if lock else 0.0,
        "el": dec["el"] if lock else 0.0,
        "size": dec["size"] if lock else 0.0,
        "az_dot": az_dot,
        "el_dot": el_dot,
        "az_dot_s": az_dot_s,
        "el_dot_s": el_dot_s,
        "size_dot": size_dot,
        "theta": theta,
        "theta_win": win,
        "theta_dot": theta_dot,
        "rho": rho,
        "tau_contact": tau_contact,
        "lock": lock,
        "fov": 2.0 * half_fov,
        "image": img,
        "range": rng_m,  # отдельный заявленный датчик дальности (шумится честно)
        # истинные углы — ТОЛЬКО для операторской телеметрии, контур не читает
        "truth_az": raw["az"],
        "truth_el": raw["el"],
        "truth_range": raw["range"],
    }
