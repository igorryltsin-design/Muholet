"""PNG-графики для автоотчёта без внешних зависимостей (zlib + struct).

Шрифтов в рантайме нет, поэтому подписи осей не рисуются: числа рядом лежат
в markdown-таблицах отчёта, картинка даёт форму кривой. Палитра — фосфор
стенда: тёмный фон, зелёная кривая, янтарные вспомогательные.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

BG = (7, 12, 11)
GRID = (30, 52, 46)
PHOS = (125, 255, 200)
AMBER = (231, 193, 90)
RED = (255, 106, 74)


def write_png(path: Path, px: np.ndarray) -> Path:
    """Минимальный кодировщик PNG (RGB 8 бит) из массива H×W×3 uint8."""
    h, w, _ = px.shape
    raw = b"".join(b"\x00" + px[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    blob = b"\x89PNG\r\n\x1a\n"
    blob += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    blob += chunk(b"IDAT", zlib.compress(raw, 6))
    blob += chunk(b"IEND", b"")
    path.write_bytes(blob)
    return path


def _canvas(w: int, h: int) -> np.ndarray:
    return np.full((h, w, 3), BG, dtype=np.uint8)


def _grid(px: np.ndarray, nx: int = 4, ny: int = 4, pad: int = 10) -> None:
    h, w, _ = px.shape
    for i in range(ny + 1):
        y = pad + (h - 2 * pad) * i // ny
        px[y, pad : w - pad] = GRID
    for j in range(nx + 1):
        x = pad + (w - 2 * pad) * j // nx
        px[pad : h - pad, x] = GRID


def _scale(values: list[float], log: bool = False) -> list[float]:
    """Нормировка 0…1; лог — для промахов с редкими огромными выбросами."""
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    if log:
        lo, hi = np.log1p(max(lo, 0.0)), np.log1p(hi)
        return [float((np.log1p(max(v, 0.0)) - lo) / (hi - lo)) for v in values]
    return [float((v - lo) / (hi - lo)) for v in values]


def lines_png(
    path: Path,
    series: list[dict],
    w: int = 640,
    h: int = 300,
    log: bool = False,
) -> Path:
    """Линейный график: series = [{values, color}]; оси — сетка без подписей."""
    px = _canvas(w, h)
    _grid(px, pad=14)
    pad = 14
    all_vals = [v for s in series for v in s["values"] if v == v]
    if not all_vals:
        return write_png(path, px)
    lo, hi = min(all_vals), max(all_vals)
    if hi - lo < 1e-9:
        hi, lo = hi + 1, lo - 1
    n_max = max(len(s["values"]) for s in series)
    for s in series:
        vals = [v for v in s["values"] if v == v]
        if len(vals) < 2:
            continue
        t = _scale(vals, log=log)
        col = s.get("color", PHOS)
        for i in range(len(vals) - 1):
            x0 = pad + (w - 2 * pad) * i / max(n_max - 1, 1)
            x1 = pad + (w - 2 * pad) * (i + 1) / max(n_max - 1, 1)
            y0 = h - pad - (h - 2 * pad) * t[i]
            y1 = h - pad - (h - 2 * pad) * t[i + 1]
            steps = max(int(abs(x1 - x0)) + 1, int(abs(y1 - y0)) + 1, 1)
            for k in range(steps + 1):
                x = int(x0 + (x1 - x0) * k / steps)
                y = int(y0 + (y1 - y0) * k / steps)
                px[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = col
    return write_png(path, px)


def heatmap_png(path: Path, values: list[list[float]], cell: int = 56) -> Path:
    """Тепловая карта (матрица переносимости/карта): зелёный — хорошо, красный — плохо.
    Шкала логарифмическая по столбцам: манёвры с дикими выбросами не съедают цвет."""
    n_rows, n_cols = len(values), max(len(r) for r in values)
    col_max = [max(values[r][c] for r in range(n_rows)) for c in range(n_cols)]
    px = _canvas(n_cols * (cell + 2) + 2, n_rows * (cell + 2) + 2)
    for r in range(n_rows):
        for c in range(n_cols):
            v = values[r][c]
            t = float(np.log1p(max(v, 0.0)) / np.log1p(max(col_max[c], 1e-9))) if col_max[c] > 0 else 0.0
            col = (
                int(BG[0] + (RED[0] - BG[0]) * t),
                int(BG[1] + (PHOS[1] * 0.55 - BG[1]) * (1 - t)),
                int(BG[2] + (PHOS[2] * 0.55 - BG[2]) * (1 - t)),
            )
            y0, x0 = 2 + r * (cell + 2), 2 + c * (cell + 2)
            px[y0 : y0 + cell, x0 : x0 + cell] = col
    return write_png(path, px)


def bars_png(path: Path, values: list[float], w: int = 640, h: int = 240) -> Path:
    """Столбчатая диаграмма (абляция): высота — значение, шкала лог по максимуму."""
    px = _canvas(w, h)
    _grid(px, pad=14)
    pad = 14
    if not values:
        return write_png(path, px)
    hi = max(values) if max(values) > 0 else 1.0
    n = len(values)
    bw = (w - 2 * pad) / n
    for i, v in enumerate(values):
        t = float(np.log1p(max(v, 0.0)) / np.log1p(hi))
        bh = int((h - 2 * pad) * t)
        x0 = int(pad + i * bw + 2)
        x1 = int(pad + (i + 1) * bw - 2)
        col = AMBER if i == 0 else PHOS  # первый столбец — база
        px[h - pad - bh : h - pad, max(x0, 0) : max(x1, 1)] = col
    return write_png(path, px)
