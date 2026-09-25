"""PNG-кодировщик и автоотчёт с графиками: валидность файлов, содержимое отчёта."""

import struct
import zlib
from pathlib import Path

import numpy as np

from navedenie import png


def _parse_png(blob: bytes) -> tuple[int, int]:
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    assert blob[12:16] == b"IHDR"
    w, h = struct.unpack(">II", blob[16:24])
    # IDAT декомпрессится целиком: строки по 1+3*w байт
    idat = blob[blob.index(b"IDAT") + 4 : blob.index(b"IEND") - 4]
    raw = zlib.decompress(idat)
    assert len(raw) == h * (1 + 3 * w)
    return w, h


def test_write_png_roundtrip(tmp_path: Path) -> None:
    px = np.zeros((7, 11, 3), dtype=np.uint8)
    px[3, 5] = (125, 255, 200)
    p = png.write_png(tmp_path / "t.png", px)
    w, h = _parse_png(p.read_bytes())
    assert (w, h) == (11, 7)


def test_chart_helpers_produce_valid_png(tmp_path: Path) -> None:
    _parse_png(png.lines_png(tmp_path / "l.png", [{"values": [1.0, 4.0, 2.0], "color": png.PHOS}, {"values": [3.0, 2.0, 8.0], "color": png.AMBER}]).read_bytes())
    _parse_png(png.heatmap_png(tmp_path / "h.png", [[1.0, 900.0], [40.0, 3.0]]).read_bytes())
    _parse_png(png.bars_png(tmp_path / "b.png", [39.0, 109.0, 316.0]).read_bytes())
    # вырожденные случаи не падают
    _parse_png(png.lines_png(tmp_path / "e.png", [{"values": [5.0, 5.0], "color": png.PHOS}]).read_bytes())
    _parse_png(png.lines_png(tmp_path / "n.png", []).read_bytes())


def test_night_report_with_png(tmp_path: Path, monkeypatch) -> None:
    from navedenie.app import NightReportIn, report_night, _EXP_DIR

    monkeypatch.setattr("navedenie.app._EXP_DIR", tmp_path / "exp")
    body = NightReportIn(
        title="Тест смены",
        ablation={"base": {"miss": 39.0, "hit_rate": 0.67}, "rows": [{"zone": "пеленг", "channels": 28, "miss": 109.0, "hit_rate": 0.33}]},
        map_rows=[
            {"maneuver": "вираж", "law": "ПН", "law_id": "pn", "miss_pn": 42.0, "hit_pn": True, "miss_fly": 39.0, "advantage": 3.0},
            {"maneuver": "вираж", "law": "APN", "law_id": "apn", "miss_pn": 41.0, "hit_pn": True, "miss_fly": 44.0, "advantage": -3.0},
            {"maneuver": "горка/пике", "law": "ПН", "law_id": "pn", "miss_pn": 55.0, "hit_pn": True, "miss_fly": 60.0, "advantage": -5.0},
            {"maneuver": "горка/пике", "law": "APN", "law_id": "apn", "miss_pn": 50.0, "hit_pn": True, "miss_fly": 48.0, "advantage": 2.0},
        ],
        faults=[{"fraction": 0.0, "miss": 39.0, "worst": 41.0, "hit_rate": 1.0}, {"fraction": 0.75, "miss": 40.3, "worst": 44.0, "hit_rate": 0.67}],
        coev_history=[{"gen": 1, "maneuver": "weave", "n_target": 4, "miss_fly": 2381.0}],
        transfer={
            "kind": "stub",
            "episodes": 3,
            "seconds": 1.0,
            "rows": [{"train": "turn", "train_miss_med": 60.0, "tests": {"straight": 30.0, "turn": 55.0, "weave": 90.0}}],
        },
        scaling={
            "kind": "full",
            "episodes": 2,
            "seconds": 1.0,
            "rows": [{"size": 8, "params": 32784, "n_cells": 4200, "miss_before": 300.0, "miss_after": 120.0, "hit_rate_after": 0.67, "ref_dev_after": 50.0, "train_miss_med": 130.0}],
        },
    )
    out = report_night(body)
    folder = Path(out["path"]).parent
    assert out["ok"]
    assert set(out["files"]) == {"ablation.png", "map.png", "faults.png", "coev.png", "transfer.png", "scaling.png"}
    for f in out["files"]:
        _parse_png((folder / f).read_bytes())
    md = Path(out["path"]).read_text(encoding="utf-8")
    for f in out["files"]:
        assert f"({f})" in md
    assert "## Матрица переносимости" in md and "| turn | 30 | 55 | 90 |" in md
    assert "## Кривая масштабируемости" in md and "| 8 | 32784 |" in md
    assert (tmp_path / "exp").exists()


def test_distill_pipeline_teacher_save_load() -> None:
    from navedenie.app import DistillIn, _distill_state, brain_distill, brain_distill_load, brain_distill_save

    _distill_state.clear()
    d = brain_distill(DistillIn(teacher="stub"))
    assert d["teacher_kind"] == "stub"
    assert _distill_state["teacher"] == "stub"
    saved = brain_distill_save()
    assert Path(saved["path"]).exists()
    loaded = brain_distill_load()
    assert loaded["ok"]
    assert loaded["distilled"]["miss"] == d["distilled"]["miss"]  # тот же дистиллят из файла
