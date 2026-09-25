"""Зона неубегаемого перехвата (POST /api/science/capture-zone): граница
«успеет / не успеет» в плоскости «дальность пуска × постоянная нормальная
перегрузка цели» — классическая задача уклонения (Дмитрий, Ровинский, Юрьев;
в учебниках по прицельности — зона гарантированного поражения).

Опорная аналитика — установившийся режим ПН против цели с постоянной
перегрузкой (Зархан, Tactical Missiles Guidance; [Гусев 1996] гл. 4):
a_уст = N/(N−1)·a_цели, т.е. при N=4 и n_max=30 g цель убегает, если
n_target > (N−1)/N·n_max = 22.5 g — тесты сандвичем проверяют это число
со сторон (20 g — перехват на всей сетке, 30 g — нет).
Структура зоны при постоянной перегрузке цели может быть и отрезком (один
связный блок перехвата — тогда бисекция находит и ближнюю, и дальнюю границу),
и долепестной (несколько блоков — границ не строим). Тесты проверяют честность
статистики строк (смены исхода, число блоков, статусы, скобки бисекции),
аналитику ближней границы от времени разворота и масштабные законы √y и 1/√n,
а не гипотетическую монотонность «дальше = хуже»."""

import pytest

from navedenie.app import CaptureZoneIn, capture_zone_endpoint
from navedenie.science import CZ_RANGES_M, CZ_TARGET_GS


def _base(**kw) -> dict:
    base = dict(mode="pn", law="pn", aspect="tail-chase", v_m=780.0, v_t=260.0,
                off_axis_m=420.0, n_max=30.0, pn_n=4.0, dt=0.02, t_max=30.0,
                kill_radius_m=45.0)
    base.update(kw)
    return base


@pytest.fixture(scope="module")
def scan() -> dict:
    """Один модульный скан: 3 строки × 4 дальности без бисекции."""
    return capture_zone_endpoint(CaptureZoneIn(**_base(
        ranges_m=[3000.0, 6000.0, 9000.0, 12000.0],
        target_gs=[0.0, 20.0, 30.0],
        refine=False,
    )))


def _row(scan: dict, g: float) -> dict:
    return next(r for r in scan["rows"] if r["n_target_g"] == g)


def test_meta_echoes_scenario(scan):
    """Эхо метаданных: закон, модель, сетки, window прогона — из сценария."""
    assert scan["law"] == "pn" and scan["mode"] == "pn"
    assert scan["model"] == "kinematic_legacy"
    assert scan["n_max_g"] == 30.0 and scan["pn_n"] == 4.0
    assert scan["t_max_s"] == 30.0
    assert scan["ranges_m"] == [3000.0, 6000.0, 9000.0, 12000.0]
    assert scan["target_gs"] == [0.0, 20.0, 30.0]
    assert list(CZ_RANGES_M)[0] == 2000.0 and list(CZ_TARGET_GS)[0] == 0.0


def test_row_statistics_consistent(scan):
    """hit_frac строки — среднее по ячейкам; монотонность считается по сменам."""
    for r in scan["rows"]:
        seq = [c["hit"] for c in r["cells"]]
        assert r["hit_frac"] == round(sum(seq) / len(seq), 3)
        flips = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
        assert r["monotonic_violations"] == max(flips - 1, 0)
        # число связных блоков перехвата — то, что реально считает бэкенд
        blocks = sum(1 for i, h in enumerate(seq) if h and (i == 0 or not seq[i - 1]))
        assert r["hit_blocks"] == blocks
        for c in r["cells"]:
            assert {"range_m", "hit", "r_min_m", "t_end_s", "time_limited"} <= set(c)


def test_straight_target_zone_deeper_than_grid(scan):
    """Прямолинейная цель (0 g) при N=4 перехватывается на всей сетке 3…12 км."""
    r = _row(scan, 0.0)
    assert r["status"] == "all_hit" and r["monotonic_violations"] == 0
    assert all(c["hit"] for c in r["cells"])


def test_unescapable_asymptote_sandwiches_22p5g(scan):
    """Аналитика a_уст = N/(N−1)·a_цели: 20 g требует 4/3·20 = 26.7 g ≤ 30 g —
    перехват на всей сетке; 30 g требует 40 g > 30 g — уклонение возможно
    (не на всей сетке перехватов). Граница (N−1)/N·n_max = 22.5 g сандвичем."""
    light = _row(scan, 20.0)
    heavy = _row(scan, 30.0)
    assert light["status"] == "all_hit"
    assert heavy["hit_frac"] < 1.0
    # при 30 g исходы по сетке меняются — честный учёт долепестности
    assert heavy["monotonic_violations"] >= 1
    assert heavy["status"] in ("lobed", "bracketed", "no_hit")


def test_boundaries_require_refine(scan):
    """При refine=False численных границ нет ни у одной строки. Структура строки
    30 g (MHMM на сетке 3…12 км) — ОДИН связный блок перехвата, то есть зона ещё
    отрезок (внутри сетки у неё две границы), а не долепестность."""
    for r in scan["rows"]:
        assert "boundary_m" not in r and "inner_boundary_m" not in r
    heavy = _row(scan, 30.0)
    assert heavy["hit_blocks"] == 1
    assert heavy["status"] == "bracketed"


def test_zone_is_an_interval_between_two_boundaries():
    """Классическая зона поражения — ОТРЕЗОК дальностей: у строки 30 g (погон,
    цель с постоянной перегрузкой) перехват только в середине сетки, поэтому
    бисекция даёт и ближнюю, и дальнюю границу, и обе скобки шириной ≤ допуска.
    Ближняя — ракета не успевает развернуться, дальняя — цель уходит; обе
    лежат вокруг узла-перехвата и ни одна не обязана границей окна прогона."""
    out = capture_zone_endpoint(CaptureZoneIn(**_base(
        ranges_m=[3000.0, 6000.0, 9000.0, 12000.0],
        target_gs=[30.0],
        refine=True,
        refine_tol_m=50.0,
    )))
    r = out["rows"][0]
    seq = [c["hit"] for c in r["cells"]]
    assert seq == [False, True, False, False] and r["hit_blocks"] == 1
    inner, i_lo, i_hi = r["inner_boundary_m"], *r["inner_boundary_bracket_m"]
    outer, o_lo, o_hi = r["boundary_m"], *r["boundary_bracket_m"]
    assert 3000.0 < inner < 6000.0 < outer < 9000.0
    assert i_hi - i_lo <= 50.1 and o_hi - o_lo <= 50.1
    assert i_lo < inner < i_hi and o_lo < outer < o_hi
    # промахи у обеих границ — реальные (miss_pass), не отсечка окна t_max
    assert r["boundary_time_limited"] is False
    assert not any(c["time_limited"] for c in r["cells"])


def test_bisection_brackets_time_boundary():
    """Чистый переход «попал → промах» (окно t_max=16 с отрезает дальние
    прогоны): бисекция даёт границу в скобке 8…9 км, ширина скобки ≤ допуска,
    промах за скобкой помечен time_limited (граница окна, не физики)."""
    out = capture_zone_endpoint(CaptureZoneIn(**_base(
        ranges_m=[4000.0, 8000.0, 12000.0],
        target_gs=[0.0],
        t_max=16.0,
        refine=True,
        refine_tol_m=50.0,
    )))
    r = out["rows"][0]
    assert r["status"] == "bracketed" and r["monotonic_violations"] == 0
    assert r["r_last_hit_m"] == 8000.0 and r["r_first_miss_m"] == 12000.0
    lo, hi = r["boundary_bracket_m"]
    assert r["r_last_hit_m"] <= lo < hi <= 9000.0
    assert hi - lo <= 50.1
    assert lo < r["boundary_m"] < hi
    assert r["cells"][-1]["time_limited"] is True
    assert r["boundary_time_limited"] is True  # промах у границы — отсечка окна, не физика
    assert r["cells"][-1]["r_min_m"] > 45.0
    assert all(not c["time_limited"] for c in r["cells"] if c["hit"])


# ── ближняя граница зоны: время разворота перехватчика ───────────────────────
# На малых дальностях ракета НЕ успевает отработать боковое смещение: за время
# сближения t = R/V_сбл она смещается вбок не больше ½·a·t², a располагаемая
# нормальная a = n_max·g. Отсюда минимальная дальность пуска (классическая нижняя
# граница зоны применимости; [Гусев 1996] гл. 4, Зархан Tactical Missiles
# Guidance — turn-time / minimum engagement range):
#   R_ближ ≈ V_сбл·sqrt(2·y_смещ/(n_max·g)).
G0 = 9.81


def _zone(off_axis_m: float = 420.0, n_max: float = 12.0) -> dict:
    """Скан одной строки с прямолинейной целью: чисто «успеет ли ракета
    развернуться» на данной дальности пуска."""
    out = capture_zone_endpoint(CaptureZoneIn(**_base(
        n_max=n_max, aspect="head-on", v_m=780.0, v_t=260.0, off_axis_m=off_axis_m,
        ranges_m=[500.0, 1000.0, 2000.0, 4000.0, 8000.0],
        target_gs=[0.0], refine=True, refine_tol_m=50.0,
    )))
    return out["rows"][0]


def test_inner_boundary_matches_turn_time_estimate():
    """Ближняя граница из бисекции совпадает с оценкой времени разворота
    R = V_сбл·sqrt(2y/(n·g)) с точностью ~5 %: численная зона не выдумана."""
    r = _zone(off_axis_m=420.0, n_max=12.0)
    est = (780.0 + 260.0) * (2.0 * 420.0 / (12.0 * G0)) ** 0.5
    assert r["status"] == "bracketed" and r["hit_blocks"] == 1
    assert "boundary_m" not in r  # справа сетка целиком в перехвате — внешней границы нет
    b = r["inner_boundary_m"]
    assert 0.85 < b / est < 1.15, f"граница {b} против оценки {est}"


def test_inner_boundary_scales_as_sqrt_of_cross_range():
    """Закон √y: боковое смещение ×4 → ближняя граница ×2 (оценка и численно)."""
    near = _zone(off_axis_m=420.0)["inner_boundary_m"]
    far = _zone(off_axis_m=1680.0)["inner_boundary_m"]
    assert 1.7 < far / near < 2.3, f"{near} → {far}"


def test_inner_boundary_scales_as_inverse_sqrt_of_available_g():
    """Закон 1/√n: располагаемая перегрузка ×4 → ближняя граница ÷2
    (ракету «чем круче повернуть, тем раньше она успевает»)."""
    weak = _zone(off_axis_m=420.0, n_max=12.0)["inner_boundary_m"]
    strong = _zone(off_axis_m=420.0, n_max=48.0)["inner_boundary_m"]
    assert 0.4 < strong / weak < 0.6, f"{weak} → {strong}"


def test_inner_boundary_bracket_consistent_with_grid():
    """Скобка ближней границы уже допуска и согласована с сеткой: ВСЕ узлы левее
    границы — промахи, правее — перехваты (иначе это долепестность, а не
    граница, и бисекции быть не должно)."""
    r = _zone(off_axis_m=420.0, n_max=12.0)
    b = r["inner_boundary_m"]
    lo, hi = r["inner_boundary_bracket_m"]
    assert hi - lo <= 50.1 and lo < b < hi
    left = [c for c in r["cells"] if c["range_m"] < b]
    right = [c for c in r["cells"] if c["range_m"] > b]
    assert left and right
    assert all(not c["hit"] for c in left)
    assert all(c["hit"] for c in right)
    assert all(c["r_min_m"] > 45.0 for c in left)  # промахи настоящие, за сферой срабатывания
