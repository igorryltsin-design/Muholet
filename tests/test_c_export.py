"""Экспорт закона наведения в чистый C: компиляция и сравнение с python-сетью.

Компилируем сгенерированный файл (dn() + dn_approx()) настоящим компилятором
и сверяем отклик с brain_dn() на сетке признаков. Без cc тест пропускается.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from navedenie.brain_store import get_circuit
from navedenie.formula import brain_dn, brain_formula_c

CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
pytestmark = pytest.mark.skipif(CC is None, reason="нет компилятора C — экспорт нечем проверить")


def _compile_and_eval(code: str, feats: np.ndarray, tmp_path: Path) -> tuple[np.ndarray, np.ndarray]:
    src = tmp_path / "mozg.c"
    src.write_text(code, encoding="utf-8")
    exe = tmp_path / "mozg"
    subprocess.run([CC, "-DMUHOLET_MAIN", "-O0", str(src), "-o", str(exe), "-lm"], check=True, capture_output=True)
    exact = np.zeros((len(feats), 2))
    approx = np.zeros((len(feats), 2))
    for i, f in enumerate(feats):
        inp = " ".join(f"{v:.6f}" for v in f)
        out = subprocess.run([str(exe)], input=inp, capture_output=True, text=True, check=True).stdout
        a0, a1 = out.splitlines()[0].split()
        b0, b1 = out.splitlines()[1].split()
        exact[i] = [float(a0), float(a1)]
        approx[i] = [float(b0), float(b1)]
    return exact, approx


GRID = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.9, -0.4, 0.2, -0.1, 0.5, 0.0, 0.3, 0.0, 0.0, 1.0],
        [-0.7, 0.6, -0.9, 0.8, 0.1, -0.3, 0.7, -0.2, 0.4, 1.0],
        [0.3, 0.3, 0.3, 0.3, -0.8, 0.4, -0.4, 0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


@pytest.mark.parametrize("kind", ["stub", "connectome", "full"])
def test_c_export_matches_python_network(kind: str, tmp_path: Path) -> None:
    circuit = get_circuit(kind)
    code = brain_formula_c(kind)
    exact_c, approx_c = _compile_and_eval(code, GRID, tmp_path)
    exact_py = brain_dn(kind, circuit, GRID)
    assert np.max(np.abs(exact_c - exact_py)) < 2e-3, "точная сеть в C разошлась с python"

    # dn_approx: тот же полином, считаем в python по выгруженным коэффициентам
    from navedenie.formula import FORMULA_DEGREE, _grid_surrogate, _poly_design

    deg, terms, fits, _flight = _grid_surrogate(kind, circuit, FORMULA_DEGREE.get(kind, 2))
    A, _t = _poly_design(GRID, deg)
    for ch, k in enumerate(("pitch", "yaw")):
        coef, _m = fits[k]
        expected = A @ coef
        assert np.max(np.abs(approx_c[:, ch] - expected)) < 2e-3, f"полином {k} в C разошёлся"


def test_c_export_degree_override_changes_polynomial(monkeypatch) -> None:
    # степень подменяется явно: в шапке и в МНОЖЕСТВЕ мономов видно. Мозг
    # подменяется на детерминированно КВАДРАТИЧНУЮ функцию — тогда степень 2
    # обязана добавить квадратичный моном, которого у степени 1 не было.
    import numpy as np

    from navedenie import formula as f
    from navedenie.circuit import ConnectomeCircuit

    def quad_dn(kind, circuit, X):
        return np.stack([0.8 * X[:, 0] * X[:, 0], 0.5 * X[:, 1]], axis=1)

    monkeypatch.setattr(f, "get_circuit", lambda kind: ConnectomeCircuit())
    monkeypatch.setattr(f, "brain_dn", quad_dn)
    d1, d2 = f.brain_formula_c("connectome", degree=1), f.brain_formula_c("connectome", degree=2)
    assert "МНК, степень 1;" in d1 and "МНК, степень 2;" in d2
    # квадратичный моном beta3² (показатели {2, 0, ...}) выживает только при степени 2
    assert "+0.800000f, {2, 0, 0" in d2, "степень 2 включает квадрат beta3"
    assert "+0.800000f, {2, 0, 0" not in d1, "у степени 1 квадратного монома нет"


def test_python_export_honors_degree() -> None:
    from navedenie.formula import brain_formula, brain_formula_python

    data = brain_formula("stub", degree=2)
    assert data["poly"]["degree"] == 2
    py = brain_formula_python("stub", degree=2)
    assert "APPROX_DEGREE = 2" in py
