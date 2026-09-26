"""Реактивные законы уклонения цели (дуэль «муха-ракета против мухи-самолёта»,
navedenie/evader.py): аналитика команд на числах (ортогональность, клип
n_target·g, вырождение), рост CPA по градиенту, знак «ПН наоборот», боевая
жизнь fuse_life_s с финалом fuse_expired, замкнутый контур на бортовой и
догонной геометриях, реестр меток и проводка API."""

from __future__ import annotations

import numpy as np
import pytest

from navedenie.app import ScenarioIn, run_once
from navedenie.engine import collect
from navedenie.evader import EVADERS, cpa_max_accel, evader_accel, negpn_accel, away_accel
from navedenie.glossary import EVADER_LABEL
from navedenie.sim import G, Body, Scenario, los_omega

# «мухиные» сопоставимые скорости: ракета лишь чуть быстрее цели
FAIR = dict(
    mode="pn", law="pn", v_m=320, v_t=260, n_max=10, pn_n=4, dt=0.02,
    t_max=35, kill_radius_m=45, range_m=6000,
)

# аналитикой плоского evader_accel покрыты только реактивные законы;
# brain обслуживает движок через EvaderBrainSensor (tests/test_evader_brain.py)
REACTION_LAWS = tuple(law for law in EVADERS if law != "brain")


def _sc(**kw) -> Scenario:
    return Scenario(**{**FAIR, "duel": True, "n_target": 8.0, **kw})


# ── аналитика отдельных законов ──────────────────────────────────────────────

def test_evader_orthogonal_clipped_and_helpless() -> None:
    """Команда любого уклониста ортогональна скорости цели и ограничена
    n_target·g; при n_target=0 и при «ракете в стороне» (без сближения) — ноль."""
    rng = np.random.default_rng(11)
    for law in REACTION_LAWS:
        for _ in range(30):
            t = Body(p=rng.normal(size=3) * 100.0, v=np.array([260.0, 0.0, 0.0]) + rng.normal(size=3) * 20.0)
            m = Body(p=t.p + rng.normal(size=3) * 4000.0, v=rng.normal(size=3) * 300.0)
            sc = _sc(evader_law=law)
            a = evader_accel(t, m, sc)
            assert abs(float(np.dot(a, t.v))) <= 1e-6 * (np.linalg.norm(a) * np.linalg.norm(t.v) + 1.0)
            assert float(np.linalg.norm(a)) <= 8.0 * G + 1e-9
        # неманёвренная цель
        assert np.allclose(evader_accel(t, m, _sc(evader_law=law, n_target=0.0)), 0.0)
    # уклоняться не от чего: ракета убегает
    t = Body(p=np.zeros(3), v=np.array([260.0, 0.0, 0.0]))
    m = Body(p=np.array([5000.0, 0.0, 0.0]), v=np.array([400.0, 0.0, 0.0]))
    for law, f in (("away", away_accel), ("cpa_max", cpa_max_accel)):
        assert np.allclose(f(t, m, 8.0 * G), 0.0, atol=1e-9), law


def test_head_on_collision_course_full_side_step() -> None:
    """Чисто коллизионный лобовой курс (d=0 и CPA_vec=0 одновременно): уклон
    вырожден — команда обязана быть полной по модулю (n_target·g) и вбок
    (детерминированный «нырок» перпендикулярно скорости)."""
    t = Body(p=np.zeros(3), v=np.array([260.0, 0.0, 0.0]))
    m = Body(p=np.array([7800.0, 0.0, 0.0]), v=np.array([-780.0, 0.0, 0.0]))
    sc = _sc()
    for law in ("away", "cpa_max"):
        a = evader_accel(t, m, _sc(evader_law=law))
        assert abs(float(np.linalg.norm(a)) - 8.0 * G) < 1e-6, (law, a)
        assert abs(float(a[0])) < 1e-6 and a[1] * a[1] > 0, (law, a)  # ⊥ v, вдоль ±Y


def test_cpa_max_gradient_increases_cpa() -> None:
    """Определение градиента: для случайной закрывающейся геометрии малый
    доворот скорости цели по команде cpa_max за τ обязан увеличивать CPA²
    прямой экстраполяции сильнее, чем любой противоположный доворот."""
    def cpa2(s, v):
        lam = float(np.dot(s, v)) / (float(np.dot(v, v)) ** 0.5)
        return max(float(np.dot(s, s)) - lam * lam, 0.0)
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(60):
        s = rng.normal(size=3) * 4000.0
        v_m = rng.normal(size=3) * 250.0
        v_t = np.array([260.0, 0.0, 0.0])
        if float(np.dot(s, v_m - v_t)) >= 0.0:  # нет сближения — пропускаем
            continue
        t = Body(p=np.zeros(3), v=v_t)
        m = Body(p=s, v=v_m)
        a = cpa_max_accel(t, m, 8.0 * G)
        assert abs(float(np.linalg.norm(a)) - 8.0 * G) < 1e-6  # «полный банк»
        u = a / float(np.linalg.norm(a))
        tau = 0.25
        base = cpa2(s, v_m - v_t)
        up = cpa2(s, v_m - (v_t + u * tau))
        down = cpa2(s, v_m - (v_t - u * tau))
        assert up > base and up > down, (base, up, down)
        checked += 1
    assert checked >= 10


def test_negpn_turns_against_los_rate() -> None:
    """Знак «ПН наоборот»: команда проекцией ВПРОТИВ базового члена ППН
    N·(ω_ЛВ × v_ц) — доворот против вращения линии визирования на ракету."""
    t = Body(p=np.zeros(3), v=np.array([260.0, 0.0, 0.0]))
    m = Body(p=np.array([4000.0, 1200.0, 0.0]), v=np.array([-200.0, 60.0, 0.0]))
    omega = los_omega(m.p - t.p, m.v - t.v)
    base = np.cross(omega, t.v)
    a = negpn_accel(t, m, 4.0)
    assert float(np.dot(a, base)) < -1.0, (a, base)


# ── боевая жизнь и финалы ───────────────────────────────────────────────────

def test_fuse_expired_caps_the_run() -> None:
    """Короткая боевая жизнь (fuse_life_s=5) обрывает прогон без перехвата:
    reason=fuse_expired, duel_result=evader, t_survived ≈ 5."""
    sc = _sc(aspect="tail-chase", evader_law="negpn", fuse_life_s=5.0)
    r = collect(sc, stride=100_000)
    assert r.duel and r.fuse_expired and r.duel_result == "evader"
    assert r.reason == "fuse_expired"
    assert r.t_end is not None and abs(r.t_end - 5.0) < 3 * sc.dt
    assert r.t_survived == r.t_end


def test_no_duel_behaves_exactly_as_before() -> None:
    """duel=False — поведение бит-в-бит прежнее: без дуэльных полей, fuse не
    ограничивает (прогон до t_max/перехвата), вердикта нет."""
    sc = Scenario(**{**FAIR, "aspect": "beam"})
    r = collect(sc, stride=100_000)
    assert r.reason == "hit" and r.duel is False
    assert r.duel_result is None and r.fuse_expired is False and r.t_survived is None


# ── замкнутый контур: уклонист живёт дольше и изматывает ракету ─────────────

def test_duel_beam_evaders_survive_longer_than_straight() -> None:
    """Бортовой курс, честные скорости: каждый закон-уклонист держится дольше
    прямолинейной цели и нагоняет ракете больший суммарный импульс перегрузки
    (n_int) — ровно то, за что уклонист получает очки."""
    base = collect(_sc(duel=False, aspect="beam"), stride=100_000)
    assert base.reason == "hit"
    for law in REACTION_LAWS:
        r = collect(_sc(aspect="beam", evader_law=law), stride=100_000)
        assert r.duel_result in ("evader", "missile")
        assert (r.t_end or 0.0) > base.t_end, law  # дольше живёт
        assert r.n_int > base.n_int, law  # сильнее изматывает
        assert r.n_peak > base.n_peak, law  # гонит ракету к насыщению


def test_duel_tail_chase_away_outlives_fuse() -> None:
    """Догон: при n_target=8 и боевой жизни 30 с «уклон от точки встречи»
    выживает до истечения времени (miss больше сферы БЧ) — ракета выдыхается."""
    r = collect(_sc(aspect="tail-chase", evader_law="away", fuse_life_s=30.0), stride=100_000)
    assert r.fuse_expired and r.duel_result == "evader"
    assert r.miss_m > r.trigger_range_m
    assert abs(r.t_survived - 30.0) < 0.1


def test_evader_registry_and_unknown_law() -> None:
    """Реестр: id законов совпадают с метками словаря; неизвестный закон —
    явная ошибка, а не молчаливый straight."""
    assert set(EVADERS) == set(EVADER_LABEL)
    assert "brain" in EVADERS  # четвёртый закон — обучаемая схема, не плоская формула
    t = Body(p=np.zeros(3), v=np.array([260.0, 0.0, 0.0]))
    m = Body(p=np.array([4000.0, 0.0, 0.0]), v=np.array([-300.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        evader_accel(t, m, _sc(evader_law="kitesurfing"))
    with pytest.raises(ValueError):
        evader_accel(t, m, _sc(evader_law="brain"))  # прямая проводка мозга — через движок


def test_api_run_accepts_duel() -> None:
    """Проводка API: плоское тело /api/run с duel-полями считает прогон и
    возвращает дуэльный вердикт, совпадающий с in-process collect."""
    kw = {
        **{k: v for k, v in _sc().__dict__.items()},
        "duel": True, "aspect": "tail-chase", "evader_law": "cpa_max",
        "fuse_life_s": 8.0, "n_target": 8.0, "dt": 0.05,
    }
    out = run_once(ScenarioIn(**kw))
    assert out["duel"] is True
    assert out["fuse_expired"] is True
    assert out["duel_result"] == "evader"
    assert out["t_survived"] is not None and out["t_survived"] >= 7.5
