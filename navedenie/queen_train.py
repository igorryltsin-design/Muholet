"""Самообучение на сервере: Красная королева воюет без открытой вкладки.

Цикл поколений крутился фронтом — закрой вкладку, и гонка вооружений встаёт.
Здесь тот же цикл (поколение за поколением через redqueen.queen_generation, те
же seed и порядок) живёт в фоновом потоке стенда: запуск одним POST, прогресс
читается вторым, а по готовности чемпионы сами садятся за штурвалы и ставятся на
полку ринга. Одна задача за раз: вторая попытка — QueenTrainBusy (409 честнее,
чем молча подменить чужой прогон). Детерминизм наследуется от поколения: тот же
seed и то же число поколений → бит-в-бит те же чемпионы. Доигранная кампания
заходит в хронику (`queen_chronicle`), чтобы память о войне не умирала вместе с
процессом. С `inherit` война начинается не с чистого листа: часть стартовой
популяции прививается из выученных дуэтов полки ринга (`queen_ring.heritage`),
так что сила копится от кампании к кампании. С `auto_ring` сезон замыкается:
немного не доживший до конца бой не сводится, а досчитанная кампания тут же
гоняет круговой самобой сильнейших с полки вместе со своим новичком — форма
полки (и её движение) обновляется без ручного клика, и следующая война стартует
уже от неё.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from navedenie.redqueen import apply_champions, init_brain_population, queen_generation
from navedenie.sim import Scenario
from navedenie.swarm import fly_from_json

_LOCK = threading.Lock()
_JOB: dict[str, Any] | None = None
_THREAD: threading.Thread | None = None


class QueenTrainBusy(RuntimeError):
    """На стенде уже воюет другая королева."""


def _shape(job: dict[str, Any] | None) -> dict[str, Any]:
    """Снимок всегда одной формы — и когда задачи никогда не было: фронту не
    приходится гадать, пустой это ответ или ещё не нарисовался."""
    job = job or {}
    return {
        "running": bool(job.get("running")),
        "generations": int(job.get("generations", 0)),
        "generations_done": int(job.get("generations_done", 0)),
        "pop": int(job.get("pop", 0)),
        "seed": int(job.get("seed", 0)),
        "scenario": dict(job.get("scenario") or {}),
        "inherit": list(job.get("inherit") or []),
        "auto_ring": bool(job.get("auto_ring", False)),
        "save_duel": bool(job.get("save_duel", False)),
        "apply": bool(job.get("apply", False)),
        "log": list(job.get("log") or []),
        "champions": job.get("champions"),
        "saved": dict(job.get("saved") or {"ring": None, "weights": None}),
        "error": job.get("error"),
        "stop_requested": bool(job.get("stop_requested")),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "seconds": round((job.get("finished_at") or time.time()) - job["started_at"], 2) if job.get("started_at") else 0.0,
    }


def status() -> dict[str, Any]:
    with _LOCK:
        return _shape(_JOB)


def start(
    sc: Scenario,
    *,
    generations: int = 6,
    pop: int = 4,
    seed: int = 7,
    save_duel: bool = True,
    apply: bool = True,
    inherit: list[int] | None = None,
    auto_ring: bool = False,
) -> dict[str, Any]:
    """Запустить фоновую гонку вооружений; возвращает снимок стартовавшей задачи.

    `inherit` — id дуэтов с полки ринга: с ними кампания стартует не с врождённого
    рефлекса, а с выученных мозгов (см. `queen_ring.heritage`).
    `auto_ring` — по окончании досчитанной кампании тут же свести круговой бой
    сильнейших полки с новым чемпионом (см. `_season_battle`)."""
    global _JOB, _THREAD
    gens = max(1, int(generations))
    n = max(2, int(pop))
    heirs = [int(i) for i in (inherit or [])][:4]
    if heirs:
        # Чужой id проверяется до подъёма потока: иначе «наследие» молча сгорело
        # бы в error'е задачи, а фронт показал бы пустую войну.
        from navedenie.queen_ring import list_duels

        have = {int(d.get("id", -1)) for d in list_duels()}
        for i in heirs:
            if i not in have:
                raise ValueError(f"на полке ринга нет дуэта {i}")
    with _LOCK:
        if _JOB is not None and _JOB.get("running"):
            raise QueenTrainBusy("королева уже воюет: дождись окончания или останови")
        job: dict[str, Any] = {
            "running": True,
            "generations": gens,
            "pop": n,
            "seed": int(seed),
            "scenario": {"aspect": sc.aspect, "range_m": sc.range_m, "v_t": sc.v_t, "off_axis_m": sc.off_axis_m},
            "inherit": heirs,
            "auto_ring": bool(auto_ring),
            "save_duel": bool(save_duel),
            "apply": bool(apply),
            "generations_done": 0,
            "log": [],
            "champions": None,
            "saved": {"ring": None, "weights": None},
            "error": None,
            "stop_requested": False,
            "started_at": time.time(),
            "finished_at": None,
        }
        _JOB = job
        _THREAD = threading.Thread(target=_run, args=(job, sc), name="queen-train", daemon=True)
        _THREAD.start()
        return _shape(job)


def stop() -> dict[str, Any]:
    """Просить остановку: поток доделывает текущее поколение и сворачивается."""
    with _LOCK:
        if _JOB is not None and _JOB.get("running"):
            _JOB["stop_requested"] = True
        return _shape(_JOB)


def wait(timeout: float | None = None) -> dict[str, Any]:
    """Дождаться завершения задачи (для тестов и синхронных сценариев)."""
    thread = _THREAD
    if thread is not None and thread.is_alive():
        thread.join(timeout)
    with _LOCK:
        return _shape(_JOB)


def _finish(job: dict[str, Any], sc: Scenario) -> None:
    """Чемпионы последнего досчитанного поколения — в мозг и на полку ринга."""
    champs = job.get("champions")
    if not champs:
        return
    if job["apply"]:
        try:
            job["saved"]["weights"] = apply_champions(missile=champs["missile"], evader=champs["evader"])
        except Exception as exc:  # посадка не удалась — бой всё равно засчитан, веса останутся прежними
            job["error"] = f"применение чемпионов: {type(exc).__name__}: {exc}"
    if job["save_duel"]:
        try:
            from navedenie.queen_ring import save_duel

            label = f"Королева ×{job['generations_done']} (сервер, seed {job['seed']})"
            job["saved"]["ring"] = save_duel(label, champs["missile"], champs["evader"])
        except Exception as exc:
            job["error"] = job["error"] or f"полка ринга: {type(exc).__name__}: {exc}"


def _season_battle(job: dict[str, Any], sc: Scenario) -> None:
    """Финал сезона: круговой самобой сильнейших полки вместе с новичком этой
    кампании. Берём leaders по последней форме (не больше трёх) и добавляем дуэт,
    только что вставший на полку, — всего не больше четырёх, иначе фон превратился
    бы в многочасовой марафон. Свод пишет форму полки (и её движение) сам, поэтому
    следующая кампания с inherit растёт уже от свежих рангов. Прерванная по «Стоп»
    или без дуэтов на полке война не сводится: сводить нечем или не тот честный итог."""
    from navedenie.queen_ring import best_duels, list_duels, ring_battle

    shelf = list_duels()
    if len(shelf) < 2:
        return
    ids: list[int] = []
    for cid in best_duels(3)["ids"]:
        if cid not in ids:
            ids.append(int(cid))
    fresh = (job.get("saved") or {}).get("ring")
    if fresh is not None and int(fresh["id"]) not in ids:
        ids.append(int(fresh["id"]))
    ids = ids[:4]
    subset = [d for d in shelf if int(d.get("id", -1)) in ids]
    if len(subset) < 2:
        return
    out = ring_battle(sc, subset)
    job["saved"]["ring_battle"] = {
        "duels": [{"id": int(d["id"]), "label": str(d["label"])} for d in out["duels"]],
        "standings": [
            {
                "id": int(s["id"]),
                "label": str(s["label"]),
                "rank": int(s["rank"]),
                "score": int(s["score"]),
                "p_attack": float(s["p_attack"]),
                "p_defense": float(s["p_defense"]),
            }
            for s in out["standings"]
        ],
    }


def _run(job: dict[str, Any], sc: Scenario) -> None:
    try:
        if job.get("inherit"):
            # наследие полки: те же слоты старта, но часть из них — выученные мозги
            from navedenie.queen_ring import heritage

            heirs = heritage(job["inherit"], job["pop"], job["seed"])
            mp = [fly_from_json(x) for x in heirs["missile"]]
            ep = [fly_from_json(x) for x in heirs["evader"]]
        else:
            mp = init_brain_population(job["pop"], seed=job["seed"])
            ep = init_brain_population(job["pop"], seed=job["seed"] + 1)
        for g in range(job["generations"]):
            if job["stop_requested"]:
                break
            t0 = time.monotonic()
            out = queen_generation(sc, mp, ep, gen=g, seed=job["seed"])
            mp = [fly_from_json(x) for x in out["missile_population"]]
            ep = [fly_from_json(x) for x in out["evader_population"]]
            with _LOCK:
                job["champions"] = out["champions"]
                job["generations_done"] = g + 1
                job["log"].append(
                    {
                        "gen": g,
                        "p_hit_ring": out["stats"]["p_hit_ring"],
                        "exam_p_hit": out["exam"]["p_hit"],
                        "missile_best": out["stats"]["missile_best"],
                        "evader_best": out["stats"]["evader_best"],
                        "t_survived_median": out["exam"]["t_survived_median"],
                        "cpa_m_median": out["exam"]["cpa_m_median"],
                        "seconds": round(time.monotonic() - t0, 2),
                    }
                )
        _finish(job, sc)
        if job.get("auto_ring") and not job.get("stop_requested"):
            # сезон замыкается без клика: свежий чемпион сразу сведён с полкой
            try:
                _season_battle(job, sc)
            except Exception as exc:  # свод не имеет права унести за собой кампанию
                job["error"] = job["error"] or f"свод ринга: {type(exc).__name__}: {exc}"
    except Exception as exc:  # фоновый поток не должен уносить собой стенд
        with _LOCK:
            job["error"] = job["error"] or f"{type(exc).__name__}: {exc}"
    finally:
        with _LOCK:
            job["running"] = False
            job["finished_at"] = time.time()
            try:
                from navedenie.queen_chronicle import record

                record(_shape(job))
            except Exception as exc:  # хроника не имеет права унести за собой бой
                job["error"] = job["error"] or f"хроника: {type(exc).__name__}: {exc}"
