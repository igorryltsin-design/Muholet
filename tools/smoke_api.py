"""Смоук-проверка API стенда: прогон, робастность, обучение, пересоздание мозга.

Запуск: .venv/bin/python tools/smoke_api.py [http://127.0.0.1:8091] [--destructive]

Без флага --destructive НЕ трогает состояние стенда: пересоздание мозга
(оно перезаписывает data/weights_*.npz!) выполняется только явно.
"""
import json
import sys
import urllib.request

ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
DESTRUCTIVE = "--destructive" in sys.argv[1:]
BASE = ARGS[0] if ARGS else "http://127.0.0.1:8091"


def call(path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}, method="POST" if payload is not None else "GET"
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def check(cond: bool, what: str) -> None:
    print(("✓" if cond else "✗") + " " + what)
    if not cond:
        sys.exit(1)


check(call("/api/health").get("ok") == "navedenie", "здоровье стенда")

# /api/run принимает плоский ScenarioIn (как и фронтенд: {...sc}); вложенная
# «scenario» здесь молча игнорировалась бы pydantic — проверялся бы дефолт
run = call("/api/run", {"range_m": 6000, "t_max": 12, "mode": "pn", "law": "pn"})
check(run["hit"] is True and run["t_guide"] is not None, "ПН: перехват и время наведения")
check(run["ref_nrms"] is not None and run["ref_nrms"] < 0.05, "ПН: траектория совпадает с эталоном")

rob = call("/api/robustness", {"scenario": {"range_m": 6000, "t_max": 10}, "kinds": ["stub"], "noise_levels": [0, 2]})
check(len(rob["levels"]) == 2 and len(rob["series"][0]["miss"]) == 2, "робастность: ряды по уровням шума")

if DESTRUCTIVE:
    reb = call("/api/brain/rebuild", {"kind": "connectome", "channels": 32})
    check(reb["ok"] is True, "пересоздание коннектома (32 канала)")
else:
    print("– пересоздание пропущено (деструктивно: перезаписывает веса; флаг --destructive)")

tr = call("/api/train/start", {"kind": "stub", "episodes": 4})
step = call("/api/train/step", {"kind": "stub", "ep": 0})
check("miss" in step and "ref_dev" in step and "t_guide" in step, "обучение: метрики эпизода")

sci = call("/api/transfer", {"kind": "stub", "episodes": 2})
check(len(sci["rows"]) == 4 and all("tests" in r for r in sci["rows"]), "матрица переносимости: строки и испытания")

scl = call("/api/scaling", {"kind": "full", "sizes": [8, 16], "episodes": 2})
check(len(scl["rows"]) == 2 and scl["rows"][0]["params"] < scl["rows"][1]["params"], "scaling: параметры растут с размером")

rep = call("/api/report/night", {"title": "Смоук", "faults": [{"fraction": 0.0, "miss": 39.0, "worst": 41.0, "hit_rate": 1.0}]})
check(rep["ok"] is True and "faults.png" in rep.get("files", []), "автоотчёт: markdown + PNG")

with urllib.request.urlopen(BASE + "/api/brain/formula/c?kind=stub", timeout=60) as r:
    c_code = r.read().decode("utf-8")
check("void dn(" in c_code and "dn_approx" in c_code, "экспорт закона в C: dn() и dn_approx()")

print("СМОУК ПРОЙДЕН")
