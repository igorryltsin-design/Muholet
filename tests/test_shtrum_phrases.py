"""Юмор-режим «вторая муха-штурман»: сверка фронтового планировщика реплик
(web/src/shtrum.ts), генератора голоса (tools/make_voice.py) и озвучки (voice.ts).

Реплики обязаны быть честными и согласованными по полу:
  - ключи фраз совпадают во всех трёх источниках;
  - тексты shtrum.ts — дословное зеркало VOICES['shtrum_m'|'shtrum_f'];
  - род глаголов: говорящий о себе — в своём роде (он «говорил», она «говорила»),
    обращения — к противоположному полу (пилоту основной мухе);
  - для каждого ключа и варианта лежит записанный wav в обеих папках;
  - App.tsx пробрасывает пол напарника (он = противоположный текущему голосу).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHTRUM_TS = REPO / "web" / "src" / "shtrum.ts"
VOICE_TS = REPO / "web" / "src" / "voice.ts"
APP_TSX = REPO / "web" / "src" / "App.tsx"
MAKE_VOICE = REPO / "tools" / "make_voice.py"
AUDIO = REPO / "web" / "public" / "audio"


def _voices() -> dict[str, dict[str, list[str]]]:
    """VOICES из make_voice.py без импорта torch — literal_eval дерева."""
    tree = ast.parse(MAKE_VOICE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id == "VOICES":
                    raw = ast.literal_eval(node.value)
                    return {k: v[1] for k, v in raw.items()}
    raise AssertionError("VOICES не найден в make_voice.py")


def _phrases_ts() -> dict[str, dict[str, list[str]]]:
    """PHRASES из shtrum.ts построчным разбором (формат файла фиксированный)."""
    src = SHTRUM_TS.read_text(encoding="utf-8")
    body = src[src.index("export const PHRASES") :]
    out: dict[str, dict[str, list[str]]] = {}
    key: str | None = None
    gender: str | None = None
    for line in body.splitlines():
        s = line.strip()
        if (m := re.fullmatch(r"(\w+): \{", s)) and key is None:
            key, gender = m.group(1), None
            out[key] = {}
        elif (m := re.fullmatch(r"(m|f): \[", s)) and key:
            gender = m.group(1)
            out[key][gender] = []
        elif (m := re.fullmatch(r"'(.+)',", s)) and key and gender:
            out[key][gender].append(m.group(1))
        elif s in ("},", "]},"):
            key, gender = None, None
    return out


def _shtrum_keys_voice_ts() -> list[str]:
    src = VOICE_TS.read_text(encoding="utf-8")
    m = re.search(r"const SHTRUM_KEYS = \[([^\]]+)\]", src)
    assert m, "SHTRUM_KEYS не найден в voice.ts"
    return re.findall(r"'([^']+)'", m.group(1))


def test_shtrum_keys_match_across_sources() -> None:
    v = _voices()
    p = _phrases_ts()
    k = _shtrum_keys_voice_ts()
    assert set(p) == set(k) == set(v["shtrum_m"]) == set(v["shtrum_f"]), (
        "ключи фраз разошлись: shtrum.ts / SHTRUM_KEYS / VOICES"
    )
    for key, pair in p.items():
        assert set(pair) == {"m", "f"}, key
        assert len(pair["m"]) == len(pair["f"]) == 3, key


def test_phrases_mirror_make_voice() -> None:
    """Тексты в TS — дословное зеркало записанных в WAV строк."""
    v = _voices()
    p = _phrases_ts()
    for key in p:
        assert p[key]["m"] == v["shtrum_m"][key], key
        assert p[key]["f"] == v["shtrum_f"][key], key


# Род согласуется ДВУМЯ линиями: Я-формы — по роду говорящего, обращения ко 2-му
# лицу — по роду пилота (он противоположен напарнику). Потому в мужском наборе
# («он»-Штруман, пилот — она) законны «Слышала/Могла/Упустила/самовольничала»,
# а Я-формы обязаны быть мужскими, и наоборот для женского набора.
def _joined(p: dict, which: str) -> str:
    return " ".join(t for pair in p.values() for t in pair[which])


def _has(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def test_gender_of_speaker_forms() -> None:
    p = _phrases_ts()
    m, f = _joined(p, "m"), _joined(p, "f")
    # Я-формы: он говорит о себе по-мужски, она — по-женски
    for w in ("говорил", "подсказывал", "вёл", "хвалил", "верил"):
        assert _has(m, w), f"мужской набор: нет Я-формы «{w}»"
        assert not _has(f, w), f"женский набор: мужская Я-форма «{w}»"
    for w in ("говорила", "подсказывала", "вела", "хвалила", "верила"):
        assert _has(f, w), f"женский набор: нет Я-формы «{w}»"
        assert not _has(m, w), f"мужской набор: женская Я-форма «{w}»"
    # обращения ко 2-му лицу (пилот opposite пола): он зовёт её, она — его
    for w in ("Слышала", "Могла", "самовольничала"):
        assert _has(m, w) and not _has(f, w), f"женское обращение «{w}» не на месте"
    for w in ("Слышал", "Мог", "самовольничал"):
        assert _has(f, w) and not _has(m, w), f"мужское обращение «{w}» не на месте"



def test_accusative_lost_matches_pilot_gender() -> None:
    """«Упустил(а)!» — про пилота основной мухи: он (Штруман) говорит ей «упустила»."""
    p = _phrases_ts()
    assert p["lost"]["m"][0].startswith("Упустила")
    assert p["lost"]["f"][0].startswith("Упустил")


def test_graze_and_impact_angle_are_metric_driven() -> None:
    """«Волосок» включается только по метрике (промах < 2.5 радиусов БЧ), числит
    честный недостающий метраж, а финал перехвата уточняется углом встречи η."""
    src = SHTRUM_TS.read_text(encoding="utf-8")
    assert "m.miss < 2.5 * m.triggerRangeM" in src
    assert "m.miss - m.triggerRangeM" in src
    assert "key = 'graze'" in src
    # η-причёмка: лоб ≥150°, догон ≤30°, боковая 60…120°
    assert "eta >= 150" in src and "eta <= 30" in src and "eta >= 60 && eta <= 120" in src
    assert "m.impactAngleM" in src


def test_partner_voice_waits_for_pilot_to_finish() -> None:
    """Голоса мух разведены по времени единой шиной речи: штурман, пока пилот
    договаривает, встаёт в очередь (с TTL), а не накладывается на её фразу;
    субтитр при этом показывается сразу — очередь касается только звука."""
    v = VOICE_TS.read_text(encoding="utf-8")
    # один канал занят говорющим: есть владелец (main/shtrum) и очередь штурмана
    assert "type SpeakerOwner = 'main' | 'shtrum'" in v
    assert "shtrumTurn" in v
    # очередь сгорает по TTL (устаревшая реплика не ожилает через пол-прогона)
    assert "TURN_TTL" in v and "Date.now() - t.since <= TURN_TTL" in v
    # штурман при занятом канале НЕ стартует источник сразу
    assert "if (free) go()" in v and "else shtrumTurn = { start: go, since: now }" in v
    # субтитр не зависит от очереди: App показывает текст до вызова sayShtrum
    a = APP_TSX.read_text(encoding="utf-8")
    i = a.index("const showShtrum")
    body = a[i : i + 700]
    assert body.index("setShtrumCaption") < body.index("sayShtrum")


def test_wav_clips_present_for_every_key_variant() -> None:
    p = _phrases_ts()
    for which in ("m", "f"):
        for key, pair in p.items():
            for i in range(len(pair[which])):
                f = AUDIO / f"shtrum_{which}" / f"{key}_{i}.wav"
                assert f.exists() and f.stat().st_size > 10_000, f


def test_app_wires_partner_gender() -> None:
    src = APP_TSX.read_text(encoding="utf-8")
    # пол напарника — противоположный текущему голосу, фиксируется на старте прогона
    assert re.search(r"const he = voiceKind === 'female'", src)
    # пол напарника доходит до планировщика во всех вызовах: стрим летает по
    # фактическому сценарию (flown), POST/офлайн — по активному (s, надбавка патча)
    assert "planShtrum(frames, m, s, he)" in src and "planShtrum(frames, m, flown, he)" in src
    assert "sayShtrum(line.key, he, line.variant)" in src
    assert "♂ Штруман" in src and "♀ Штрумана" in src
    # режим по умолчанию выключен, переживает перезагрузку (чтение — в инициализаторе
    # useState: эффект записи в dev-StrictMode иначе затирает хранилище дефолтами)
    assert "useState(savedVoice.humor === true)" in src
    assert "humor: humorOn" in src
