"""Юмор-режим «Роя»: муха-командир роя (она).

Сверка наборщика реплик (web/src/roy.ts), генератора голоса
(tools/make_voice.py), озвучки (web/src/voice.ts) и проводки (App.tsx):
  - ключи фраз совпадают в roy.ts, voice.ts и make_voice.py;
  - тексты roy.ts — дословное зеркало VOICES['roy'];
  - род глаголов: она говорит о себе по-женски;
  - для каждого ключа и варианта лежит записанный wav;
  - шина речи заводит командира вторым по старшинству владельцем с очередью,
    субтитр — до звука; события привязаны к честной метрике поколения.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROY_TS = REPO / "web" / "src" / "roy.ts"
VOICE_TS = REPO / "web" / "src" / "voice.ts"
APP_TSX = REPO / "web" / "src" / "App.tsx"
SWARM_TSX = REPO / "web" / "src" / "shell" / "SwarmWorkspace.tsx"
AUDIO = REPO / "web" / "public" / "audio"

from test_shtrum_phrases import _voices  # прецедент: разбор VOICES без импорта torch


def _phrases_ts() -> dict[str, list[str]]:
    """ROY_PHRASES из roy.ts построчным разбором (формат файла фиксированный)."""
    src = ROY_TS.read_text(encoding="utf-8")
    body = src[src.index("export const ROY_PHRASES") :]
    out: dict[str, list[str]] = {}
    key: str | None = None
    for line in body.splitlines():
        s = line.strip()
        if (m := re.fullmatch(r"(\w+): \[", s)) and key is None:
            key = m.group(1)
            out[key] = []
        elif (m := re.fullmatch(r"'(.+)',", s)) and key:
            out[key].append(m.group(1))
        elif s == "],":
            key = None
    return out


def _roy_keys_voice_ts() -> list[str]:
    src = VOICE_TS.read_text(encoding="utf-8")
    m = re.search(r"const ROY_KEYS = \[([^\]]+)\]", src)
    assert m, "ROY_KEYS не найден в voice.ts"
    return re.findall(r"'([^']+)'", m.group(1))


def test_roy_keys_match_across_sources() -> None:
    p = _phrases_ts()
    v = _voices()["roy"]
    k = _roy_keys_voice_ts()
    src = ROY_TS.read_text(encoding="utf-8")
    keys_ts = re.findall(r"'([^']+)'", re.search(r"ROY_KEYS = \[([^\]]+)\]", src).group(1))
    assert set(p) == set(k) == set(v) == set(keys_ts), "ключи разошлись: roy.ts / voice.ts / make_voice.py"
    for key, arr in p.items():
        assert len(arr) == 3, key


def test_phrases_mirror_make_voice() -> None:
    """Тексты в TS — дословное зеркало записанных в WAV строк."""
    p = _phrases_ts()
    v = _voices()["roy"]
    for key in p:
        assert p[key] == v[key], key


def test_gender_of_commander_is_female() -> None:
    """Командир — она: Я-формы женские, мужских форм о себе в наборе нет."""
    joined = " ".join(t for arr in _phrases_ts().values() for t in arr)
    for w in ("Выпустила", "Утихомирила", "Остановила"):
        assert w in joined, f"нет женской Я-формы «{w}»"
    for w in ("выпустил ", "утихомирил ", "остановил "):
        assert w not in joined, f"мужская Я-форма «{w}»"


def test_wav_clips_present_for_every_key_variant() -> None:
    p = _phrases_ts()
    for key, arr in p.items():
        for i in range(len(arr)):
            f = AUDIO / "roy" / f"{key}_{i}.wav"
            assert f.exists() and f.stat().st_size > 10_000, f


def test_variant_is_deterministic_hash() -> None:
    """Вариант — детерминированный хеш (поколение, ключ): тот же прогон — та же реплика."""
    src = ROY_TS.read_text(encoding="utf-8")
    assert "export function royVariant" in src
    assert "(gen * 31" in src and "% ROY_PHRASES[key].length" in src


def test_voice_bus_has_fourth_owner_queue() -> None:
    v = VOICE_TS.read_text(encoding="utf-8")
    assert "let royTurn: { start: () => void; since: number } | null = null" in v
    assert "else royTurn = { start: go, since: now }" in v
    assert "export function sayRoy" in v and "export function warmRoy" in v
    # командир — второй по старшинству после пилота: его очередь забирает канал первым
    assert "if (owner === 'main') void (tryTurn('roy') || tryTurn('shtrum') || tryTurn('wendy'))" in v
    # выключение звука гасит и очередь командира
    i = v.index("export function setVoiceEnabled")
    assert "royTurn = null" in v[i : i + 900]


def test_app_wires_commander_to_honest_metrics() -> None:
    src = APP_TSX.read_text(encoding="utf-8")
    # события берутся из метрик поколения, а не из воздуха
    for marker in (
        "mFactor > 1) rkey = 'stagnation'",
        "mFactor < 1) rkey = 'calm'",
        "hitR > prevHit + 1e-9) rkey = 'hits'",
        "div < 0.9 * prevDiv) rkey = 'tight'",
        "best.miss_m < prevBest - 1) rkey = 'lead'",
        "champ < prevChamp - 1e-9) rkey = 'champion'",
        "else if (validate) rkey = 'validate'",
    ):
        assert marker in src, marker
    # субтитр показывается до звука — очередь касается только аудио
    i = src.index("const showRoy")
    body = src[i : i + 600]
    assert body.index("setRoyCaption") < body.index("sayRoy")
    assert "♀ Командир роя" in src
    # прогрев клипов — тем же тумблером юмора; старт и стоп роя тоже при словах
    assert "warmRoy()" in src
    assert "showRoy(royLine('launch', 0))" in src
    assert "showRoy(royLine('stop', 99))" in src
    # космос «Рой» показывает субтитр командира
    d = SWARM_TSX.read_text(encoding="utf-8")
    assert "royCaption" in d and "♀ Командир роя" in d
