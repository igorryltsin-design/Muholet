"""Юмор-режим «Дуэли»: муха-пилот самолёта-цели — позывной «Кобра» (она).

Сверка планировщика реплик (web/src/wendy.ts), генератора голоса
(tools/make_voice.py), озвучки (web/src/voice.ts) и проводки (App.tsx):
  - ключи фраз совпадают во всех трёх источниках;
  - тексты wendy.ts — дословное зеркало VOICES['wendy'];
  - род глаголов: она говорит о себе по-женски;
  - для каждого ключа и варианта лежит записанный wav;
  - шина речи заводит «Кобру» третьим владельцем с очередью, субтитр — до звука.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WENDY_TS = REPO / "web" / "src" / "wendy.ts"
VOICE_TS = REPO / "web" / "src" / "voice.ts"
APP_TSX = REPO / "web" / "src" / "App.tsx"
DUEL_TSX = REPO / "web" / "src" / "shell" / "DuelWorkspace.tsx"
MAKE_VOICE = REPO / "tools" / "make_voice.py"
AUDIO = REPO / "web" / "public" / "audio"

from test_shtrum_phrases import _voices  # прецедент: разбор VOICES без импорта torch


def _phrases_ts() -> dict[str, list[str]]:
    """WENDY_PHRASES из wendy.ts построчным разбором (формат файла фиксированный)."""
    src = WENDY_TS.read_text(encoding="utf-8")
    body = src[src.index("export const WENDY_PHRASES") :]
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


def _wendy_keys_voice_ts() -> list[str]:
    src = VOICE_TS.read_text(encoding="utf-8")
    m = re.search(r"const WENDY_KEYS = \[([^\]]+)\]", src)
    assert m, "WENDY_KEYS не найден в voice.ts"
    return re.findall(r"'([^']+)'", m.group(1))


def test_wendy_keys_match_across_sources() -> None:
    p = _phrases_ts()
    v = _voices()["wendy"]
    k = _wendy_keys_voice_ts()
    src = WENDY_TS.read_text(encoding="utf-8")
    keys_ts = re.findall(r"'([^']+)'", re.search(r"WENDY_KEYS = \[([^\]]+)\]", src).group(1))
    assert set(p) == set(k) == set(v) == set(keys_ts), "ключи разошлись: wendy.ts / voice.ts / make_voice.py"
    for key, arr in p.items():
        assert len(arr) == 3, key


def test_phrases_mirror_make_voice() -> None:
    """Тексты в TS — дословное зеркало записанных в WAV строк."""
    p = _phrases_ts()
    v = _voices()["wendy"]
    for key in p:
        assert p[key] == v[key], key


def test_gender_of_cobra_is_female() -> None:
    """«Кобра» — она: Я-формы женские, мужских форм о себе в наборе нет.
    («Не успел» в финале — про взрыватель/ракету, не про неё.)"""
    joined = " ".join(t for arr in _phrases_ts().values() for t in arr)
    for w in ("Взлетела", "держалась", "Расстроюсь", "значит, жива"):
        assert w in joined, f"нет женской Я-формы «{w}»"
    for w in ("взлетел", "держался", "признаю себя"):
        assert not re.search(rf"\b{w}\b", joined), f"мужская Я-форма «{w}»"


def test_wav_clips_present_for_every_key_variant() -> None:
    p = _phrases_ts()
    for key, arr in p.items():
        for i in range(len(arr)):
            f = AUDIO / "wendy" / f"{key}_{i}.wav"
            assert f.exists() and f.stat().st_size > 10_000, f


def test_time_discipline_same_as_shtrum() -> None:
    """Та же дисциплина, что у штурмана: ≤3 реплики, зазор 1,5 с, ≥2,5 с до конца."""
    src = WENDY_TS.read_text(encoding="utf-8")
    assert "t > tEnd - 2.5" in src
    assert "t - last.t < 1.5" in src
    assert "inflight.length >= 3" in src


def test_voice_bus_has_third_owner_queue() -> None:
    v = VOICE_TS.read_text(encoding="utf-8")
    assert "let wendyTurn: { start: () => void; since: number } | null = null" in v
    assert "else wendyTurn = { start: go, since: now }" in v
    # освободившийся канал переходит по очереди между напарницами
    assert "type SpeakerOwner = 'main' | 'roy' | 'shtrum' | 'wendy'" in v
    assert "if (owner === 'main') void (tryTurn('roy') || tryTurn('shtrum') || tryTurn('wendy'))" in v
    # выключение звука гасит и очередь «Кобры»
    i = v.index("export function setVoiceEnabled")
    assert "wendyTurn = null" in v[i : i + 900]


def test_app_wires_cobra_only_in_duel() -> None:
    src = APP_TSX.read_text(encoding="utf-8")
    # план рождается только когда есть исход дуэли (duelInfo) — и в стриме, и в POST
    assert "if (humorOn && duelInfo) wendyPlan = planWendy(frames, m, flown, duelInfo)" in src
    assert "humorOn && duelInfo ? planWendy(frames, m, s, duelInfo) : null" in src
    # субтитр показывается до звука — очередь касается только аудио
    i = src.index("const showWendy")
    body = src[i : i + 600]
    assert body.index("setWendyCaption") < body.index("sayWendy")
    assert "♀ Кобра" in src
    # прогрев клипов — тем же тумблером юмора
    assert "warmWendy()" in src
    # дуэльный космос показывает оба субтитра напарниц
    d = DUEL_TSX.read_text(encoding="utf-8")
    assert "wendyCaption" in d and "♀ Кобра" in d and "shtrumCaption" in d
