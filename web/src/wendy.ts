import type { Frame, RunMetrics, Scenario } from './types'

/**
 * Юмор-режим «Дуэли»: муха-пилот самолёта-цели — позывной «Кобра» (она).
 * Планировщик — чистая функция поверх УЖЕ посчитанных кадров, та же
 * дисциплина времени, что у штурмана (shtrum.ts): не более трёх реплик
 * в пролёте, зазор 1,5 с, финал — отдельно. Реплики выводятся из честной
 * телеметрии уклоняющейся стороны: её манёвр, чужой захват на ней,
 * чужое напряжение. Тексты и ключи сверены с tools/make_voice.py
 * (tests/test_wendy_phrases.py).
 */

export const WENDY_KEYS = ['launch', 'merge', 'break', 'weave', 'escape', 'graze', 'shot', 'fuse'] as const
export type WendyKey = (typeof WENDY_KEYS)[number]

export const WENDY_PHRASES: Record<WendyKey, string[]> = {
  launch: [
    'Взлетела. Кто тут на меня сегодня?',
    'Отрыв. Смотрим, кто кого догонит.',
    'В небе. Пусть начинают.',
  ],
  merge: [
    'Чувствую прицел. Ну, попробуй.',
    'Меня держат. Приятно, но рано.',
    'Есть захват на мне. Держись, я сейчас.',
  ],
  break: [
    'Рву в сторону! Догони.',
    'Манёвр! Кто тут цель — я или ты?',
    'Ухожу в вираж, смотри красивый.',
  ],
  weave: [
    'Каскадёрит бедняга… Жалко, но летает.',
    'Бедняга потел над моим курсом.',
    'Считает меня, представляешь. Не сосчитает.',
  ],
  escape: [
    'Есть пролёт! Я ещё здесь.',
    'Мимо. Небесный простор, а он мимо.',
    'Пролетел. Записывай, учись.',
  ],
  graze: [
    'Сквозняком! Ещё чуть-чуть — и была бы одуванчиком.',
    'Волосок! Обиднее всего промахнуться почти.',
    'Чуть не задело. Расстроюсь, если повторится.',
  ],
  shot: [
    'Взяли. В следующем заходе — я.',
    'Ловкая. Признаю — ловкая.',
    'Сбили. Передай там: я держалась.',
  ],
  fuse: [
    'Время вышло — значит, жива.',
    'Взрыватель сдался раньше меня.',
    'Не успел. Это тоже победа.',
  ],
}

export type WendyLine = { t: number; key: WendyKey; variant: number; text: string }
export type WendyPlan = { inFlight: WendyLine[]; final: WendyLine | null }

/** Исход дуэли: ракета взяла / цель ушла / цель пережила ракету (взрыватель). */
export type WendyOutcome = { result: 'missile' | 'evader' | null; fuse: boolean }

export function planWendy(frames: Frame[], m: RunMetrics, sc: Scenario, out: WendyOutcome): WendyPlan {
  const inflight: WendyLine[] = []
  if (frames.length < 4) return { inFlight: inflight, final: null }
  const tEnd = frames[frames.length - 1].t
  const h = (frames.length * 17 + Math.round(sc.t_max) + Math.round(sc.v_t)) >>> 0
  const next = (key: WendyKey, t: number) => {
    if (t > tEnd - 2.5) return
    const last = inflight[inflight.length - 1]
    if (last && t - last.t < 1.5) return
    if (inflight.length >= 3) return
    const variant = (h + inflight.length) % WENDY_PHRASES[key].length
    inflight.push({ t, key, variant, text: WENDY_PHRASES[key][variant] })
  }

  next('launch', frames[0].t + 0.4)

  // захват ракеты НА ней — она его чувствует: первый lock в пролёте
  const firstLock = frames.find((fr) => fr.lock)
  if (firstLock) next('merge', firstLock.t + 0.5)

  // её собственный манёвр: максимальная угловая скорость разворота вектора скорости
  let bestTurn = 0
  let bestTurnT = -1
  for (let i = 1; i < frames.length; i += 1) {
    const a = frames[i - 1].target_v
    const b = frames[i].target_v
    const la = Math.hypot(a[0], a[1], a[2])
    const lb = Math.hypot(b[0], b[1], b[2])
    if (la < 1 || lb < 1) continue
    const cos = Math.max(-1, Math.min(1, (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (la * lb)))
    const w = Math.acos(cos) / Math.max(0.001, frames[i].t - frames[i - 1].t)
    if (w > bestTurn) {
      bestTurn = w
      bestTurnT = frames[i].t
    }
  }
  if (bestTurnT > 0 && bestTurn > 0.2) next('break', bestTurnT + 0.3)

  // чужое напряжение: пик требуемой перегрузки ракеты во второй половине пролёта
  let peak = 0
  let peakT = -1
  for (const fr of frames) {
    if (fr.t < 0.5 * tEnd) continue
    if (fr.n_req > peak) {
      peak = fr.n_req
      peakT = fr.t
    }
  }
  if (peakT > 0 && peak > Math.max(2, 0.7 * sc.n_max)) next('weave', peakT + 0.4)

  // финал — по честному исходу, грамматически только её пол
  const key: WendyKey =
    out.result === 'missile' ? 'shot' : out.result === 'evader' ? (out.fuse ? 'fuse' : 'escape') : 'escape'
  const text = WENDY_PHRASES[key][(h + 1) % WENDY_PHRASES[key].length]
  return { inFlight: inflight, final: { t: tEnd, key, variant: 0, text } }
}
