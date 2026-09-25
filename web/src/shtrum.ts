import type { Frame, RunMetrics, Scenario } from './types'

/**
 * Юмор-режим: вторая муха-штурман противоположного полу основной.
 * Планировщик реплик — чистая функция поверх УЖЕ посчитанных кадров:
 * физика и мозг не трогаются, реплики выводятся из настоящей телеметрии.
 * Тексты и ключи сверены с tools/make_voice.py (tests/test_shtrum_phrases.py).
 */

export type ShtrumLine = { t: number; key: string; variant: number; text: string }
export type ShtrumPlan = { inFlight: ShtrumLine[]; final: ShtrumLine | null }

/** he=true — «Штруман» (он, говорит о себе «говорил», зовёт «слышала»). */
export function planShtrum(frames: Frame[], m: RunMetrics, sc: Scenario, he: boolean): ShtrumPlan {
  const inflight: ShtrumLine[] = []
  if (frames.length < 4) return { inFlight: inflight, final: null }
  const tEnd = frames[frames.length - 1].t
  // детерминированный выбор варианта: один сценарий — одна и та же фраза
  const h = (frames.length * 31 + Math.round(sc.t_max) + Math.round(sc.v_m)) >>> 0
  const next = (key: string, t: number) => {
    if (t > tEnd - 2.5) return // не успеем произнести до конца пролёта
    const last = inflight[inflight.length - 1]
    if (last && t - last.t < 1.5) return
    if (inflight.length >= 3) return
    const variant = (h + inflight.length) % 3
    inflight.push({ t, key, variant, text: PHRASES[key][he ? 'm' : 'f'][variant] })
  }

  next('send', frames[0].t + 0.3)

  // «куда смотреть»: к 30 % пролёта захвата нет — подсказка по истинному
  // смещению цели относительно носа (свой знак считаем здесь, не у seeker-а)
  const firstLock = frames.findIndex((fr) => fr.lock)
  if (firstLock === -1 || frames[firstLock].t > 0.3 * tEnd) {
    const i = Math.max(2, Math.floor(frames.length * 0.3))
    const fr = frames[i]
    const off = losOffset(fr)
    if (off.left > off.up && off.left > 2) next('look_left', fr.t)
    else if (off.up > 2) next('look_up', fr.t)
  }

  // потеряла захват — «упустила!»
  const lost = frames.find((fr) => fr.event === 'lost')
  if (lost) next('lost', lost.t + 0.6)

  // терминал: перегрузка на упоре или растущий прогноз промаха
  for (const fr of frames) {
    if (fr.t < 0.75 * tEnd) continue
    if (fr.n_lim > 0 && fr.n_req >= 0.9 * fr.n_lim) {
      next('overload', fr.t)
      break
    }
  }

  let key = m.hit ? 'hit' : 'miss'
  const variant = h % 3
  let text = PHRASES[key][he ? 'm' : 'f'][variant]
  if (!m.hit) {
    // «волосок»: промах в пределах 2.5 радиусов БЧ — это досада, а не ворчание;
    // честная цифра — сколько метров не хватило до сферы срабатывания
    if (m.triggerRangeM > 0 && m.miss < 2.5 * m.triggerRangeM) {
      key = 'graze'
      text = PHRASES.graze[he ? 'm' : 'f'][variant]
      text += ` Не хватило ${Math.max(1, Math.round(m.miss - m.triggerRangeM))} м.`
    } else {
      // причина — по-честному из метрик; глаголы про пилотшу (её пол — наоборот напарника)
      const lateLost = lost && lost.t > 0.7 * tEnd
      const cause =
        m.lockFrac === 0 ? (he ? ' Цель вообще не взяла.' : ' Цель вообще не взял.')
        : m.satFrac > 0.3 ? ' Перегрузки не хватило.'
        : lateLost ? (he ? ' Уронила в самом конце.' : ' Уронил в самом конце.')
        : tEnd >= sc.t_max - 0.5 ? (he ? ' Не успела — далеко.' : ' Не успел — далеко.')
        : ''
      text += cause
    }
  } else {
    // угол встречи — та же метрика η, что в сводке: 180° лоб, 0° догон
    const eta = m.impactAngleM
    if (eta != null) {
      text +=
        eta >= 150 ? ' В лоб — по-честному.'
        : eta <= 30 ? ' В догон — и в точку.'
        : eta >= 60 && eta <= 120 ? ' Боком — красивее не бывает.'
        : ''
    }
  }
  return { inFlight: inflight, final: { t: tEnd + 0.2, key, variant, text } }
}

/** Смещение цели от носа ракеты в кадре, град.: «левее» и «выше» (мир: X —
 *  вперёд, Y — бок в плюс, Z — вверх; левая сторона курса = ẑ × v̂). */
function losOffset(fr: Frame): { left: number; up: number } {
  const r = [fr.target[0] - fr.missile[0], fr.target[1] - fr.missile[1], fr.target[2] - fr.missile[2]]
  const v = fr.missile_v
  const vn = Math.hypot(v[0], v[1], v[2]) || 1
  const rN = Math.hypot(r[0], r[1], r[2]) || 1
  const c = [-v[1], v[0], 0] // ẑ × v — вектор «левее от курса»
  const cn = Math.hypot(c[0], c[1], c[2]) || 1
  const sLeft = (r[0] * c[0] + r[1] * c[1] + r[2] * c[2]) / (rN * cn)
  const left = Math.asin(Math.max(-1, Math.min(1, sLeft))) / (Math.PI / 180)
  const up =
    Math.asin(Math.max(-1, Math.min(1, r[2] / rN))) / (Math.PI / 180) -
    Math.asin(Math.max(-1, Math.min(1, v[2] / vn))) / (Math.PI / 180)
  return { left, up }
}

/** Тексты-пары (m — говорит он, f — она); дословно повторяют VOICES['shtrum_m']
 *  и VOICES['shtrum_f'] в tools/make_voice.py — сверено тестом. */
export const PHRASES: Record<string, { m: string[]; f: string[] }> = {
  send: {
    m: [
      'Слышала, куда лететь? Повтори вслух.',
      'Лети-лети, я смотрю. И не поминай лихом.',
      'Напоминаю: штурман здесь я.',
    ],
    f: [
      'Слышал, куда лететь? Повтори вслух.',
      'Лети-лети, я смотрю. И не поминай лихом.',
      'Напоминаю: штурман здесь я.',
    ],
  },
  look_left: {
    m: [
      'Слево! Слево смотри, там она!',
      'Левее бери, левее — вижу!',
      'Она слева от трассы, разворачивайся.',
    ],
    f: [
      'Слево! Слево смотри, там она!',
      'Левее бери, левее — вижу!',
      'Она слева от трассы, разворачивайся.',
    ],
  },
  look_up: {
    m: [
      'Выше! Цель выше, задери нос!',
      'Смотри вверх, она на высоте!',
      'Не в пол смотри, в потолок!',
    ],
    f: [
      'Выше! Цель выше, задери нос!',
      'Смотри вверх, она на высоте!',
      'Не в пол смотри, в потолок!',
    ],
  },
  overload: {
    m: [
      'Дай перегрузку, уходит!',
      'Поддай, поддай — а то упустим!',
      'Не жалей крыльев, догоняй!',
    ],
    f: [
      'Дай перегрузку, уходит!',
      'Поддай, поддай — а то упустим!',
      'Не жалей крыльев, догоняй!',
    ],
  },
  lost: {
    m: [
      'Упустила! Ищи, ищи!',
      'Опять потеряли. Глаза разуй.',
      'Это не физика виновата.',
    ],
    f: [
      'Упустил! Ищи, ищи!',
      'Опять потеряли. Глаза разуй.',
      'Это не физика виновата.',
    ],
  },
  hit: {
    m: [
      'Молодец… а кто тебе говорил, куда лететь?',
      'Есть перехват! Теперь — домой.',
      'Вот. Могла, когда я подсказывал.',
    ],
    f: [
      'Молодец… А я же говорила, куда лететь!',
      'Есть перехват! Теперь — домой.',
      'Вот. Мог, когда я подсказывала.',
    ],
  },
  miss: {
    m: [
      'Я же тебе говорил, куда лететь!',
      'Говорил я: конём ходи — век воли не видать!',
      'Мимо. Я же вёл, а ты самовольничала.',
    ],
    f: [
      'Я же тебе говорила, куда лететь!',
      'Говорила я: конём ходи — век воли не видать!',
      'Мимо. Я же вела, а ты самовольничал.',
    ],
  },
  graze: {
    m: [
      'Волосок! Ещё чуть-чуть — и хвалил бы.',
      'Разошлись в волосок. Бывает лишь с теми, кто летает.',
      'Ну волосатый же промах! Я-то верил.',
    ],
    f: [
      'Волосок! Ещё чуть-чуть — и хвалила бы.',
      'Разошлись в волосок. Бывает лишь с теми, кто летает.',
      'Ну волосатый же промах! Я-то верила.',
    ],
  },
}
