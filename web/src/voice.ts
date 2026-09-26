// Озвучка мухи: жужжание «в тон манёвра» (WebAudio, без файлов) и голосовые
// фразы (Silero TTS, заранее записаны tools/make_voice.py в /audio/{голос}/).
// Юмор-режим: вторая муха-штурман противоположного пола (/audio/shtrum_m|shtrum_f/),
// говорит тише первой и НЕ перекрывает её: канал речи один — штурман ждёт,
// пока пилот договорит (субтитр при этом показывается сразу).

export type VoiceKind = 'male' | 'female'

// число вариантов у каждого голоса: пилот — пять (дуэль 1-на-1, реплик много не бывает),
// штурман — четыре, «Кобра» — три; записаны в tools/make_voice.py
const PILOT_VARIANTS = 5
const SHTRUM_VARIANTS = 4
const WENDY_VARIANTS = 3
const SHTRUM_KEYS = ['send', 'look_left', 'look_up', 'overload', 'lost', 'hit', 'miss', 'graze']
const WENDY_KEYS = ['launch', 'merge', 'break', 'weave', 'escape', 'graze', 'shot', 'fuse']
const ROY_KEYS = ['launch', 'new_geo', 'lead', 'stagnation', 'validate', 'champion', 'hits', 'tight', 'calm', 'stop']
const COOLDOWNS: Record<string, number> = {
  launch: 2000,
  capture: 2500,
  lost: 6000,
  hit: 1500,
  miss: 1500,
  overload: 25000,
}
const SHTRUM_COOLDOWNS: Record<string, number> = {
  send: 2000,
  look_left: 8000,
  look_up: 8000,
  overload: 10000,
  lost: 8000,
  hit: 1500,
  miss: 1500,
}

let ctx: AudioContext | null = null
let enabled = false
let kind: VoiceKind = 'female'

// ─── жужжание ───────────────────────────────────────────────────────────────
let buzzGain: GainNode | null = null
let buzzFilter: BiquadFilterNode | null = null
const buzzOsc: OscillatorNode[] = []

function buildBuzz(c: AudioContext) {
  buzzGain = c.createGain()
  buzzGain.gain.value = 0
  buzzFilter = c.createBiquadFilter()
  buzzFilter.type = 'lowpass'
  buzzFilter.frequency.value = 900
  buzzFilter.Q.value = 2
  buzzGain.connect(buzzFilter)
  buzzFilter.connect(c.destination)
  // два слегка расстроенных пилообразных осциллятора — «комарино-мушиный» тембр
  for (const detune of [0, 1.006]) {
    const osc = c.createOscillator()
    osc.type = 'sawtooth'
    osc.frequency.value = 118 * detune
    osc.connect(buzzGain)
    osc.start()
    buzzOsc.push(osc)
  }
}

/** Плавно ведёт жужжание: тон и громкость — от нагрузки мухи (0…1), active — летит ли. */
export function setBuzz(load: number, active: boolean) {
  if (!enabled) return
  const c = ensureCtx()
  if (!c || !buzzGain || buzzOsc.length === 0) return
  const t = c.currentTime
  const l = Math.max(0, Math.min(1, load))
  buzzGain.gain.setTargetAtTime(active ? 0.022 + l * 0.045 : 0, t, 0.09)
  const base = 112 + l * 52 // тон растёт с манёвром — «жужжит в тон манёвра»
  buzzOsc[0].frequency.setTargetAtTime(base, t, 0.12)
  buzzOsc[1].frequency.setTargetAtTime(base * 1.006 + 2.5, t, 0.12)
  buzzFilter?.frequency.setTargetAtTime(550 + l * 1500, t, 0.12)
}

// ─── фразы ──────────────────────────────────────────────────────────────────
const buffers = new Map<string, AudioBuffer>()
const lastSaid = new Map<string, number>()
let loadedKind: VoiceKind | null = null

function ensureCtx(): AudioContext | null {
  if (!enabled) return null
  if (!ctx) {
    const AC: typeof AudioContext | undefined =
      window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!AC) return null
    ctx = new AC()
    buildBuzz(ctx)
  }
  if (ctx.state === 'suspended') void ctx.resume()
  return ctx
}

async function loadBuffers(): Promise<void> {
  const c = ensureCtx()
  if (!c) return
  const want = kind
  await Promise.all(
    ['launch', 'capture', 'lost', 'hit', 'miss', 'overload'].flatMap((key) =>
      Array.from({ length: PILOT_VARIANTS }, (_, i) => i).map(async (i) => {
        const id = `${want}/${key}_${i}`
        if (buffers.has(id)) return
        try {
          const res = await fetch(`${import.meta.env.BASE_URL}audio/${want}/${key}_${i}.wav`)
          if (!res.ok) return
          const buf = await c.decodeAudioData(await res.arrayBuffer())
          buffers.set(id, buf)
        } catch {
          /* нет файла или приватный режим — фраза просто не прозвучит */
        }
      }),
    ),
  )
  loadedKind = want
}

/** Включить/выключить озвучку (тумблер в лаборатории). */
export function setVoiceEnabled(on: boolean) {
  enabled = on
  if (on) {
    ensureCtx()
    void loadBuffers()
  } else {
    setBuzz(0, false)
    // выключили звук — гасим и очередь речи, чтобы никто не «ожил» в паузе
    if (speaker) {
      const prev = speaker.src
      speaker = null
      try {
        prev.stop()
      } catch { /* уже остановлена */ }
      prev.onended = null
    }
    shtrumTurn = null
    wendyTurn = null
    royTurn = null
  }
}

/** Выбрать голос: male | female. Догружает буферы нужной папки. */
export function setVoiceKind(v: VoiceKind) {
  if (kind === v) return
  kind = v
  if (enabled) void loadBuffers()
}

// ─── единая шина речи: никто не говорит вдвоём одновременно ────────────────
// Кто-то один держит канал; новая реплика той же мухи перебивает свою же
// прошлую, а между мухами — очередь: пилот не ждёт (перебивает напарницу),
// командир роя, штурман и «Кобра» дожидаются, пока эфир освободится, но не
// вечно (TURN_TTL — реплика устаревает и молча уходит, субтитр-то уже показан).
type SpeakerOwner = 'main' | 'roy' | 'shtrum' | 'wendy'
let speaker: { src: AudioBufferSourceNode; owner: SpeakerOwner } | null = null
let royTurn: { start: () => void; since: number } | null = null
let shtrumTurn: { start: () => void; since: number } | null = null
let wendyTurn: { start: () => void; since: number } | null = null
const TURN_TTL = 5000

function releaseSpeaker(src: AudioBufferSourceNode, owner: SpeakerOwner) {
  if (!speaker || speaker.src !== src) return
  speaker = null
  // освободившийся канал достаётся очерёдному: после пилота — все напарницы
  // (командир роя старше по заведению шины), иначе — следующая по старшинству
  const tryTurn = (who: 'roy' | 'shtrum' | 'wendy'): boolean => {
    const t = who === 'roy' ? royTurn : who === 'shtrum' ? shtrumTurn : wendyTurn
    if (who === 'roy') royTurn = null
    else if (who === 'shtrum') shtrumTurn = null
    else wendyTurn = null
    if (!t) return false
    if (Date.now() - t.since <= TURN_TTL) {
      t.start()
      return true
    }
    return false
  }
  if (owner === 'main') void (tryTurn('roy') || tryTurn('shtrum') || tryTurn('wendy'))
  else if (owner === 'roy') void (tryTurn('shtrum') || tryTurn('wendy'))
  else if (owner === 'shtrum') void (tryTurn('wendy') || tryTurn('roy'))
  else void (tryTurn('roy') || tryTurn('shtrum'))
}

function occupy(src: AudioBufferSourceNode, owner: SpeakerOwner) {
  src.onended = () => releaseSpeaker(src, owner)
  speaker = { src, owner }
  src.start()
}

/** Освободить канал (новая реплика мухи-владельца или её переключение). */
function takeSpeaker(owner: SpeakerOwner) {
  if (speaker && speaker.owner === owner) {
    const prev = speaker.src
    speaker = null
    try {
      prev.stop()
    } catch { /* уже остановлена */ }
    prev.onended = null
  }
  // занята чужим голосом: main перебивает напарницу (она — пассажир);
  // штурман при занятом канале получает false и встаёт в очередь (shtrumTurn)
  if (owner === 'main' && speaker) {
    const prev = speaker.src
    speaker = null
    try {
      prev.stop()
    } catch { /* уже остановлена */ }
    prev.onended = null
  }
  return !speaker
}

/** Прогнать случайный вариант фразы (если пришло её время — cooldown на ключ). */
export function say(key: string) {
  if (!enabled) return
  const c = ensureCtx()
  if (!c) return
  const now = Date.now()
  if (now - (lastSaid.get(key) ?? 0) < (COOLDOWNS[key] ?? 4000)) return
  const variants = Array.from({ length: PILOT_VARIANTS }, (_, i) => i)
    .map((i) => buffers.get(`${kind}/${key}_${i}`))
    .filter((b): b is AudioBuffer => Boolean(b))
  if (variants.length === 0) return
  if (!takeSpeaker('main')) return
  lastSaid.set(key, now)
  const src = c.createBufferSource()
  src.buffer = variants[Math.floor(Math.random() * variants.length)]
  src.connect(c.destination)
  occupy(src, 'main')
}

// ─── юмор-режим: вторая муха-штурман ────────────────────────────────────────
const shtrumLoaded = new Set<'m' | 'f'>()
let shtrumGain: GainNode | null = null

async function loadShtrum(which: 'm' | 'f'): Promise<void> {
  const c = ensureCtx()
  if (!c || shtrumLoaded.has(which)) return
  shtrumLoaded.add(which)
  await Promise.all(
    SHTRUM_KEYS.flatMap((key) =>
      Array.from({ length: SHTRUM_VARIANTS }, (_, i) => i).map(async (i) => {
        const id = `shtrum_${which}/${key}_${i}`
        if (buffers.has(id)) return
        try {
          const res = await fetch(`${import.meta.env.BASE_URL}audio/shtrum_${which}/${key}_${i}.wav`)
          if (!res.ok) return
          buffers.set(id, await c.decodeAudioData(await res.arrayBuffer()))
        } catch {
          /* нет файла — реплика останется только текстом */
        }
      }),
    ),
  )
}

/** Прогрев: скачать клипы нужного пола заранее — иначе первая реплика
 *  прогона останется только текстом (буферы грузятся асинхронно). */
export function warmShtrum(which: 'm' | 'f') {
  if (enabled) void loadShtrum(which)
}

/** Реплика второй мухи: he — он (Штруман), иначе она (Штрумана). variant —
 *  тот же номер варианта, что показан субтитром; клип ещё не загружен —
 *  догружаем и молчим в этом прогоне (текст всё равно виден).
 *  Пилот ещё говорит — встаём в очередь и начинаем, когда она закончит. */
export function sayShtrum(key: string, he: boolean, variant = 0) {
  if (!enabled) return
  const c = ensureCtx()
  if (!c) return
  const which = he ? 'm' : 'f'
  if (!shtrumLoaded.has(which)) {
    void loadShtrum(which)
    return
  }
  const now = Date.now()
  const id = `shtrum_${which}/${key}_${[variant, ...Array.from({ length: SHTRUM_VARIANTS }, (_, i) => i)].find((i) => buffers.has(`shtrum_${which}/${key}_${i}`)) ?? 0}`
  const buf = buffers.get(id)
  if (!buf || now - (lastSaid.get(id) ?? 0) < (SHTRUM_COOLDOWNS[key] ?? 4000)) return
  const free = takeSpeaker('shtrum') // свою прошлую реплику перебиваем; пилот говорит — уступаем
  lastSaid.set(id, now)
  if (!shtrumGain) {
    shtrumGain = c.createGain()
    shtrumGain.gain.value = 0.75 // тише первой мухи, но разборчиво
    shtrumGain.connect(c.destination)
  }
  const src = c.createBufferSource()
  src.buffer = buf
  src.connect(shtrumGain)
  const go = () => {
    if (enabled) occupy(src, 'shtrum')
  }
  if (free) go()
  else shtrumTurn = { start: go, since: now } // ждём своей очереди (субтитр уже на экране)
}

// ─── юмор-режим «Дуэли»: муха-пилот самолёта-цели («Кобра», она) ───────────
const WENDY_COOLDOWNS: Record<string, number> = {
  launch: 2000,
  merge: 8000,
  break: 8000,
  weave: 10000,
  escape: 1500,
  graze: 1500,
  shot: 1500,
  fuse: 1500,
}
let wendyLoaded = false
let wendyGain: GainNode | null = null

async function loadWendy(): Promise<void> {
  const c = ensureCtx()
  if (!c || wendyLoaded) return
  wendyLoaded = true
  await Promise.all(
    WENDY_KEYS.flatMap((key) =>
      Array.from({ length: WENDY_VARIANTS }, (_, i) => i).map(async (i) => {
        const id = `wendy/${key}_${i}`
        if (buffers.has(id)) return
        try {
          const res = await fetch(`${import.meta.env.BASE_URL}audio/wendy/${key}_${i}.wav`)
          if (!res.ok) return
          buffers.set(id, await c.decodeAudioData(await res.arrayBuffer()))
        } catch {
          /* нет файла — реплика останется только текстом */
        }
      }),
    ),
  )
}

/** Прогрев клипов «Кобры» заранее — иначе первая реплика дуэли немая. */
export function warmWendy() {
  if (enabled) void loadWendy()
}

/** Реплика «Кобры»: тот же порядок, что у штурмана — свою прошлую перебивает,
 *  чужой эфир дожидается в очереди (wendyTurn) с TTL. */
export function sayWendy(key: string, variant = 0) {
  if (!enabled) return
  const c = ensureCtx()
  if (!c) return
  if (!wendyLoaded) {
    void loadWendy()
    return
  }
  const now = Date.now()
  const id = `wendy/${key}_${[variant, 0, 1, 2].find((i) => buffers.has(`wendy/${key}_${i}`)) ?? 0}`
  const buf = buffers.get(id)
  if (!buf || now - (lastSaid.get(id) ?? 0) < (WENDY_COOLDOWNS[key] ?? 4000)) return
  const free = takeSpeaker('wendy') // пилот или штурман в эфире — уступаем канал
  lastSaid.set(id, now)
  if (!wendyGain) {
    wendyGain = c.createGain()
    wendyGain.gain.value = 0.7 // напарницы друг друга не заглушают
    wendyGain.connect(c.destination)
  }
  const src = c.createBufferSource()
  src.buffer = buf
  src.connect(wendyGain)
  const go = () => {
    if (enabled) occupy(src, 'wendy')
  }
  if (free) go()
  else wendyTurn = { start: go, since: now }
}

// ─── юмор-режим «Рой»: муха-командир роя (она) ─────────────────────────────
const ROY_COOLDOWNS: Record<string, number> = {
  launch: 2000,
  new_geo: 14000,
  lead: 16000,
  stagnation: 22000,
  validate: 26000,
  champion: 30000,
  hits: 24000,
  tight: 30000,
  calm: 20000,
  stop: 6000,
}
let royLoaded = false
let royGain: GainNode | null = null

async function loadRoy(): Promise<void> {
  const c = ensureCtx()
  if (!c || royLoaded) return
  royLoaded = true
  await Promise.all(
    ROY_KEYS.flatMap((key) =>
      Array.from({ length: WENDY_VARIANTS }, (_, i) => i).map(async (i) => {
        const id = `roy/${key}_${i}`
        if (buffers.has(id)) return
        try {
          const res = await fetch(`${import.meta.env.BASE_URL}audio/roy/${key}_${i}.wav`)
          if (!res.ok) return
          buffers.set(id, await c.decodeAudioData(await res.arrayBuffer()))
        } catch {
          /* нет файла — реплика останется только текстом */
        }
      }),
    ),
  )
}

/** Прогрев клипов командира заранее — иначе первая реплика роя немая. */
export function warmRoy() {
  if (enabled) void loadRoy()
}

/** Реплика командира роя: тот же порядок, что у «Кобры» — свою прошлую
 *  перебивает, чужой эфир дожидается в очереди (royTurn) с TTL. */
export function sayRoy(key: string, variant = 0) {
  if (!enabled) return
  const c = ensureCtx()
  if (!c) return
  if (!royLoaded) {
    void loadRoy()
    return
  }
  const now = Date.now()
  const id = `roy/${key}_${[variant, 0, 1, 2].find((i) => buffers.has(`roy/${key}_${i}`)) ?? 0}`
  const buf = buffers.get(id)
  if (!buf || now - (lastSaid.get(id) ?? 0) < (ROY_COOLDOWNS[key] ?? 15000)) return
  const free = takeSpeaker('roy') // пилот в эфире — уступаем канал
  lastSaid.set(id, now)
  if (!royGain) {
    royGain = c.createGain()
    royGain.gain.value = 0.7 // как у напарниц: разборчиво, не заглушая пилота
    royGain.connect(c.destination)
  }
  const src = c.createBufferSource()
  src.buffer = buf
  src.connect(royGain)
  const go = () => {
    if (enabled) occupy(src, 'roy')
  }
  if (free) go()
  else royTurn = { start: go, since: now }
}

// отладка: внутреннее состояние модуля (вызывается из консоли/тестов)
export function voiceDebug() {
  return { enabled, kind, loadedKind, ctxState: ctx?.state ?? null, buffers: [...buffers.keys()] }
}
