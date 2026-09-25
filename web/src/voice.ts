// Озвучка мухи: жужжание «в тон манёвра» (WebAudio, без файлов) и голосовые
// фразы (Silero TTS, заранее записаны tools/make_voice.py в /audio/{голос}/).
// Юмор-режим: вторая муха-штурман противоположного пола (/audio/shtrum_m|shtrum_f/),
// говорит тише первой и НЕ перекрывает её: канал речи один — штурман ждёт,
// пока пилот договорит (субтитр при этом показывается сразу).

export type VoiceKind = 'male' | 'female'

const PHRASE_VARIANTS = 3
const SHTRUM_KEYS = ['send', 'look_left', 'look_up', 'overload', 'lost', 'hit', 'miss', 'graze']
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
      [0, 1, 2].map(async (i) => {
        const id = `${want}/${key}_${i}`
        if (buffers.has(id)) return
        try {
          const res = await fetch(`/audio/${want}/${key}_${i}.wav`)
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
  }
}

/** Выбрать голос: male | female. Догружает буферы нужной папки. */
export function setVoiceKind(v: VoiceKind) {
  if (kind === v) return
  kind = v
  if (enabled) void loadBuffers()
}

// ─── единая шина речи: две мухи не говорят одновременно ────────────────────
// Кто-то один держит канал; новая реплика той же мухи перебивает свою же
// прошлую, а между мухами — очередь: пилот не ждёт (перебивает напарницу),
// штурман дожидается, пока пилот договорит, но не вечно (TURN_TTL — реплика
// устаревает и молча уходит, субтитр-то уже показан).
type SpeakerOwner = 'main' | 'shtrum'
let speaker: { src: AudioBufferSourceNode; owner: SpeakerOwner } | null = null
let shtrumTurn: { start: () => void; since: number } | null = null
const TURN_TTL = 5000

function releaseSpeaker(src: AudioBufferSourceNode, owner: SpeakerOwner) {
  if (!speaker || speaker.src !== src) return
  speaker = null
  if (owner !== 'main' || !shtrumTurn) return
  const t = shtrumTurn
  shtrumTurn = null
  if (Date.now() - t.since <= TURN_TTL) t.start()
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
  const variants = [0, 1, 2]
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
      [0, 1, 2].map(async (i) => {
        const id = `shtrum_${which}/${key}_${i}`
        if (buffers.has(id)) return
        try {
          const res = await fetch(`/audio/shtrum_${which}/${key}_${i}.wav`)
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
  const id = `shtrum_${which}/${key}_${[variant, 0, 1, 2].find((i) => buffers.has(`shtrum_${which}/${key}_${i}`)) ?? 0}`
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

// отладка: внутреннее состояние модуля (вызывается из консоли/тестов)
export function voiceDebug() {
  return { enabled, kind, loadedKind, ctxState: ctx?.state ?? null, buffers: [...buffers.keys()] }
}
