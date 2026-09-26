import { useEffect, useRef, useState } from 'react'
import type { Playback } from './EngagementView'
import { LabView, type LabTab, type AblationData, type CaptureZoneData, type CoevData, type DistillData, type FaultsData, type LadderData, type MapData, type MonteCarloData, type ScalingData, type TransferData } from './lab/LabView'
import { HelpView } from './HelpView'
import { Tour } from './shell/Tour'
import { applyServerW, isTrained, markTrained, setBrainKind } from './brain'
import { computeRunMetrics, navMetrics } from './metrics'
import { localRun } from './localSim'
import { canonicalScenarios, flyRollout, initPopulation, runGeneration, sampleGenerationScenario, sanitizeFly, scenarioLabel, synthFrameFromTel, type GenerationResult } from './swarm'
import { trainLocal } from './trainLocal'
import { setVoiceEnabled, setVoiceKind as applyVoiceKind, say, sayShtrum, sayWendy, sayRoy, warmShtrum, warmWendy, warmRoy, type VoiceKind } from './voice'
import { planShtrum, type ShtrumLine, type ShtrumPlan } from './shtrum'
import { planWendy, type WendyLine, type WendyPlan } from './wendy'
import { royLine, type RoyKey } from './roy'
import { exportCsv } from './lab/charts'
import { resetLayout } from './ui'
import { perfBudget } from './perf'
import { buzz, HAPTIC } from './haptics'
import { AppShell } from './shell/AppShell'
import type { Workspace } from './shell/TopBar'
import { FlightWorkspace } from './shell/FlightWorkspace'
import { BrainWorkspace } from './shell/BrainWorkspace'
import { SwarmWorkspace } from './shell/SwarmWorkspace'
import { DuelWorkspace } from './shell/DuelWorkspace'
import type { DuelReplay } from './shell/DuelWorkspace'
import { ASPECT_LABEL, EVENT_RU, LAW_RU, MODE_LABEL, fmt } from './shell/labels'
import { DEFAULT_SCENARIO, FEATURE_SCHEMA_VERSION, MODEL_VERSION, emptyLab, type BrainKind, type CamMode, type DuelMatrix, type Frame, type FlyGenome, type GenPoint, type LabData, type RunMetrics, type Scenario } from './types'

/** Строка сравнения: мозг или закон наведения с честными метриками прогона. */
export type CmpRow = {
  kind: string
  label: string
  n_cells: number
  trained: boolean | null
  miss_m: number
  hit: boolean
  n_peak: number
  lock_frac: number
  t_guide?: number | null
  ref_dev_m?: number | null
  ref_nrms?: number | null
  t_end?: number | null
  t_ref?: number | null
  n_int?: number
}

const dist3 = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])

/** Точка кривой прогона: время, h_cv, отклонение от эталонного ПН. */
const framePoint = (fr: Frame) => ({ t: fr.t, zem: navMetrics(fr).zem, dev: fr.ghost ? dist3(fr.missile, fr.ghost) : null, rng: fr.range_m })
/** Точка N_экв-диагностики кадра — графики адаптивности в лаборатории. */
const neffPoint = (fr: Frame) => ({
  t: fr.t,
  tgo: fr.tgo ?? null,
  rho: fr.rho ?? 0,
  vc: fr.v_c,
  nEff: fr.n_eff ?? null,
  valid: Boolean(fr.n_eff_valid),
  sat: Boolean(fr.sat),
  speedMode: fr.speed_mode ?? 'constant',
})

/** Траектории прогона в план-виде [дальность, бок], прорежено. */
const sampleTraj = (frames: Frame[]) => {
  const step = Math.max(1, Math.floor(frames.length / 160))
  const missile: number[][] = []
  const ghost: number[][] = []
  const target: number[][] = []
  frames.forEach((fr, i) => {
    if (i % step !== 0 && i !== frames.length - 1) return
    missile.push([Math.round(fr.missile[0]), Math.round(fr.missile[1]), Math.round(fr.missile[2])])
    target.push([Math.round(fr.target[0]), Math.round(fr.target[1]), Math.round(fr.target[2])])
    if (fr.ghost) ghost.push([Math.round(fr.ghost[0]), Math.round(fr.ghost[1]), Math.round(fr.ghost[2])])
  })
  return { missile, ghost, target }
}

function rusFrame(fr: Frame): Frame {
  const src = fr.layers || {}
  const layers = {
    Зрение: Number(src.Зрение ?? src.VISION ?? 0),
    Поток: Number(src.Поток ?? src.FLOW ?? 0),
    Приближение: Number(src.Приближение ?? src.LOOM ?? 0),
    Решение: Number(src.Решение ?? src.DECISION ?? 0),
    Мотор: Number(src.Мотор ?? src.MOTOR ?? 0),
  }
  return {
    ...fr,
    layers,
    event: fr.event ? EVENT_RU[fr.event] || fr.event : null,
  }
}

const nCellsLabel = (b: BrainKind) => (b === 'full' ? 4439 : b === 'connectome' ? 108781 : 279)

/** Демо-сценарий первого визита (App.tsx::runDemo) — наглядный перехват со змейкой,
 * умеренная дальность. Считается локально (localRun), без единого сетевого запроса —
 * работает одинаково на стенде и на GitHub Pages. */
const DEMO_SCENARIO: Scenario = { ...DEFAULT_SCENARIO, range_m: 5000, maneuver: 'weave', n_target: 6 }

/** Эвристика «уже реальный пользователь»: панели/секции сворачивались хоть раз —
 * значит стенд уже открывали до появления демо (ui.tsx::resetLayout — то же именование). */
function hasPriorUsage(): boolean {
  try {
    for (let i = 0; i < localStorage.length; i += 1) {
      const k = localStorage.key(i)
      if (k && (k.startsWith('muholet-panel-') || k.startsWith('muholet-acc-'))) return true
    }
  } catch {
    /* приватный режим — считаем «первый визит» */
  }
  return false
}

export function App() {
  // сценарий гидратируется из localStorage синхронно (в инициализаторе):
  // любой effect-restore проигрывает автосохранению [sc] на том же коммите
  const [sc, setSc] = useState<Scenario>(() => {
    try {
      const saved = localStorage.getItem('muholet-scenario')
      if (saved) return { ...DEFAULT_SCENARIO, ...(JSON.parse(saved) as Partial<Scenario>) }
    } catch {
      /* повреждённый localStorage — стартуем с дефолта */
    }
    return DEFAULT_SCENARIO
  })
  const [frame, setFrame] = useState<Frame | null>(null)
  const [log, setLog] = useState<string[]>(['Стенд готов. Нажмите «Пуск».'])
  const [busy, setBusy] = useState(false)
  const [done, setDone] = useState<string | null>(null)
  const [trainNote, setTrainNote] = useState('контур не обучен')
  const [trained, setTrained] = useState(isTrained())
  const [serverOnline, setServerOnline] = useState<boolean | null>(null)
  const [playback, setPlayback] = useState<Playback | null>(null)
  const [swarmRunning, setSwarmRunning] = useState(false)
  // активное рабочее пространство: одновременно виден только один режим
  const [ws, setWs] = useState<Workspace>(() => {
    try {
      const saved = localStorage.getItem('muholet-workspace')
      if (saved === 'flight' || saved === 'brain' || saved === 'swarm' || saved === 'duel' || saved === 'lab') return saved
    } catch { /* приватный режим */ }
    return 'flight'
  })
  // подпись на 3D-сцене: текущий эпизод обучения / поколение роя / демо
  const [sceneBadge, setSceneBadge] = useState<string | null>(null)
  // кривая обучения роя: наименьшее сближение по поколениям (прямо на 3D-сцене)
  const [swarmCurve, setSwarmCurve] = useState<number[] | null>(null)
  // индексы валидационных поколений в кривой эволюции (где считалось эталонное трио)
  const [swarmValid, setSwarmValid] = useState<number[]>([])
  const [swarmInfo, setSwarmInfo] = useState<{ gen: number; bestMiss: number; avgFit: number; bio: number; pn: number; fits: number[]; geo: string; champion?: number } | null>(null)
  const [swarmCfg, setSwarmCfg] = useState({ size: 24, eliteK: 4, mutation: 0.25 })
  // дуэль: вердикт последнего прогона (из server-полей /api/run) и матрица «Ринг»
  const [duelVerdict, setDuelVerdict] = useState<{ result: 'missile' | 'evader' | null; tSurvived: number | null; fuse: boolean } | null>(null)
  const [duelMatrix, setDuelMatrix] = useState<DuelMatrix | null>(null)
  const [duelBusy, setDuelBusy] = useState(false)
  const [duelRepeats, setDuelRepeats] = useState(1)
  // обучение: явные параметры (режим задаёт пресет, значения можно править в «Параметрах»)
  const [trainCfg, setTrainCfg] = useState<{ mode: 'scratch' | 'finetune' | 'result'; lr: number; episodes: number }>({
    mode: 'scratch',
    lr: 0.04,
    episodes: 24,
  })
  const [tuneChannels, setTuneChannels] = useState(64)
  const [tunePool, setTunePool] = useState(32)
  const [robBusy, setRobBusy] = useState(false)
  const [robData, setRobData] = useState<{
    levels: number[]
    series: { kind: string; label: string; miss: number[]; nrms: (number | null)[] }[]
  } | null>(null)
  const [mcBusy, setMcBusy] = useState(false)
  const [mcData, setMcData] = useState<MonteCarloData | null>(null)
  const [mcRuns, setMcRuns] = useState(50)
  const [czBusy, setCzBusy] = useState(false)
  const [czData, setCzData] = useState<CaptureZoneData | null>(null)
  const [czGmax, setCzGmax] = useState(20)
  const [czRuns, setCzRuns] = useState(1)
  // сохранённые на сервере эксперименты (data/experiments)
  const [expList, setExpList] = useState<{ name: string; saved_at: string; size: number }[]>([])
  const [day, setDay] = useState(false)
  const [geometryOn, setGeometryOn] = useState(true)
  const [camMode, setCamMode] = useState<CamMode>('auto')
  const [training, setTraining] = useState(false)
  const [labTab, setLabTab] = useState<LabTab>('run')
  const [helpOpen, setHelpOpen] = useState(false)
  const [tourOpen, setTourOpen] = useState(false)
  // пасхалка: муха за штурвалом — разрез корпуса, рычаги = реальные команды DN
  const [egg, setEgg] = useState(false)
  // кинорежим: сцена во весь экран без шапки/панелей (клавиша K, выход — Esc)
  const [cinema, setCinema] = useState(false)
  // баннер после демо-перехвата первого визита (App.tsx::runDemo)
  const [demoBanner, setDemoBanner] = useState(false)
  // озвучка: жужжание в тон манёвра + голосовые фразы (Silero, web/public/audio).
  // Настройки читаются СИНХРОННО в инициализаторах useState: в dev-StrictMode
  // эффект записи на втором проходе маунта натирает хранилище дефолтами, пока
  // эффект чтения ещё не отработал (тот же грабли-прецедент, что с гидратацией сценария)
  const savedVoice = (() => {
    try {
      return JSON.parse(localStorage.getItem('muholet-sound') || '') as {
        on?: boolean
        voice?: VoiceKind
        humor?: boolean
      }
    } catch {
      return {} as { on?: boolean; voice?: VoiceKind; humor?: boolean }
    }
  })()
  const [soundOn, setSoundOn] = useState(savedVoice.on === true)
  const [voiceKind, setVoiceKind] = useState<VoiceKind>(
    savedVoice.voice === 'male' || savedVoice.voice === 'female' ? savedVoice.voice : 'female',
  )
  // юмор-режим: вторая муха-штурман противоположного основной мухе пола
  // комментирует полёт субтитрами и голосом — физика и мозг не трогаются
  const [humorOn, setHumorOn] = useState(savedVoice.humor === true)
  const [shtrumCaption, setShtrumCaption] = useState<{ text: string; he: boolean } | null>(null)
  const shtrumHideRef = useRef<number | null>(null)
  // «Кобра» — муха-пилот самолёта-цели; говорит только в дуэли (её субтитр)
  const [wendyCaption, setWendyCaption] = useState<{ text: string } | null>(null)
  const wendyHideRef = useRef<number | null>(null)
  // командир роя — комментирует ход эволюции (её субтитр)
  const [royCaption, setRoyCaption] = useState<{ text: string } | null>(null)
  const royHideRef = useRef<number | null>(null)
  // прогресс обучения для лаборатории
  const [trainProgress, setTrainProgress] = useState<{ ep: number; total: number } | null>(null)
  // настройки лаборатории: окно сглаживания и показ факта
  const [labCfg, setLabCfg] = useState({ smooth: 7, showFact: true })
  const labRef = useRef<LabData>(emptyLab())
  const labSessionRef = useRef(0)
  const labTick = () => setLabVer((v) => v + 1)
  const [labVer, setLabVer] = useState(0)

  /** Лаборатория открывается сразу на нужном экране. */
  const openLab = (tab: LabTab) => {
    setLabTab(tab)
    setWs('lab')
  }

  // озвучка: модуль держим в курсе включения и голоса; настройка переживает перезагрузку
  useEffect(() => {
    setVoiceEnabled(soundOn)
    applyVoiceKind(voiceKind)
    // юмор при звуке греем оба пола клипов: напарник меняется вместе с голосом,
    // а скачивание буферов асинхронное — иначе первая реплика не прозвучит
    if (soundOn && humorOn) {
      warmShtrum('m')
      warmShtrum('f')
      warmWendy()
      warmRoy()
    }
    try {
      localStorage.setItem('muholet-sound', JSON.stringify({ on: soundOn, voice: voiceKind, humor: humorOn }))
    } catch { /* приватный режим */ }
  }, [soundOn, voiceKind, humorOn])
  // фразы «цель захвачена / потеряна» — по переключению захвата в кадре;
  // в рое и его подрежимах молчим: циклы короткие, фразы не успевают
  const lockRef = useRef(false)
  useEffect(() => {
    const lock = frame?.lock ?? false
    const voiced = !swarmRunning && playback?.race !== true
    if (voiced) {
      if (lock && !lockRef.current) say('capture')
      if (!lock && lockRef.current) say('lost')
    }
    lockRef.current = lock
  }, [frame?.lock, swarmRunning, playback])

  // ── ночная смена: абляция зон, карта преимуществ, наложение траекторий ──
  const [ablation, setAblation] = useState<AblationData | null>(null)
  const [ablationBusy, setAblationBusy] = useState(false)
  const [mapData, setMapData] = useState<MapData | null>(null)
  const [mapBusy, setMapBusy] = useState(false)
  const [overlayBusy, setOverlayBusy] = useState(false)
  const [distill, setDistill] = useState<DistillData | null>(null)
  const [distillBusy, setDistillBusy] = useState(false)
  const [distillTeacher, setDistillTeacher] = useState<'connectome' | 'full' | 'stub'>('connectome')
  const [distillSaved, setDistillSaved] = useState(false)
  // наука уровня 2: матрица переносимости и scaling-кривая
  const [transfer, setTransfer] = useState<TransferData | null>(null)
  const [transferBusy, setTransferBusy] = useState(false)
  const [transferKind, setTransferKind] = useState<BrainKind>('connectome')
  const [scaling, setScaling] = useState<ScalingData | null>(null)
  const [scalingBusy, setScalingBusy] = useState(false)
  const [scalingKind, setScalingKind] = useState<'full' | 'connectome'>('connectome')
  const [coev, setCoev] = useState<CoevData | null>(null)
  const [coevBusy, setCoevBusy] = useState(false)
  const [coevTraining, setCoevTraining] = useState(false)
  const [coevLadder, setCoevLadder] = useState<LadderData | null>(null)
  const [faults, setFaults] = useState<FaultsData | null>(null)
  const [faultsBusy, setFaultsBusy] = useState(false)
  const [mapRepeats, setMapRepeats] = useState(2)
  const [mapRetina, setMapRetina] = useState(0)
  const [mapColor, setMapColor] = useState<'miss' | 'energy'>('miss')

  const runFaults = async () => {
    setFaultsBusy(true)
    openLab('compare')
    try {
      const res = await fetch('/api/brain/faults')
      if (!res.ok) throw new Error(`сервер ${res.status}`)
      const d = (await res.json()) as FaultsData
      setFaults(d)
      const first = d.rows[0]
      const last = d.rows[d.rows.length - 1]
      setLog((rows) => [`Отказы сетчатки: ${first.miss} м (0%) → ${last.miss} м (${Math.round(last.fraction * 100)}%)`, ...rows].slice(0, 14))
    } catch {
      setLog((rows) => ['Карта отказов — только на стенде и для коннектома.', ...rows].slice(0, 14))
    } finally {
      setFaultsBusy(false)
    }
  }

  const coevTrain = async () => {
    setCoevTraining(true)
    try {
      const res = await fetch('/api/coevolve/train_fly', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ generations: 3, rounds: 3 }),
      })
      if (!res.ok) throw new Error(`сервер ${res.status}`)
      const d = (await res.json()) as LadderData & { canonical: { miss: number }; history?: CoevData['history'] }
      setCoevLadder({ fly_miss_before: d.fly_miss_before, fly_miss_after: d.fly_miss_after, generations: d.generations, restored: d.restored })
      if (d.history) setCoev((c) => (c ? { ...c, history: d.history! } : c))
      setLog((rows) => [
        `Петля коэволюции: против чемпиона ${d.fly_miss_before} м → ${d.fly_miss_after} м (трио ${d.canonical.miss} м)`,
        ...rows,
      ].slice(0, 14))
    } catch {
      setLog((rows) => ['Дообучение против чемпиона — только на стенде.', ...rows].slice(0, 14))
    } finally {
      setCoevTraining(false)
    }
  }

  const runDistill = async () => {
    setDistillBusy(true)
    openLab('compare')
    try {
      const res = await fetch('/api/brain/distill', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ teacher: distillTeacher }),
      })
      if (!res.ok) throw new Error(`сервер ${res.status}`)
      const d = (await res.json()) as DistillData
      setDistill(d)
      setDistillSaved(false)
      setLog((rows) => [
        `Дистилляция (${d.teacher_kind ?? distillTeacher}): учитель ${d.teacher.miss} м → дистиллят ${d.distilled.miss} м (схема с нуля ${d.scratch.miss} м)`,
        ...rows,
      ].slice(0, 14))
    } catch {
      setLog((rows) => ['Дистилляция — только на стенде.', ...rows].slice(0, 14))
    } finally {
      setDistillBusy(false)
    }
  }

  const distillSave = async () => {
    try {
      const res = await fetch('/api/brain/distill/save', { method: 'POST' })
      if (!res.ok) throw new Error(`сервер ${res.status}`)
      setDistillSaved(true)
      setLog((rows) => ['Дистиллят сохранён в data/weights_distilled.npz — переживает перезапуск.', ...rows].slice(0, 14))
    } catch {
      setLog((rows) => ['Сохранение дистиллята — только на стенде.', ...rows].slice(0, 14))
    }
  }

  const distillLoad = async () => {
    try {
      const res = await fetch('/api/brain/distill/load', { method: 'POST' })
      if (!res.ok) throw new Error(`сервер ${res.status}`)
      const d = (await res.json()) as { distilled: { miss: number; hit_rate: number; ref_dev: number }; teacher: string }
      setDistill((prev) =>
        prev ?? { epochs: 0, teacher_kind: d.teacher, teacher: { miss: 0, hit_rate: 0 }, distilled: d.distilled, scratch: { miss: 0, hit_rate: 0, ref_dev: 0 } },
      )
      setLog((rows) => [`Дистиллят загружен из файла: трио ${d.distilled.miss} м. Можно «Применить к схеме».`, ...rows].slice(0, 14))
    } catch {
      setLog((rows) => ['Сохранённого дистиллята нет — сначала дистилляция + «Сохранить».', ...rows].slice(0, 14))
    }
  }

  const runTransfer = async () => {
    setTransferBusy(true)
    openLab('transfer')
    try {
      const res = await fetch('/api/transfer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kind: transferKind }),
      })
      if (!res.ok) throw new Error(`сервер ${res.status}`)
      const d = (await res.json()) as TransferData
      setTransfer(d)
      const diag = d.rows.map((r) => r.tests[r.train] ?? NaN)
      const off = d.rows.flatMap((r) => d.test_maneuvers.filter((t) => t !== r.train).map((t) => r.tests[t] ?? NaN))
      const med = (xs: number[]) => {
        const s = xs.filter(Number.isFinite).sort((a, b) => a - b)
        return s.length ? s[Math.floor(s.length / 2)] : NaN
      }
      setLog((rows) => [
        `Матрица переносимости (${d.kind}): диагональ ${med(diag)?.toFixed(0) ?? '—'} м, перенос ${med(off)?.toFixed(0) ?? '—'} м`,
        ...rows,
      ].slice(0, 14))
    } catch {
      setLog((rows) => ['Матрица переносимости — только на стенде.', ...rows].slice(0, 14))
    } finally {
      setTransferBusy(false)
    }
  }

  const runScaling = async () => {
    setScalingBusy(true)
    openLab('scaling')
    try {
      const sizes = scalingKind === 'connectome' ? [32, 64, 128] : [8, 16, 32, 64]
      const res = await fetch('/api/scaling', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kind: scalingKind, sizes }),
      })
      if (!res.ok) throw new Error(`сервер ${res.status}`)
      const d = (await res.json()) as ScalingData
      setScaling(d)
      const best = d.rows.reduce((a, b) => (b.miss_after < a.miss_after ? b : a), d.rows[0])
      setLog((rows) => [
        `Кривая масштабируемости (${d.kind}): наименьшее сближение ${best.miss_after} м у размера ${best.size} (${best.params.toLocaleString('ru')} параметров)`,
        ...rows,
      ].slice(0, 14))
    } catch {
      setLog((rows) => ['Кривая масштабируемости — только на стенде.', ...rows].slice(0, 14))
    } finally {
      setScalingBusy(false)
    }
  }

  const applyDistill = async () => {
    try {
      const res = await fetch('/api/brain/distill/apply', { method: 'POST' })
      if (!res.ok) throw new Error(`сервер ${res.status}`)
      setLog((rows) => ['Дистиллят применён: схема теперь летает с «знаниями» коннектома.', ...rows].slice(0, 14))
    } catch {
      setLog((rows) => ['Применение дистиллята — только на стенде.', ...rows].slice(0, 14))
    }
  }

  const coevStart = async () => {
    setCoevBusy(true)
    try {
      await fetch('/api/coevolve/start', { method: 'POST' })
      setCoev(null)
      await coevStepInner()
    } finally {
      setCoevBusy(false)
    }
  }

  const coevStepInner = async () => {
    try {
      const res = await fetch('/api/coevolve/step', { method: 'POST' })
      if (!res.ok) throw new Error(`сервер ${res.status}`)
      const d = (await res.json()) as CoevData
      setCoev({ gen: d.gen, best: d.best, history: d.history })
    } catch {
      setLog((rows) => ['Коэволюция — только на стенде. Сначала «Старт».', ...rows].slice(0, 14))
    }
  }

  const coevStep = async () => {
    setCoevBusy(true)
    await coevStepInner()
    setCoevBusy(false)
  }

  const coevReset = async () => {
    setCoevBusy(true)
    try {
      await fetch('/api/coevolve/reset', { method: 'POST' })
      setCoev(null)
    } finally {
      setCoevBusy(false)
    }
  }

  const runAblation = async () => {
    setAblationBusy(true)
    openLab('compare')
    try {
      const res = await fetch('/api/brain/ablation')
      if (!res.ok) throw new Error(`сервер ${res.status}`)
      const d = (await res.json()) as AblationData
      setAblation(d)
      setLog((rows) => [
        `Абляция зон: база ${fmt(d.base.miss, 0)} м · ` +
          d.rows.map((r) => `${r.zone} ${fmt(r.miss, 0)} м`).join(' · '),
        ...rows,
      ].slice(0, 14))
    } catch {
      setLog((rows) => ['Абляция зон — только на стенде и только для коннектома.', ...rows].slice(0, 14))
    } finally {
      setAblationBusy(false)
    }
  }

  const runMap = async () => {
    setMapBusy(true)
    openLab('map')
    try {
      const res = await fetch('/api/map', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ brain: sc.brain, repeats: mapRepeats, retina_death_p: mapRetina }),
      })
      if (!res.ok) throw new Error(`сервер ${res.status}`)
      const d = (await res.json()) as MapData
      setMapData(d)
      const wins = d.rows.filter((r) => r.advantage > 5).length
      setLog((rows) => [
        `Карта преимуществ: муха точнее ПН в ${wins} из ${d.rows.length} ячеек (порог 5 м)`,
        ...rows,
      ].slice(0, 14))
    } catch {
      setLog((rows) => ['Карта преимуществ — только на стенде.', ...rows].slice(0, 14))
    } finally {
      setMapBusy(false)
    }
  }

  const overlayTrajectories = async () => {
    setOverlayBusy(true)
    try {
      const res = await fetch('/api/brain/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario: sc,
          kinds: ['stub', 'full', 'connectome'],
          laws: ['tpn', 'apn', 'pure', 'clos', 'pn_gsn'],
          with_traj: true,
        }),
      })
      if (!res.ok) throw new Error(`сервер ${res.status}`)
      const data = (await res.json()) as {
        results: { kind: string; label: string; traj_m?: number[][]; miss_m: number; hit: boolean }[]
      }
      const results = data.results
        .filter((r) => Array.isArray(r.traj_m) && r.traj_m.length > 1)
        .map((r) => ({ kind: r.kind, traj_m: r.traj_m!, traj_t: [] as number[][], fitness: -r.miss_m, hit: r.hit, miss_m: r.miss_m, label: r.label }))
      if (results.length < 2) throw new Error('нет траекторий')
      const bestIdx = results.reduce((best, r, i) => (r.miss_m < results[best].miss_m ? i : best), 0)
      const maxPts = Math.max(...results.map((r) => r.traj_m.length), 2)
      setPlayback({
        results,
        bestIdx,
        startedAt: performance.now(),
        durationMs: Math.max(6000, Math.min(16000, maxPts * 40)),
      })
      const SHORT: Record<string, string> = { pn: 'ПН', apn: 'ПН+а', pure: 'погоня', clos: '3 точки', stub: 'схема', full: 'полный', connectome: 'коннектом' }
      setSceneBadge(`наложение · ${results.map((r) => SHORT[r.kind] ?? r.kind).join(' / ')}`)
      setLog((rows) => [`Наложение траекторий: ${results.length} на 3D-сцене, точнее — ${results[bestIdx].label}.`, ...rows].slice(0, 14))
    } catch (e) {
      setLog((rows) => [`Наложение не получилось: ${e instanceof Error ? e.message : 'ошибка'}`, ...rows].slice(0, 14))
    } finally {
      setOverlayBusy(false)
    }
  }

  // история лаборатории переживает перезагрузку страницы
  // (сценарий гидратируется выше, в инициализаторе useState)
  useEffect(() => {
    try {
      const lab = localStorage.getItem('muholet-lab')
      if (lab) {
        const d = JSON.parse(lab) as Record<string, unknown>
        labRef.current = {
          // миграция старых форматов: train без метрик, zem числами
          train: (Array.isArray(d.train) ? (d.train as Record<string, unknown>[]) : []).map((p, i) => ({
            ep: Number(p.ep) || i + 1,
            miss: Number(p.miss) || 0,
            hit: Boolean(p.hit),
            tGuide: p.tGuide === null || p.tGuide === undefined ? null : Number(p.tGuide),
            refDev: Number(p.refDev) || 0,
            w: Array.isArray(p.w) ? (p.w as number[]) : [],
            session: p.session === undefined ? 0 : Number(p.session),
          })),
          gen: (Array.isArray(d.gen) ? (d.gen as Record<string, unknown>[]) : []).map((p, i) => ({
            gen: Number(p.gen) || i + 1,
            best: Number(p.best) || 0,
            avg: Number(p.avg) || 0,
            worst: p.worst === undefined ? undefined : Number(p.worst),
            hitRate: p.hitRate === undefined ? undefined : Number(p.hitRate),
            champ: p.champ === undefined ? undefined : Number(p.champ),
          })),
          zem: (Array.isArray(d.zem) ? d.zem : []).map((v) =>
            typeof v === 'number'
              ? { t: 0, zem: v, dev: null, rng: 0 }
              : { t: Number((v as Record<string, unknown>).t) || 0, zem: Number((v as Record<string, unknown>).zem) || 0, dev: (v as Record<string, unknown>).dev === null ? null : Number((v as Record<string, unknown>).dev), rng: Number((v as Record<string, unknown>).rng) || 0 },
          ),
          runs: Array.isArray(d.runs) ? (d.runs as LabData['runs']) : [],
          neff: Array.isArray(d.neff) ? (d.neff as LabData['neff']) : [],
          traj: (d.traj as LabData['traj']) ?? null,
          summary: (d.summary as LabData['summary']) ?? null,
        }
        setLabVer((v) => v + 1)
      }
    } catch {
      /* повреждённый localStorage игнорируем */
    }
  }, [])
  useEffect(() => {
    try {
      localStorage.setItem('muholet-scenario', JSON.stringify(sc))
    } catch {
      /* приватный режим — пропускаем */
    }
  }, [sc])
  useEffect(() => {
    try {
      localStorage.setItem('muholet-workspace', ws)
    } catch {
      /* приватный режим — пропускаем */
    }
  }, [ws])
  const saveLab = () => {
    try {
      localStorage.setItem('muholet-lab', JSON.stringify(labRef.current))
    } catch {
      /* переполнение квоты — история в памяти всё равно доступна */
    }
  }
  /** Очистить всю историю лаборатории: прогоны, обучение, рой, N_экв и кривые. */
  const clearLab = () => {
    const empty = !labRef.current.runs.length && !labRef.current.train.length && !labRef.current.gen.length && !labRef.current.zem.length
    if (empty) return
    if (!window.confirm('Очистить историю лаборатории: все прогоны, обучение, рой и графики? Действие необратимо.')) return
    labRef.current = emptyLab()
    saveLab()
    labTick()
    setLog((rows) => ['История лаборатории очищена.', ...rows].slice(0, 14))
  }
  const clearLog = () => setLog([])
  const stop = useRef(false)
  const lastFramesRef = useRef<Frame[]>([])
  // счётчик пуска: инстант-реплей планируется с задержкой (500мс) — если за это время
  // начался новый пуск, устаревший реплей не должен перекрыть его свежий playback
  const runIdRef = useRef(0)
  const swarmStop = useRef(false)
  const populationRef = useRef<FlyGenome[] | null>(null)

  useEffect(() => {
    const ping = async () => {
      try {
        const r = await fetch('/api/health')
        setServerOnline(r.ok)
      } catch {
        setServerOnline(false)
      }
    }
    void ping()
    const id = setInterval(ping, 4000)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    document.body.classList.toggle('theme-day', day)
    // панель мобильного браузера — в цвет страницы (--surface-page темы)
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', day ? '#e8eeea' : '#0a0f14')
  }, [day])

  // Горячие клавиши: Пробел — пуск, X — пасхалка, K — кинорежим, Esc — выйти из кино.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return
      if (e.code === 'Space' && !busy && !swarmRunning) {
        e.preventDefault()
        void run()
      }
      if (e.code === 'KeyX') {
        e.preventDefault()
        setEgg((v) => !v)
      }
      if (e.code === 'KeyK') {
        e.preventDefault()
        buzz(HAPTIC.cinemaToggle)
        setCinema((v) => !v)
      }
      // выход из кино по Esc не перехватывает событие — инспектор и другие Esc-обработчики
      // (FlightWorkspace) продолжают работать своим порядком независимо от этого
      if (e.code === 'Escape') setCinema((v) => (v ? false : v))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [busy, swarmRunning, sc])

  // Единая всплывающая подсказка: у панелей overflow скрыт, поэтому тултип один на body.
  useEffect(() => {
    const tip = document.createElement('div')
    tip.className = 'tooltip'
    document.body.appendChild(tip)
    // тач: mouseover эмулируется после тапа и подсказка залипает поверх кнопки, поэтому
    // пальцем она открывается только по значку «?» и сама гаснет — по тапу мимо, прокрутке, таймеру
    let touch = false
    let hideTimer = 0
    const hide = () => tip.classList.remove('on')
    const onDown = (e: PointerEvent) => {
      touch = e.pointerType !== 'mouse'
      if (touch) hide()
    }
    const onOver = (e: MouseEvent) => {
      const el = (e.target as HTMLElement).closest?.('[data-tip]') as HTMLElement | null
      if (!el) return
      if (touch && !el.classList.contains('hint')) return
      window.clearTimeout(hideTimer)
      if (touch) hideTimer = window.setTimeout(hide, 6000)
      tip.textContent = el.getAttribute('data-tip') || ''
      tip.classList.add('on')
      const r = el.getBoundingClientRect()
      const tw = tip.offsetWidth
      const th = tip.offsetHeight
      const x = Math.max(8, Math.min(window.innerWidth - tw - 8, r.left + r.width / 2 - tw / 2))
      let y = r.top - th - 8
      if (y < 8) y = r.bottom + 8
      tip.style.left = `${x}px`
      tip.style.top = `${y}px`
    }
    const onOut = (e: MouseEvent) => {
      // после тапа браузер шлёт синтетический mouseout — пальцем гасим только тапом мимо, прокруткой и таймером
      if (touch) return
      if ((e.target as HTMLElement).closest?.('[data-tip]')) tip.classList.remove('on')
    }
    document.addEventListener('pointerdown', onDown, true)
    document.addEventListener('scroll', hide, true)
    document.addEventListener('mouseover', onOver)
    document.addEventListener('mouseout', onOut)
    return () => {
      window.clearTimeout(hideTimer)
      document.removeEventListener('pointerdown', onDown, true)
      document.removeEventListener('scroll', hide, true)
      document.removeEventListener('mouseover', onOver)
      document.removeEventListener('mouseout', onOut)
      tip.remove()
    }
  }, [])

  /** Реплика штурмана на экран и в журнал: субтитр живёт 4 с, клип короче. */
  const showShtrum = (line: ShtrumLine, he: boolean) => {
    if (shtrumHideRef.current) window.clearTimeout(shtrumHideRef.current)
    setShtrumCaption({ text: line.text, he })
    shtrumHideRef.current = window.setTimeout(() => setShtrumCaption(null), 4000)
    sayShtrum(line.key, he, line.variant)
    setLog((rows) => [`${he ? '♂ Штруман' : '♀ Штрумана'}: ${line.text}`, ...rows].slice(0, 14))
  }

  /** Реплика «Кобры» — тем же порядком: субтитр сразу, звук по своей очереди. */
  const showWendy = (line: WendyLine) => {
    if (wendyHideRef.current) window.clearTimeout(wendyHideRef.current)
    setWendyCaption({ text: line.text })
    wendyHideRef.current = window.setTimeout(() => setWendyCaption(null), 4000)
    sayWendy(line.key, line.variant)
    setLog((rows) => [`♀ Кобра: ${line.text}`, ...rows].slice(0, 14))
  }

  /** Реплика командира роя — тот же порядок: субтитр сразу, звук по очереди. */
  const showRoy = (line: { key: RoyKey; variant: number; text: string }) => {
    if (royHideRef.current) window.clearTimeout(royHideRef.current)
    setRoyCaption({ text: line.text })
    royHideRef.current = window.setTimeout(() => setRoyCaption(null), 4000)
    sayRoy(line.key, line.variant)
    setLog((rows) => [`♀ Командир роя: ${line.text}`, ...rows].slice(0, 14))
  }

  const playFrames = async (
    frames: Frame[],
    summary?: string | null | (() => string | null),
    onFrame?: (fr: Frame) => void,
    shtrum?: ShtrumPlan | null,
    he = false,
    // потоковый прогон: кадры доезжают во время игры (frames растёт in place),
    // план штурмана и сводка прогона приходят позже начала — берём лениво
    live?: { done: () => boolean; shtrum: () => ShtrumPlan | null; wendy: () => WendyPlan | null },
    // вердикт «взяла/промах» берём из метрик (m.hit), а не из начала сводки:
    // в дуэли сводка начинается с «Ракета взяла» и текстовый тест давал промах при взятии
    hit?: boolean | null | (() => boolean | null),
    // план «Кобры» — только для дуэли (её реплики рождаются из исхода боя двух мозгов)
    wendy?: WendyPlan | null,
  ) => {
    // при включённой озвучке полёт идёт вдвое медленнее: фразы «пуск → захват → финал»
    // успевают прозвучать; без звука темп прежний
    const dt = soundOn ? 80 : 40
    let si = 0
    let wi = 0
    let lastPlan: ShtrumPlan | null = null
    let lastWendyPlan: WendyPlan | null = null
    for (let i = 0; ; i++) {
      if (stop.current) return
      if (i >= frames.length) {
        if (live && !live.done()) {
          await new Promise((r) => setTimeout(r, 25)) // игра догнала счёт — ждём следующие кадры
          i--
          continue
        }
        break
      }
      const raw = frames[i]
      const fr = rusFrame(raw)
      setFrame(fr)
      onFrame?.(fr)
      if (fr.event) setLog((rows) => [`t=${fmt(fr.t, 2)} с · ${fr.event}`, ...rows].slice(0, 14))
      const plan = shtrum ?? live?.shtrum() ?? null
      if (plan && plan !== lastPlan) {
        // план припозднился (метрики пришли в середине проигрывания):
        // реплики, для которых время уже вышло, не догоняют playback толпой
        while (si < plan.inFlight.length && plan.inFlight[si].t <= fr.t) si++
        lastPlan = plan
      }
      while (plan && si < plan.inFlight.length && plan.inFlight[si].t <= fr.t) {
        showShtrum(plan.inFlight[si], he)
        si++
      }
      const wplan = wendy ?? live?.wendy() ?? null
      if (wplan && wplan !== lastWendyPlan) {
        while (wi < wplan.inFlight.length && wplan.inFlight[wi].t <= fr.t) wi++
        lastWendyPlan = wplan
      }
      while (wplan && wi < wplan.inFlight.length && wplan.inFlight[wi].t <= fr.t) {
        showWendy(wplan.inFlight[wi])
        wi++
      }
      await new Promise((r) => setTimeout(r, dt)) // замедленное проигрывание: полёт читается глазами
    }
    const sum = typeof summary === 'function' ? summary() : summary
    if (sum) {
      const got = typeof hit === 'function' ? hit() : hit
      const gotHit = got ?? sum.startsWith('Перехват')
      say(gotHit ? 'hit' : 'miss')
      buzz(gotHit ? HAPTIC.hit : HAPTIC.miss)
      setDone(sum)
    }
    const finalPlan = shtrum ?? live?.shtrum() ?? null
    if (finalPlan?.final) showShtrum(finalPlan.final, he)
    const finalWendy = wendy ?? live?.wendy() ?? null
    if (finalWendy?.final) showWendy(finalWendy.final)
  }

  /** Демо-прогон первого визита: та же playFrames-раскадровка, что у настоящего пуска,
   *  но данные — из синхронного localRun (ноль сетевых запросов, честная работа на
   *  GitHub Pages) и НЕ попадают в лабораторию (labRef/recordRun) — это витрина, а не
   *  прогон, который должен смешаться с историей реальных экспериментов. */
  const runDemo = () => {
    // флаг «показывали» ставится здесь, а не в эффекте-планировщике: React
    // StrictMode в dev монтирует эффект дважды (mount→cleanup→mount) — если
    // писать флаг в самом эффекте, первый (одноразовый) вызов помечает демо
    // «показанным» раньше, чем второй (настоящий) успевает поставить свой
    // таймер, и демо не показывается вовсе. Здесь же исполнение гарантированно
    // одно: до этой строки доходит только таймер, который дожил до срабатывания.
    try {
      localStorage.setItem('muholet-demo-seen', '1')
    } catch {
      /* приватный режим */
    }
    const frames = [...localRun(DEMO_SCENARIO)]
    if (frames.length < 2 || stop.current) return
    lastFramesRef.current = frames
    const m = computeRunMetrics(frames, DEMO_SCENARIO.kill_radius_m)
    setLog((rows) => ['Демо-перехват (первый визит) — управление вернётся после показа.', ...rows].slice(0, 14))
    void playFrames(frames, summaryOf(m), undefined, null, false, undefined, m.hit).then(() => {
      if (!stop.current) setDemoBanner(true)
    })
  }

  // Демо при первом визите: без сохранённых панелей/секций — значит стенд открыли впервые.
  // Флаг «показывали» ставится внутри runDemo(), не здесь — см. комментарий там
  // (React StrictMode дважды монтирует этот эффект в dev). Прерывается кликом/
  // навигацией через тот же stop.current, что и настоящий пуск (playFrames
  // проверяет его каждую итерацию).
  useEffect(() => {
    let seen = true
    try {
      seen = localStorage.getItem('muholet-demo-seen') === '1'
    } catch {
      /* приватный режим — считаем «уже показывали», чтобы не мигать демо каждый визит */
    }
    if (seen || hasPriorUsage()) return
    const t = window.setTimeout(runDemo, 400)
    return () => window.clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  /** Метрики прогона в историю лаборатории: серверные поля если есть, иначе считаем по кадрам. */
  const recordRun = (
    frames: Frame[],
    server?: {
      miss_m: number
      hit: boolean
      t_guide?: number | null
      ref_dev_m?: number | null
      ref_nrms?: number | null
      n_peak: number
      fov_lock_frac: number
      h_cv_m?: number | null
      h0_m?: number | null
      end_range_m?: number | null
      impact_angle_deg?: number | null
      n_eff_median?: number | null
      n_eff_q25?: number | null
      n_eff_q75?: number | null
      n_eff_valid_frac?: number
      sat_frac?: number
      corr_n_eff_rho?: number | null
      corr_n_eff_tgo?: number | null
    },
  ): RunMetrics => {
    const m: RunMetrics = server
      ? {
          ...computeRunMetrics(frames, sc.kill_radius_m),
          metricsVersion: 4,
          miss: server.miss_m,
          cpaM: server.miss_m,
          triggerRangeM: sc.kill_radius_m,
          hit: server.hit,
          hCvM: server.h_cv_m ?? null,
          h0M: server.h0_m ?? null,
          endRangeM: server.end_range_m ?? null,
          impactAngleM: server.impact_angle_deg ?? null,
          terminalZem: server.h_cv_m ?? null,
          tGuide: server.t_guide ?? null,
          tEnd: frames.length ? frames[frames.length - 1].t : 0,
          nPeak: server.n_peak,
          nMean: 0,
          nInt: frames.reduce((acc, fr, i) => {
            const prev = frames[i - 1]
            return acc + (prev ? fr.n_req * Math.max(0, fr.t - prev.t) : 0)
          }, 0),
          lockFrac: server.fov_lock_frac,
          refDev: server.ref_dev_m ?? null,
          refNrms: server.ref_nrms ?? null,
          nEffMedian: server.n_eff_median ?? null,
          nEffQ25: server.n_eff_q25 ?? null,
          nEffQ75: server.n_eff_q75 ?? null,
          nEffValidFrac: server.n_eff_valid_frac ?? 0,
          satFrac: server.sat_frac ?? 0,
          corrNEffRho: server.corr_n_eff_rho ?? null,
          corrNEffTgo: server.corr_n_eff_tgo ?? null,
        }
      : computeRunMetrics(frames, sc.kill_radius_m)
    labRef.current.runs = [
      {
        label: `${ASPECT_LABEL[sc.aspect]} · ${MODE_LABEL[sc.mode]}`,
        at: Date.now(),
        miss: m.miss,
        hit: m.hit,
        tGuide: m.tGuide,
        refDev: m.refDev,
        nPeak: m.nPeak,
        lockFrac: m.lockFrac,
        eta: m.impactAngleM ?? null,
      },
      ...labRef.current.runs,
    ].slice(0, 30)
    return m
  }
  const summaryOf = (m: RunMetrics, duel?: { result: 'missile' | 'evader' | null; fuse: boolean } | null) =>
    `${duel ? (duel.result === 'missile' ? 'Ракета взяла' : duel.fuse ? 'Цель пережила ракету' : 'Цель ушла') : m.hit ? 'Перехват' : 'Промах'}: кратчайшее ${fmt(m.miss, 1)} м · перегрузка до ${fmt(m.nPeak, 1)} · захват ${fmt(m.lockFrac * 100, 0)}%${
      m.tGuide !== null ? ` · время ${fmt(m.tGuide, 2)} с · откл. от эталона ${fmt(m.refDev, 0)} м` : ''
    }${m.impactAngleM !== null && m.impactAngleM !== undefined ? ` · η ${fmt(m.impactAngleM, 0)}°` : ''}`

  // «Промах» при далёкой ручной цели — не тайна: успеет ли вообще состояться сближение
  const logFreeWindow = (m: RunMetrics) => {
    if (m.hit || sc.aspect !== 'free') return
    const d = Math.hypot(sc.free_tx, sc.free_ty, sc.free_talt - sc.alt_m)
    const tMin = d / Math.max(sc.v_m + sc.v_t, 1)
    if (tMin > sc.t_max)
      setLog((rows) => [`Свободная расстановка: до цели ${(d / 1000).toFixed(0)} км — даже на встречных курсах сближение ≥ ${(tMin / 60).toFixed(1)} мин, а окно счёта ${sc.t_max} с: перехват не успевает (граница расчёта, не физика)`, ...rows].slice(0, 14))
  }

  /** Честный замедленный повтор финала: тот же Playback-механизм, что у showDuelReplay —
   *  хвост реальных кадров последних ~1.5с, показан медленнее (данные те же, просто темп
   *  ниже). Только «Полёт», только перехват, не дуэль/рой, не под активной озвучкой —
   *  не спорит с уже идущими фразами штурмана. Пропускается на слабом тире устройства и
   *  при prefers-reduced-motion. myRunId — если за 500мс паузы начался новый пуск,
   *  устаревший повтор не перекрывает его свежий playback. */
  const maybeInstantReplay = (myRunId: number, frames: Frame[], hit: boolean | null, isDuel: boolean) => {
    if (ws !== 'flight' || !hit || isDuel || soundOn || frames.length < 2) return
    if (!perfBudget().allowCinematicReplay) return
    try {
      if (matchMedia('(prefers-reduced-motion: reduce)').matches) return
    } catch {
      /* приватный режим/старый браузер — просто не пропускаем через это условие */
    }
    const tEnd = frames[frames.length - 1].t
    const tail = frames.filter((fr) => fr.t >= tEnd - 1.5)
    if (tail.length < 2) return
    const traj_m = tail.map((fr) => fr.missile)
    const traj_t = tail.map((fr) => fr.target)
    const windowMs = Math.max(1, (tail[tail.length - 1].t - tail[0].t) * 1000)
    const durationMs = Math.max(2000, Math.min(6000, windowMs * 3.5))
    window.setTimeout(() => {
      if (runIdRef.current !== myRunId) return // новый пуск уже начался — не перекрываем его
      setSceneBadge('Повтор · ×3.5 замедление')
      setPlayback({ results: [{ traj_m, traj_t, fitness: 0, hit: true }], bestIdx: 0, startedAt: performance.now(), durationMs, cinematic: true })
    }, 500)
  }

  /** Потоковый прогон по /api/ws/run: движение начинается, когда прилетели
   *  первые ~0,5 с траектории, а сервер ещё доинтегрирует хвост (вся пауза
   *  «Пуск → кадр» была в полном ожидании POST-ответа, 3–6 с). Метрики
   *  догоняют игру сообщением 'done' — вердикт, лаборатория и реплики
   *  штурмана встают по ним. false — коннект/старт не удался до первого
   *  кадра: run() пересчитает прежним POST (в том числе офлайн-локалом). */
  const streamRun = async (
    body: Record<string, unknown>,
    he: boolean,
    flown: Scenario = sc,
  ): Promise<{ ok: boolean; frames: Frame[]; hit: boolean | null; duel: boolean }> => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    let ws: WebSocket
    try {
      ws = new WebSocket(`${proto}://${location.host}/api/ws/run`)
    } catch {
      return { ok: false, frames: [], hit: null, duel: false }
    }
    const frames: Frame[] = []
    let answer: Record<string, unknown> & { frames?: Frame[] } | null = null
    let finished = false // сервер досчитал (done/error) или коннект умер
    let summary: string | null = null
    let plan: ShtrumPlan | null = null
    let wendyPlan: WendyPlan | null = null
    let hitFinal: boolean | null = null
    let isDuelFinal = false
    const finalize = () => {
      if (!answer) return
      lastFramesRef.current = frames
      labRef.current.zem = frames.map(framePoint)
      labRef.current.neff = frames.map(neffPoint)
      labRef.current.traj = sampleTraj(frames)
      const m = recordRun(frames, answer as unknown as Parameters<typeof recordRun>[1] | undefined)
      hitFinal = m.hit
      logFreeWindow(m)
      isDuelFinal = Boolean(answer.duel)
      const duelInfo = answer.duel
        ? { result: (answer.duel_result as 'missile' | 'evader' | null) ?? null, tSurvived: (answer.t_survived as number | null) ?? null, fuse: Boolean(answer.fuse_expired) }
        : null
      setDuelVerdict(duelInfo)
      summary = summaryOf(m, duelInfo)
      if (humorOn) plan = planShtrum(frames, m, flown, he)
      // «Кобра» выходит на связь только в дуэли: она — пилот самолёта-цели,
      // а в обычном прогоне цель — безмозглый маневр, говорить некому
      if (humorOn && duelInfo) wendyPlan = planWendy(frames, m, flown, duelInfo)
    }
    const opened = await new Promise<boolean>((resolve) => {
      let settled = false
      const once = (v: boolean) => {
        if (!settled) {
          settled = true
          window.clearTimeout(guard)
          resolve(v)
        }
      }
      const guard = window.setTimeout(() => once(false), 8000) // стрим не поднялся — не держим «Пуск»
      ws.onopen = () => ws.send(JSON.stringify(body))
      ws.onerror = () => once(false)
      ws.onclose = () => {
        finished = true
        once(frames.length > 0)
      }
      ws.onmessage = (ev) => {
        let msg: { type: string } & Record<string, unknown>
        try {
          msg = JSON.parse(ev.data as string)
        } catch {
          finished = true
          once(false)
          return
        }
        if (msg.type === 'frame') {
          frames.push(msg as unknown as Frame)
          if (frames.length >= 12) once(true)
        } else if (msg.type === 'done') {
          answer = msg
          finished = true
          finalize()
          once(true)
        } else {
          finished = true
          once(false)
        }
      }
    })
    if (!opened) {
      try {
        ws.close()
      } catch {
        /* уже закрыт */
      }
      return { ok: false, frames, hit: null, duel: false }
    }
    await playFrames(frames, () => summary, undefined, null, he, {
      done: () => finished,
      shtrum: () => plan,
      wendy: () => wendyPlan,
    }, () => hitFinal)
    try {
      ws.close()
    } catch {
      /* уже закрыт */
    }
    return { ok: true, frames, hit: hitFinal, duel: isDuelFinal }
  }

  // patch — явная надбавка к сценарию для запуска «прямо сейчас» из дуэльного
  // космоса: setState ещё не подействовал, а дрессированный дуэт хочется
  // летящим в этом же клике (Красная королева: «Дуэль чемпионов», «Повторить бой»)
  const run = async (patch?: Partial<Scenario>) => {
    const myRunId = ++runIdRef.current
    const s = patch ? ({ ...sc, ...patch } as Scenario) : sc
    stop.current = true
    await new Promise((r) => setTimeout(r, 40))
    stop.current = false
    setBusy(true)
    setDone(null)
    setDuelVerdict(null)
    setFrame(null)
    setPlayback(null)
    setSceneBadge(null)
    setSwarmCurve(null)
    setSwarmValid([])
    setDemoBanner(false) // настоящий пуск — демо-баннер своё сказал
    if (shtrumHideRef.current) window.clearTimeout(shtrumHideRef.current)
    setShtrumCaption(null)
    if (wendyHideRef.current) window.clearTimeout(wendyHideRef.current)
    setWendyCaption(null)
    if (royHideRef.current) window.clearTimeout(royHideRef.current)
    setRoyCaption(null)
    setLog((rows) => [`Пуск: ${ASPECT_LABEL[s.aspect]}, ${MODE_LABEL[s.mode]}`, ...rows].slice(0, 14))
    say('launch')
    // пол напарника — противоположный текущему голосу; фиксируется на старте
    // прогона: смена голоса в середине не переключает уже запланированные реплики
    const he = voiceKind === 'female'

    // стрим: сцена трогается через доли секунды, хвост считается параллельно;
    // если стрим не задался до первого кадра — прежний полный POST ниже
    let streamed = false
    try {
      const r = await streamRun({ ...s, brain: s.brain }, he, s)
      streamed = r.ok
      if (streamed) {
        saveLab()
        labTick()
        maybeInstantReplay(myRunId, r.frames, r.hit, r.duel)
      }
    } catch {
      streamed = false // любое падение стрима — пересчитываем прежним путём
    }
    if (streamed) {
      setBusy(false)
      return
    }

    try {
      const res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...s, brain: s.brain }),
      })
      if (res.ok) {
        const data = (await res.json()) as {
          frames: Frame[]
          miss_m: number
          hit: boolean
          n_peak: number
          fov_lock_frac: number
          t_guide?: number | null
          ref_dev_m?: number | null
          ref_nrms?: number | null
          n_eff_median?: number | null
          n_eff_q25?: number | null
          n_eff_q75?: number | null
          n_eff_valid_frac?: number
          sat_frac?: number
          corr_n_eff_rho?: number | null
          corr_n_eff_tgo?: number | null
          duel?: boolean
          duel_result?: 'missile' | 'evader' | null
          fuse_expired?: boolean
          t_survived?: number | null
        }
        const frames = data.frames || []
        lastFramesRef.current = frames
        labRef.current.zem = frames.map(framePoint)
        labRef.current.neff = frames.map(neffPoint)
        labRef.current.traj = sampleTraj(frames)
        const m = recordRun(frames, data)
        logFreeWindow(m)
        const duelInfo = data.duel
          ? { result: data.duel_result ?? null, tSurvived: data.t_survived ?? null, fuse: Boolean(data.fuse_expired) }
          : null
        setDuelVerdict(duelInfo)
        await playFrames(
          frames,
          summaryOf(m, duelInfo),
          undefined,
          humorOn ? planShtrum(frames, m, s, he) : null,
          he,
          undefined,
          m.hit,
          humorOn && duelInfo ? planWendy(frames, m, s, duelInfo) : null,
        )
        saveLab()
        labTick()
        maybeInstantReplay(myRunId, frames, m.hit, Boolean(duelInfo))
        return
      }
      throw new Error(`сервер ${res.status}`)
    } catch {
      setLog((rows) => ['Стенд не отвечает — считаю здесь, в окне.', ...rows].slice(0, 14))
      const frames = [...localRun(s)]
      lastFramesRef.current = frames
      labRef.current.zem = frames.map(framePoint)
      labRef.current.neff = frames.map(neffPoint)
      labRef.current.traj = sampleTraj(frames)
      const m = recordRun(frames)
      logFreeWindow(m)
      await playFrames(frames, summaryOf(m), undefined, humorOn ? planShtrum(frames, m, s, he) : null, he, undefined, m.hit)
      saveLab()
      labTick()
      maybeInstantReplay(myRunId, frames, m.hit, false)
    } finally {
      setBusy(false)
    }
  }

  const wait = (ms: number) => new Promise<void>((r) => setTimeout(r, ms))

  const startSwarm = async () => {
    swarmStop.current = false
    stop.current = true
    setFrame(null)
    setPlayback(null)
    setSwarmRunning(true)
    setBusy(true)
    setDone(null)
    if (!populationRef.current || populationRef.current.length !== swarmCfg.size) {
      populationRef.current = initPopulation(swarmCfg.size, 7)
      setLog((rows) => [`Рой выпущен: ${swarmCfg.size} мух (мозг + ПН)`, ...rows].slice(0, 14))
    }
    let seed = 7
    let gen = 0
    if (humorOn) showRoy(royLine('launch', 0))
    // память поколений для командира: прогресс, перехваты, разнообразие, чемпион
    let prevBest: number | null = null
    let prevHit = 0
    let prevDiv = -1
    let prevChamp: number | undefined
    try {
      while (!swarmStop.current) {
        gen += 1
        // каждое поколение — новая геометрия, иначе рой учится на одной траектории
        const genSc = sampleGenerationScenario(sc, gen, 7)
        const validate = gen % 5 === 0
        const scenarios = validate ? [genSc, ...canonicalScenarios(sc)] : [genSc]
        // адаптивная мутация: застой усиливает поиск, прогресс ослабляет
        const genPts = labRef.current.gen
        let mFactor = 1
        if (genPts.length >= 3) {
          const tail = genPts.slice(-3).map((p) => p.best)
          const improved = tail.every((v, i) => i === 0 || v < tail[i - 1] - 1e-9)
          mFactor = improved ? 0.75 : 1.4
        }
        const res: GenerationResult = await runGeneration(scenarios, populationRef.current!, {
          eliteK: swarmCfg.eliteK,
          mutation: Math.max(0.05, Math.min(1, swarmCfg.mutation * mFactor)),
          seed,
        })
        seed += 1
        populationRef.current = res.next_population.map(sanitizeFly)
        const maxPts = res.results.reduce((m, r) => Math.max(m, r.traj_m.length), 2)
        const durationMs = Math.max(1600, Math.min(9000, maxPts * 34))
        setPlayback({ results: res.results, bestIdx: res.stats.best_idx, startedAt: performance.now(), durationMs })
        const best = res.results[res.stats.best_idx]
        const bio = res.results.filter((r) => r.fly.kind === 'bio').length
        setSceneBadge(`рой · поколение ${gen}${validate ? ' · валидация' : ''} · наименьшее сближение ${fmt(best.miss_m, 0)} м`)
        setSwarmInfo({
          gen,
          bestMiss: best.miss_m,
          avgFit: res.stats.avg,
          bio,
          pn: res.results.length - bio,
          fits: res.results.map((r) => r.fitness),
          geo: scenarioLabel(genSc),
          champion: res.stats.canon_best,
        })
        // командир роя: не больше одной реплики за поколение — по самому
        // значимому событию; кулдауны и очередь шины добьются тишины в эфире
        if (humorOn) {
          const champ = res.stats.canon_best
          const hitR = res.stats.hit_rate ?? 0
          const div = res.stats.diversity ?? 0
          let rkey: RoyKey | null = null
          if (validate && champ !== undefined && prevChamp !== undefined && champ < prevChamp - 1e-9) rkey = 'champion'
          else if (validate) rkey = 'validate'
          else if (mFactor > 1) rkey = 'stagnation'
          else if (mFactor < 1) rkey = 'calm'
          else if (hitR > prevHit + 1e-9) rkey = 'hits'
          else if (prevDiv >= 0 && div < 0.9 * prevDiv) rkey = 'tight'
          else if (prevBest !== null && best.miss_m < prevBest - 1) rkey = 'lead'
          else if (gen % 3 === 2) rkey = 'new_geo'
          if (rkey) showRoy(royLine(rkey, gen))
          if (champ !== undefined) prevChamp = champ
          prevBest = best.miss_m
          prevHit = hitR
          prevDiv = div
        }
        setLog((rows) =>
          [
            `Поколение ${gen}${validate ? ' (валидация)' : ''} · ${scenarioLabel(genSc)} · наименьшее сближение ${fmt(best.miss_m, 0)} м${res.local ? ' · в окне' : ''}`,
            ...rows,
          ].slice(0, 14),
        )
        labRef.current.gen.push({
          gen,
          best: best.miss_m,
          avg: res.stats.avg,
          worst: res.stats.worst,
          hitRate: res.stats.hit_rate,
          diversity: res.stats.diversity,
          champ: res.stats.canon_best,
          local: Boolean(res.local),
        })
        setSwarmCurve(labRef.current.gen.map((p) => p.best))
        setSwarmValid(labRef.current.gen.flatMap((p, i) => (p.champ !== undefined ? [i] : [])))
        saveLab()
        labTick()
        // сцена живёт по телеметрии лидера, пока летит поколение
        const tel = best.tel ?? []
        const t0 = performance.now()
        const iv = window.setInterval(() => {
          const prog = Math.min(1, Math.max(0, (performance.now() - t0) / durationMs))
          if (tel.length >= 2) setFrame(rusFrame(synthFrameFromTel(genSc, tel, prog)))
        }, 90)
        await wait(durationMs + 120)
        clearInterval(iv)
      }
    } finally {
      setSwarmRunning(false)
      setBusy(false)
    }
  }

  const stopSwarm = () => {
    swarmStop.current = true
    if (humorOn) showRoy(royLine('stop', 99))
    setLog((rows) => ['Рой остановлен.', ...rows].slice(0, 14))
  }

  /** Демо «стажёр против ветерана»: необученная муха и лучшая из роя летят одну цель одновременно. */
  const rookieVsVeteran = () => {
    setFrame(null)
    setPlayback(null)
    setSwarmCurve(null)
    setSwarmValid([])
    const rookie: FlyGenome = { kind: 'bio', w: Array(16).fill(0), gain: 1, pn_n: 4 }
    // ветеран — элита последнего поколения роя; без роя — врождённый рефлекс
    const veteran = sanitizeFly(populationRef.current?.[0] ?? { kind: 'bio' })
    const rr = flyRollout(sc, rookie)
    const vr = flyRollout(sc, veteran)
    const maxPts = Math.max(rr.traj_m.length, vr.traj_m.length, 2)
    const durationMs = Math.max(5000, Math.min(14000, maxPts * 45))
    setPlayback({
      results: [
        { ...rr, label: `стажёр · ${fmt(rr.miss_m, 0)} м` },
        { ...vr, label: `ветеран · ${fmt(vr.miss_m, 0)} м` },
      ],
      bestIdx: 1,
      startedAt: performance.now(),
      durationMs,
      race: true,
    })
    setSceneBadge(
      `стажёр против ветерана · стажёр ${rr.hit ? 'попал' : `${fmt(rr.miss_m, 0)} м`} · ветеран ${vr.hit ? 'попал' : `${fmt(vr.miss_m, 0)} м`}`,
    )
    setLog((rows) => [
      `Демо «стажёр против ветерана»: необученная муха (${fmt(rr.miss_m, 0)} м) против лучшей из роя (${fmt(vr.miss_m, 0)} м) — одна цель.`,
      ...rows,
    ].slice(0, 14))
    const t0 = performance.now()
    const iv = window.setInterval(() => {
      const prog = Math.min(1, Math.max(0, (performance.now() - t0) / durationMs))
      const tel = vr.tel ?? []
      if (tel.length >= 2) setFrame(rusFrame(synthFrameFromTel(sc, tel, prog)))
    }, 90)
    window.setTimeout(() => clearInterval(iv), durationMs + 150)
  }

  /** Демо «муха против ПН»: обученный мозг против эталонного ПН на одной цели. */
  const flyVsPN = async () => {
    setFrame(null)
    setPlayback(null)
    setSwarmCurve(null)
    setSwarmValid([])
    setBusy(true)
    try {
      const runOne = async (over: Partial<Scenario>) => {
        const res = await fetch('/api/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...sc, brain: sc.brain, ...over }),
        })
        if (!res.ok) throw new Error(`сервер ${res.status}`)
        const d = (await res.json()) as { frames: Frame[]; miss_m: number; hit: boolean }
        const frames = d.frames || []
        return {
          traj_m: frames.map((f) => f.missile),
          traj_t: frames.map((f) => f.target),
          fitness: d.miss_m,
          hit: Boolean(d.hit),
          miss_m: d.miss_m,
        }
      }
      const pn = await runOne({ mode: 'pn', law: 'pn' })
      const fly = await runOne({ mode: 'bio' })
      const maxPts = Math.max(pn.traj_m.length, fly.traj_m.length, 2)
      const durationMs = Math.max(6000, Math.min(16000, maxPts * 40)) // демо беззвучно — темп обычный
      setPlayback({
        results: [
          { ...pn, label: `ПН (эталон) · ${fmt(pn.miss_m, 0)} м` },
          { ...fly, label: `муха · ${fmt(fly.miss_m, 0)} м` },
        ],
        bestIdx: fly.miss_m <= pn.miss_m ? 1 : 0,
        startedAt: performance.now(),
        durationMs,
        race: true,
      })
      const winner = fly.miss_m <= pn.miss_m ? 'муха точнее' : 'ПН точнее'
      setSceneBadge(`муха против ПН · ПН ${fmt(pn.miss_m, 0)} м · муха ${fmt(fly.miss_m, 0)} м · ${winner}`)
      setLog((rows) => [
        `Гонка «муха против ПН»: эталон ${fmt(pn.miss_m, 0)} м, обученная муха ${fmt(fly.miss_m, 0)} м — ${winner}.`,
        ...rows,
      ].slice(0, 14))
    } finally {
      setBusy(false)
    }
  }

  const downloadText = (name: string, text: string, mime = 'text/csv;charset=utf-8') => {
    const blob = new Blob([text], { type: mime })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = name
    a.click()
    URL.revokeObjectURL(a.href)
  }

  const downloadJson = (name: string, payload: unknown) => {
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = name
    a.click()
    URL.revokeObjectURL(a.href)
  }

  const saveBrain = async () => {
    try {
      const res = await fetch(`/api/brain/export?kind=${sc.brain}`)
      if (!res.ok) throw new Error('стенд не отвечает')
      const data = (await res.json()) as { n_cells: number; tau?: number; wiring?: unknown }
      downloadJson(`navedenie-mozg-${sc.brain}.json`, data)
      const hidden = data.wiring && typeof data.wiring === 'object' && 'blocks' in data.wiring
      setLog((rows) => [
        `Мозг «${sc.brain}» сохранён целиком (${data.n_cells} нейронов${hidden ? ', скрытые слои в проводке' : ''}, тау ${fmt(data.tau ?? 0, 3)} с).`,
        ...rows,
      ].slice(0, 14))
    } catch {
      setLog((rows) => ['Сохранить мозг можно только при работающем стенде.', ...rows].slice(0, 14))
    }
  }

  /** Полный снимок роя: история поколений, конфиг, финальная популяция, чемпион. */
  const swarmExport = () => {
    downloadJson('muholet-roy.json', {
      kind: 'muholet-swarm-experiment',
      savedAt: new Date().toISOString(),
      scenario: sc,
      cfg: swarmCfg,
      champion: swarmInfo?.champion ?? null,
      generations: labRef.current.gen,
      population: populationRef.current ?? [],
    })
    setLog((rows) => [`Снимок роя выгружен в файл: ${labRef.current.gen.length} поколений, популяция ${populationRef.current?.length ?? 0} мух.`, ...rows].slice(0, 14))
  }

  /** История поколений роя в CSV (экспорт лаборатории). */
  const exportSwarmCsv = () => {
    exportCsv(
      'muholet-swarm.csv',
      'gen,best_miss,avg_miss,worst_miss,hit_rate,diversity,champion',
      labRef.current.gen.map((p) => [
        p.gen,
        p.best.toFixed(1),
        p.avg.toFixed(1),
        p.worst === undefined ? '' : p.worst.toFixed(1),
        p.hitRate === undefined ? '' : p.hitRate.toFixed(3),
        p.diversity === undefined ? '' : p.diversity.toFixed(4),
        p.champ === undefined ? '' : p.champ.toFixed(1),
      ]),
    )
  }

  /** Диагностика N_экв по кадрам в CSV (экспорт лаборатории). */
  const exportNeffCsv = () => {
    exportCsv(
      'muholet-neff.csv',
      't_s,tgo_s,rho_1s,vc_ms,n_eff,n_eff_valid,saturation,speed_mode',
      labRef.current.neff.map((p) => [
        p.t.toFixed(3),
        p.tgo === null ? '' : p.tgo.toFixed(2),
        p.rho.toFixed(3),
        p.vc.toFixed(1),
        p.nEff === null ? '' : p.nEff.toFixed(3),
        p.valid ? 1 : 0,
        p.sat ? 1 : 0,
        p.speedMode,
      ]),
    )
  }

  /** Чемпион роя (лучшая биомуха) → в обучаемую схему. */
  const championToBrain = async () => {
    const bio = (playback?.results ?? []).filter((r) => r.fly && r.fly.kind === 'bio' && r.miss_m !== undefined)
    if (!bio.length) {
      setLog((rows) => ['В последнем поколении нет биомух — сначала пустите рой.', ...rows].slice(0, 14))
      return
    }
    const bestBio = bio.reduce((a, b) => ((b.miss_m ?? 1e9) < (a.miss_m ?? 1e9) ? b : a))
    const genome = bestBio.fly!
    const w = [genome.w.slice(0, 10), genome.w.slice(10, 20)]
    try {
      const res = await fetch('/api/brain/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kind: 'stub', w, trained: true, gain: genome.gain }),
      })
      const out = (await res.json()) as { ok: boolean; error?: string }
      if (!res.ok || !out.ok) throw new Error(out.error ?? String(res.status))
      setTrained(true)
      setTrainNote(`чемпион роя в схеме · промах ${fmt(bestBio.miss_m, 0)} м`)
      setLog((rows) => [`Чемпион роя (${fmt(bestBio.miss_m, 0)} м) записан в схему — сохраните его кнопкой «В файл».`, ...rows].slice(0, 14))
    } catch {
      applyServerW(w)
      markTrained()
      setTrained(true)
      setTrainNote(`чемпион роя в локальной схеме · промах ${fmt(bestBio.miss_m, 0)} м`)
      setLog((rows) => ['Стенд не отвечает — чемпион роя записан в схему окна.', ...rows].slice(0, 14))
    }
  }

  const resetBrain = async () => {
    if (!window.confirm(`Сбросить «${sc.brain}» к заводскому состоянию? Обученные веса будут удалены безвозвратно.`)) return
    try {
      const res = await fetch('/api/brain/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kind: sc.brain }),
      })
      if (!res.ok) throw new Error(`сервер ${res.status}`)
      const d = (await res.json()) as { n_cells: number }
      setTrainNote('контур не обучен')
      setLog((rows) => [`Мозг «${sc.brain}» сброшен к заводскому (${d.n_cells} нейронов) — обученные веса удалены, остался врождённый рефлекс.`, ...rows].slice(0, 14))
    } catch {
      setLog((rows) => ['Сброс мозга — только при работающем стенде.', ...rows].slice(0, 14))
    }
  }

  const loadBrainFile = (file: File) => {
    const reader = new FileReader()
    reader.onload = async () => {
      try {
        const data = JSON.parse(String(reader.result)) as {
          w?: number[][]
          trained?: boolean
          gain?: number
          tau?: number
          wiring?: Record<string, unknown>
        }
        if (!data.w) throw new Error('нет весов')
        const res = await fetch('/api/brain/import', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            kind: sc.brain,
            w: data.w,
            trained: data.trained ?? true,
            gain: data.gain ?? null,
            tau: data.tau ?? null,
            wiring: data.wiring ?? null,
          }),
        })
        const out = (await res.json()) as { ok: boolean; error?: string; n_cells?: number; trained?: boolean; gain?: number }
        if (out.ok) {
          setTrained(Boolean(out.trained))
          setTrainNote(`загружен из файла · ${out.n_cells} нейронов`)
          setLog((rows) => [`Мозг «${sc.brain}» загружен из файла.`, ...rows].slice(0, 14))
        } else {
          setLog((rows) => [`Загрузка не удалась: ${out.error ?? 'неизвестная ошибка'}`, ...rows].slice(0, 14))
        }
      } catch {
        setLog((rows) => ['Файл не читается — нужен файл, сохранённый этим же стендом.', ...rows].slice(0, 14))
      }
    }
    reader.readAsText(file)
  }

  const [cmp, setCmp] = useState<{ loading: boolean; rows?: CmpRow[]; error?: string } | null>(null)

  const runCompare = async () => {
    setCmp({ loading: true })
    openLab('compare')
    try {
      const res = await fetch('/api/brain/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario: sc,
          kinds: ['stub', 'full', 'connectome'],
          laws: ['tpn', 'apn', 'pure', 'clos', 'pn_gsn'],
        }),
      })
      if (!res.ok) throw new Error(`сервер ${res.status}`)
      const data = (await res.json()) as { results: CmpRow[] }
      // подписи законов приводим к русским независимо от версии стенда
      data.results.forEach((r) => {
        if (r.kind !== 'pn' && r.kind in LAW_RU && r.n_cells === 0) r.label = `закон: ${LAW_RU[r.kind]}`
      })
      setCmp({ loading: false, rows: data.results })
      setLog((rows) =>
        [`Сравнение: ${data.results.map((r) => `${r.label} ${fmt(r.miss_m, 0)} м`).join(' · ')}`, ...rows].slice(0, 14),
      )
    } catch {
      // без сервера честно сравним в окне: эталонный ПН, текущий закон и локальная схема
      try {
        const rows: CmpRow[] = []
        const runLocal = (kind: string, label: string, over: Partial<Scenario>) => {
          const m = computeRunMetrics([...localRun({ ...sc, ...over })], sc.kill_radius_m)
          rows.push({
            kind,
            label,
            n_cells: kind === 'stub' ? 87 : 0,
            trained: kind === 'stub' ? isTrained() : null,
            miss_m: m.miss,
            hit: m.hit,
            n_peak: m.nPeak,
            lock_frac: m.lockFrac,
            t_guide: m.tGuide,
            ref_dev_m: m.refDev,
          })
        }
        runLocal('pn', 'ПН (эталон, окно)', { mode: 'pn', law: 'pn' })
        if (sc.law !== 'pn') runLocal(sc.law, `закон: ${LAW_RU[sc.law] ?? sc.law} (окно)`, { mode: 'pn', law: sc.law })
        runLocal('stub', 'схема (окно)', { mode: 'bio', brain: 'stub' })
        setCmp({ loading: false, rows })
        setLog((rows) => ['Стенд не отвечает — сравнение посчитано в окне (лаборатория → Сравнение).', ...rows].slice(0, 14))
      } catch {
        setCmp({ loading: false, error: 'Локальное сравнение не удалось: коннектом и полный мозг считаются только на стенде.' })
      }
    }
  }

  /** «Ринг»: матрица сторон на текущих условиях прогона (сервер; без сервера — честный отказ). */
  const runDuelMatrix = async () => {
    setDuelBusy(true)
    try {
      const res = await fetch('/api/duel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario: sc, repeats: duelRepeats }),
      })
      if (!res.ok) throw new Error(`сервер ${res.status}`)
      const data = (await res.json()) as DuelMatrix
      setDuelMatrix(data)
      const wins = data.cells.filter((c) => c.win === 'missile').length
      setLog((rows) => [`Ринг: ${data.cells.length} боёв, ракета взяла ${wins}`, ...rows].slice(0, 14))
    } catch {
      setLog((rows) => ['Ринг считается только на стенде — он недоступен.', ...rows].slice(0, 14))
    } finally {
      setDuelBusy(false)
    }
  }

  const exportDuelCsv = () => {
    if (!duelMatrix) return
    const head = 'ракета,цель,вердикт,доля_перехватов,время_цели_с,промах_м,усилие_gс,пик_g'
    const body = duelMatrix.cells
      .map((c) => [c.row, c.col, c.win, c.hit_rate, c.t_survived.toFixed(2), c.cpa_m.toFixed(1), c.n_int.toFixed(1), c.n_peak.toFixed(2)].join(','))
      .join('\n')
    const a = document.createElement('a')
    a.href = URL.createObjectURL(new Blob([`${head}\n${body}\n`], { type: 'text/csv' }))
    a.download = 'muholet-ring.csv'
    a.click()
    URL.revokeObjectURL(a.href)
  }

  /** учебный бой поколения — на сцене: те же проигрываемые траектории, что в «Рое»,
   *  только стороны две (ракета и уклонист), а не веер мух */
  const showDuelReplay = (r: DuelReplay) => {
    const pts = Math.max(r.traj_m.length, r.traj_t.length)
    setPlayback({
      results: [{ traj_m: r.traj_m, traj_t: r.traj_t, fitness: r.fitness ?? 0, hit: r.hit }],
      bestIdx: 0,
      startedAt: performance.now(),
      durationMs: Math.max(1600, Math.min(9000, pts * 34)),
      caption: r.label,
    })
    setSceneBadge(`${r.label} · ${r.hit ? 'цель сбита' : `наименьшее сближение ${fmt(r.cpa_m, 0)} м`}`)
  }

  const refreshExperiments = async () => {
    try {
      const r = await fetch('/api/experiments')
      if (!r.ok) throw new Error()
      setExpList(((await r.json()) as { experiments: { name: string; saved_at: string; size: number }[] }).experiments)
    } catch {
      setExpList([])
    }
  }

  /** Выгрузка последнего прогона для MATLAB: все координаты и метрики по кадрам. */
  const exportMatlab = () => {
    const rows = lastFramesRef.current.map((fr) => {
      const g = fr.ghost ?? [NaN, NaN, NaN]
      const dev = fr.ghost ? dist3(fr.missile, fr.ghost) : NaN
      return [
        fr.t.toFixed(3),
        fr.missile.map((v) => v.toFixed(1)).join(','),
        fr.target.map((v) => v.toFixed(1)).join(','),
        g.map((v) => v.toFixed(1)).join(','),
        fr.n_req.toFixed(2),
        fr.lock ? 1 : 0,
        dev.toFixed(1),
      ].join(',')
    })
    downloadText(
      'muholet-matlab.csv',
      't,m_x,m_y,m_z,t_x,t_y,t_z,g_x,g_y,g_z,n_req,lock,dev\n' + rows.join('\n'),
    )
  }

  /** Исследовательский экспорт: траектории ракеты, цели и эталона ПН + команды, события,
   *  захват, фаза сближения (theta/rho) и диагностика N_экв (CSV). */
  const exportTrajectories = () => {
    const frames = lastFramesRef.current
    if (!frames.length) {
      setLog((rows) => ['Нечего выгружать: сначала сделайте пуск.', ...rows].slice(0, 14))
      return
    }
    const f2 = (v: number) => v.toFixed(2)
    const head =
      '# model_version=' + MODEL_VERSION +
      ' metrics_version=4 feature_schema_version=' + FEATURE_SCHEMA_VERSION +
      ' (истинная геометрия в n_eff/tgo — постфактум-диагностика, не для управления)\n' +
      '# столбцы: range_m=дальность R, м; v_c=скорость сближения V_сбл, м/с; tgo_s=радиальная оценка времени R/V_сбл, с;' +
      ' theta_rad=угол ЛВ φ, рад; theta_dot=ω_ЛВ, рад/с; tau_contact_s=оптическая оценка времени до контакта τ, с;' +
      ' az/el=азимут/угол места цели, рад; n_eff=эквивалентный навигационный коэффициент N_экв, безразм.;' +
      ' n_req_g=заданная нормальная перегрузка, g; saturation=ограничение команды; hit=перехват (пересечение сферы срабатывания);' +
      ' cpa_m=R_min — минимальное расстояние сближения, м; dev_ref_m=отклонение от эталона, м\n' +
      't,event,range_m,v_m,v_t,v_c,tgo_s,theta_rad,theta_dot,tau_contact_s,az,el,az_dot,el_dot,n_eff,n_eff_valid,n_eff_reason,n_req_g,saturation,hit,cpa_m,' +
      'm_x,m_y,m_z,m_vx,m_vy,m_vz,t_x,t_y,t_z,t_vx,t_vy,t_vz,g_x,g_y,g_z,dev_ref_m,a_cmd_x_g,a_cmd_y_g,a_cmd_z_g,dn_pitch,dn_yaw,lock,speed_mode,target_speed'
    let cpa = Number.POSITIVE_INFINITY
    const rows = frames.map((fr) => {
      const g = fr.ghost ?? [NaN, NaN, NaN]
      const dev = fr.ghost ? dist3(fr.missile, fr.ghost) : NaN
      const dn = fr.circuit?.dn
      cpa = Math.min(cpa, fr.miss)
      return [
        fr.t.toFixed(3),
        fr.event ?? '',
        fr.range_m.toFixed(1),
        f2(Math.hypot(...fr.missile_v)),
        f2(Math.hypot(...fr.target_v)),
        f2(fr.v_c),
        fr.tgo !== null && fr.tgo !== undefined ? fr.tgo.toFixed(2) : '',
        (fr.theta ?? 0).toFixed(5),
        (fr.theta_dot ?? 0).toFixed(4),
        (fr.tau_contact ?? 0).toFixed(2),
        fr.az.toFixed(5),
        fr.el.toFixed(5),
        (fr.seeker?.az_dot ?? 0).toFixed(4),
        (fr.seeker?.el_dot ?? 0).toFixed(4),
        fr.n_eff !== null && fr.n_eff !== undefined ? fr.n_eff.toFixed(3) : '',
        fr.n_eff_valid ? 1 : 0,
        fr.n_eff_reason ?? '',
        fr.n_req.toFixed(2),
        fr.sat ? 1 : 0,
        fr.event === 'hit' || fr.event === 'перехват' ? 1 : 0,
        Number.isFinite(cpa) ? cpa.toFixed(1) : '',
        ...fr.missile.map(f2),
        ...fr.missile_v.map(f2),
        ...fr.target.map(f2),
        ...fr.target_v.map(f2),
        ...g.map((v) => f2(v)),
        Number.isFinite(dev) ? dev.toFixed(1) : '',
        ...fr.a_cmd.map((v) => (v / 9.81).toFixed(3)),
        dn ? dn.pitch.toFixed(3) : '',
        dn ? dn.yaw.toFixed(3) : '',
        fr.lock ? 1 : 0,
        fr.speed_mode ?? 'constant',
        (fr.target_speed ?? 0).toFixed(1),
      ].join(',')
    })
    downloadText('muholet-traektorii.csv', head + '\n' + rows.join('\n'))
    setLog((allRows) => [
      `Траектории выгружены: ${frames.length} кадров — геометрия, фаза сближения (θ/ρ), N_экв и версии модели/метрик/схемы.`,
      ...allRows,
    ].slice(0, 14))
  }

  /** Серия прогонов с нарастающим шумом: устойчивость мозгов. */
  const runRobustness = async () => {
    setRobBusy(true)
    try {
      const r = await fetch('/api/robustness', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario: sc, kinds: ['stub', 'full', 'connectome'], noise_levels: [0, 1, 2, 3, 4] }),
      })
      if (!r.ok) throw new Error()
      const d = (await r.json()) as { levels: number[]; series: { kind: string; label: string; miss: number[]; nrms: (number | null)[] }[] }
      setRobData(d)
      setLog((rows) => [`Устойчивость к возмущениям: уровни шума ${d.levels.join(' → ')}°, серии по ${d.series.length} мозгам.`, ...rows].slice(0, 14))
    } catch {
      setLog((rows) => ['Устойчивость к возмущениям считается только на работающем стенде.', ...rows].slice(0, 14))
    } finally {
      setRobBusy(false)
    }
  }

  /** Monte-Carlo рассеивания: n прогонов текущего сценария с разными seed. */
  const runMonteCarlo = async () => {
    setMcBusy(true)
    try {
      const r = await fetch('/api/science/monte-carlo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...sc, n_runs: mcRuns, seed_start: 1000 }),
      })
      if (!r.ok) throw new Error()
      const d = (await r.json()) as MonteCarloData
      setMcData(d)
      setLog((rows) => [`Monte-Carlo: ${d.n_runs} прогонов · p_hit ${(d.p_hit * 100).toFixed(0)} % · СКО R_min ${Math.round(d.r_min_std_m ?? 0)} м.`, ...rows].slice(0, 14))
    } catch {
      setLog((rows) => ['Серия разброса считается только на работающем стенде.', ...rows].slice(0, 14))
    } finally {
      setMcBusy(false)
    }
  }

  /** Зона неубегаемого перехвата: сетка «дальность × постоянная перегрузка цели» + бисекция границ. */
  const runCaptureZone = async () => {
    setCzBusy(true)
    try {
      const gs = Array.from({ length: 6 }, (_, i) => Math.round(((czGmax * i) / 5) * 10) / 10)
      const r = await fetch('/api/science/capture-zone', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...sc, target_gs: gs, refine: true, n_runs: czRuns, seed_start: 1000 }),
      })
      if (!r.ok) throw new Error()
      const d = (await r.json()) as CaptureZoneData
      setCzData(d)
      const b = d.rows.find((x) => x.boundary_m !== undefined)
      setLog(
        rows => [
          `Зона перехвата: ${d.rows.length}×${d.ranges_m.length} сетка${czRuns > 1 ? ` × ${czRuns} прогонов Monte-Carlo на ячейку` : ''} за ${Math.round(d.seconds)} с${b ? ` · при ${b.n_target_g} g зона ${Math.round(b.inner_boundary_m ?? d.ranges_m[0])}…${Math.round(b.boundary_m ?? 0)} м` : ''}.`,
          ...rows,
        ].slice(0, 14),
      )
    } catch {
      setLog((rows) => ['Зона перехвата считается только на работающем стенде.', ...rows].slice(0, 14))
    } finally {
      setCzBusy(false)
    }
  }

  /** Пересоздать мозг с новыми размерами скрытых слоёв (сброс обучения). */
  const rebuildBrain = async (pool: number, channels: number) => {
    try {
      const r = await fetch('/api/brain/rebuild', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kind: sc.brain, pool_size: pool, channels }),
      })
      const d = (await r.json()) as { ok?: boolean; n_cells?: number; error?: string }
      if (!r.ok || !d.ok) throw new Error(d.error ?? String(r.status))
      setTrained(false)
      setTrainNote(`мозг пересоздан · ${d.n_cells} нейронов`)
      setLog((rows) => [`Мозг «${sc.brain}» пересоздан (${d.n_cells} нейронов) — запустите обучение.`, ...rows].slice(0, 14))
    } catch {
      setLog((rows) => ['Пересоздание мозга доступно только при работающем стенде.', ...rows].slice(0, 14))
    }
  }

  /** Снимок роя на сервер, в data/experiments (переживает перезапуск контейнера). */
  const saveExperimentServer = async () => {
    const name = window.prompt('Имя эксперимента:', `рой ${new Date().toLocaleDateString('ru-RU')}`)
    if (!name) return
    try {
      const r = await fetch('/api/experiments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name,
          payload: {
            savedAt: new Date().toISOString(),
            scenario: sc,
            cfg: swarmCfg,
            champion: swarmInfo?.champion ?? null,
            generations: labRef.current.gen,
            population: populationRef.current ?? [],
          },
        }),
      })
      if (!r.ok) throw new Error()
      await refreshExperiments()
      setLog((rows) => [`Эксперимент «${name}» сохранён в архив стенда (папка data/experiments).`, ...rows].slice(0, 14))
    } catch {
      setLog((rows) => ['Архив недоступен — выгрузка в файл работает и без него.', ...rows].slice(0, 14))
    }
  }

  const loadExperimentServer = async (name: string) => {
    try {
      const r = await fetch(`/api/experiments/${encodeURIComponent(name)}`)
      if (!r.ok) throw new Error()
      const d = (await r.json()) as { generations?: GenPoint[]; cfg?: typeof swarmCfg }
      labRef.current.gen = (d.generations ?? []).map((p) => ({
        gen: Number(p.gen) || 0,
        best: Number(p.best) || 0,
        avg: Number(p.avg) || 0,
        worst: p.worst === undefined ? undefined : Number(p.worst),
        hitRate: p.hitRate === undefined ? undefined : Number(p.hitRate),
        champ: p.champ === undefined ? undefined : Number(p.champ),
      }))
      setSwarmCurve(labRef.current.gen.length >= 2 ? labRef.current.gen.map((p) => p.best) : null)
      setSwarmValid(labRef.current.gen.flatMap((p, i) => (p.champ !== undefined ? [i] : [])))
      if (d.cfg) setSwarmCfg((c) => ({ ...c, ...d.cfg }))
      saveLab()
      labTick()
      setLog((rows) => [`Эксперимент «${name}» загружен: ${labRef.current.gen.length} поколений.`, ...rows].slice(0, 14))
    } catch {
      setLog((rows) => [`Не удалось загрузить «${name}».`, ...rows].slice(0, 14))
    }
  }

  const deleteExperimentServer = async (name: string) => {
    if (!window.confirm(`Удалить «${name}» из архива стенда (файл data/experiments/${name}.json)?`)) return
    try {
      const r = await fetch(`/api/experiments/${encodeURIComponent(name)}`, { method: 'DELETE' })
      if (!r.ok) throw new Error()
      await refreshExperiments()
      setLog((rows) => [`Эксперимент «${name}» удалён из архива.`, ...rows].slice(0, 14))
    } catch {
      setLog((rows) => ['Удаление из архива доступно только при работающем стенде.', ...rows].slice(0, 14))
    }
  }

  const train = async () => {
    setBusy(true)
    setTraining(true)
    setTrainNote('обучение…')
    setFrame(null)
    setPlayback(null)
    setDone(null)
    setLog((rows) =>
      [
        `Обучение ${sc.brain === 'full' ? 'полного мозга' : sc.brain === 'connectome' ? 'коннектома' : 'схемы'} · шаг ${trainCfg.lr} · эпизодов ${trainCfg.episodes}…`,
        ...rows,
      ].slice(0, 14),
    )
    const lr = trainCfg.lr
    const totalEpisodes = trainCfg.episodes
    labSessionRef.current += 1

    if (trainCfg.mode === 'result') {
      // доводка выхода по результату (эволюция W_dn) — только на стенде
      try {
        const st = await fetch('/api/train/evolve/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ kind: sc.brain, generations: totalEpisodes, sigma: 0.3, gain: sc.circuit_gain, tau_s: sc.tau_s }),
        })
        if (!st.ok) throw new Error(`сервер ${st.status}`)
        const s0 = (await st.json()) as { generations: number; miss_before: number }
        setSceneBadge(`доводка · старт · промах ${fmt(s0.miss_before, 0)} м`)
        setLog((rows) => [
          `Доводка по результату: ${s0.generations} поколений эволюции выхода, стартовый промах ${fmt(s0.miss_before, 0)} м.`,
          ...rows,
        ].slice(0, 14))
        for (let g = 1; g <= s0.generations; g += 1) {
          if (stop.current) break
          const r = await fetch('/api/train/evolve/step', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ kind: sc.brain }),
          })
          if (!r.ok) throw new Error(`сервер ${r.status}`)
          const d = (await r.json()) as { gen: number; miss_best_overall: number; hit_rate: number; generations: number }
          setTrainProgress({ ep: d.gen, total: s0.generations })
          setSceneBadge(`доводка · поколение ${d.gen}/${s0.generations} · лучший ${fmt(d.miss_best_overall, 0)} м`)
          setTrainNote(`доводка по результату · поколение ${d.gen}/${s0.generations} · промах ${fmt(d.miss_best_overall, 0)} м`)
          labRef.current.train.push({
            ep: labRef.current.train.length + 1,
            miss: d.miss_best_overall,
            hit: (d.hit_rate ?? 0) > 0.34,
            tGuide: null,
            refDev: 0,
            w: [],
            session: labSessionRef.current,
          })
          saveLab()
          labTick()
        }
        const fin = await fetch('/api/train/evolve/finish', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ kind: sc.brain }),
        })
        if (!fin.ok) throw new Error(`сервер ${fin.status}`)
        const fd = (await fin.json()) as { n_cells: number; miss_after: number; hit_rate_after?: number | null }
        setTrained(true)
        setTrainNote(`доведён по результату · ${fd.n_cells} нейронов · промах ${fmt(fd.miss_after, 0)} м · перехваты ${Math.round((fd.hit_rate_after ?? 0) * 100)}%`)
        setSceneBadge(`готов: промах ${fmt(fd.miss_after, 0)} м`)
        setLog((rows) => [
          `Доводка завершена: промах по трио ${fmt(fd.miss_after, 0)} м, перехваты ${Math.round((fd.hit_rate_after ?? 0) * 100)}%.`,
          ...rows,
        ].slice(0, 14))
      } catch {
        setLog((rows) => ['Доводка по результату — только на стенде (сервер не ответил).', ...rows].slice(0, 14))
      } finally {
        setTraining(false)
        setBusy(false)
        setTrainProgress(null)
      }
      return
    }

    const mapTel = (tel: Record<string, unknown>[]) =>
      tel.map((p) => ({
        t: Number(p.t) || 0,
        az: Number(p.az) || 0,
        el: Number(p.el) || 0,
        lock: Boolean(p.lock),
        size: Number(p.size) || 0,
        sizeDot: Number(p.size_dot ?? 0),
        azDot: Number(p.az_dot ?? 0),
        elDot: Number(p.el_dot ?? 0),
        pitch: Number(p.pitch) || 0,
        yaw: Number(p.yaw) || 0,
        nReq: Number(p.n_req ?? 0),
        miss: Number(p.miss) || 0,
        rng: Number(p.rng) || 0,
        v: (p.v as number[]) || [sc.v_m, 0, 0],
      }))
    try {
      const st = await fetch('/api/train/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kind: sc.brain, episodes: totalEpisodes, lr }),
      })
      if (!st.ok) throw new Error(`сервер ${st.status}`)
      const s0 = (await st.json()) as {
        episodes: number
        miss_before: number
        metrics_before?: { miss: number; hit_rate: number; t_guide: number | null; ref_dev: number }
      }
      for (let ep = 0; ep < s0.episodes; ep += 1) {
        if (stop.current) break
        const r = await fetch('/api/train/step', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ kind: sc.brain, ep, lr }),
        })
        if (!r.ok) throw new Error(`сервер ${r.status}`)
        const d = (await r.json()) as {
          miss: number
          hit: boolean
          t_guide: number | null
          ref_dev: number
          nrms: number
          n_avg: number
          n_peak: number
          lock_frac: number
          tel: Record<string, unknown>[]
          w: number[][]
        }
        setTrainNote(`обучение ${ep + 1}/${s0.episodes} · промах ${fmt(d.miss, 0)} м · откл. от эталона ${fmt(d.ref_dev, 0)} м`)
        setSceneBadge(`обучение · эпизод ${ep + 1}/${s0.episodes} · промах ${fmt(d.miss, 0)} м`)
        setTrainProgress({ ep: ep + 1, total: s0.episodes })
        labRef.current.train.push({
          ep: ep + 1,
          miss: d.miss,
          hit: Boolean(d.hit),
          tGuide: d.t_guide ?? null,
          refDev: d.ref_dev ?? 0,
          nrms: d.nrms,
          nAvg: d.n_avg,
          nPeak: d.n_peak,
          lockFrac: d.lock_frac,
          w: d.w.flat(),
          session: labSessionRef.current,
        })
        saveLab()
        labTick()
        // анимация эпизода: глаз/мозг/рули по телеметрии
        const tel = mapTel(d.tel)
        for (let i = 0; i < tel.length - 1; i += 3) {
          if (stop.current) break
          setFrame(rusFrame(synthFrameFromTel(sc, tel, i / Math.max(tel.length - 1, 1))))
          await wait(14)
        }
      }
      const fin = await fetch('/api/train/finish', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kind: sc.brain, miss_before: s0.miss_before }),
      })
      if (!fin.ok) throw new Error(`сервер ${fin.status}`)
      const fd = (await fin.json()) as {
        n_cells: number
        miss_before?: number
        miss_after: number
        hit_rate_after?: number | null
        t_guide_after?: number | null
        ref_dev_after?: number | null
      }
      labRef.current.summary = {
        missBefore: s0.metrics_before?.miss ?? s0.miss_before,
        missAfter: fd.miss_after,
        tGuideAfter: fd.t_guide_after ?? null,
        refDevBefore: s0.metrics_before?.ref_dev,
        refDevAfter: fd.ref_dev_after ?? undefined,
        hitRateAfter: fd.hit_rate_after ?? undefined,
      }
      setTrained(true)
      setTrainNote(
        `обучен · ${fd.n_cells} нейронов · промах ${fmt(s0.metrics_before?.miss ?? s0.miss_before, 0)} → ${fmt(fd.miss_after, 0)} м · откл. ${fmt(
          s0.metrics_before?.ref_dev,
          0,
        )} → ${fmt(fd.ref_dev_after, 0)} м`,
      )
      setLog((rows) =>
        [
          `Обучение готово: промах ${fmt(s0.metrics_before?.miss ?? s0.miss_before, 0)} → ${fmt(fd.miss_after, 0)} м, отклонение от ПН ${fmt(
            s0.metrics_before?.ref_dev,
            0,
          )} → ${fmt(fd.ref_dev_after, 0)} м.`,
          ...rows,
        ].slice(0, 14),
      )
    } catch {
      setLog((rows) => ['Сервер не ответил — учусь здесь, в окне.', ...rows].slice(0, 14))
      const points = trainLocal(
        sc.brain,
        trainCfg.episodes,
        (ep, total, point) => {
          labRef.current.train.push({ ...point, session: labSessionRef.current })
          saveLab()
          labTick()
          if (ep % 4 === 0 || ep === total) {
            setTrainNote(`обучение ${ep}/${total} · промах ${fmt(point.miss, 0)} м · откл. от эталона ${fmt(point.refDev, 0)} м`)
          }
          setSceneBadge(`обучение в окне · эпизод ${ep}/${total} · промах ${fmt(point.miss, 0)} м`)
          setTrainProgress({ ep, total })
        },
        (fr) => setFrame(rusFrame(fr)),
        lr,
      )
      labRef.current.summary = null
      const last = points[points.length - 1]
      setTrained(true)
      setTrainNote(
        `обучен в окне · ${nCellsLabel(sc.brain)} клеток · промах ${fmt(last?.miss ?? 0, 0)} м · откл. ${fmt(last?.refDev ?? 0, 0)} м`,
      )
      setLog((rows) => ['Обучение в окне закончено. Запустите пуск по коннектому.', ...rows].slice(0, 14))
    } finally {
      setTraining(false)
      setBusy(false)
      setTrainProgress(null)
    }
  }

  const set = <K extends keyof Scenario>(key: K, value: Scenario[K]) => {
    if (key === 'brain') setBrainKind(value as Scenario['brain'])
    setSc((s) => ({ ...s, [key]: value }))
  }

  /** Выбор манёвра: при нулевой перегрузке цели манёвр молча ничего не делает — включаем 6 g сами. */
  const onManeuver = (m: Scenario['maneuver']) => {
    if (m !== 'straight' && sc.n_target <= 0) {
      set('n_target', 6)
      setLog((rows) => ['Перегрузка цели была 0 — манёвр бы не сработал. Включил 6 g.', ...rows].slice(0, 14))
    }
    set('maneuver', m)
  }

  // статус в шапке: что сейчас делает стенд
  const running = busy || swarmRunning || training
  const statusText = running ? (training ? trainNote : swarmRunning ? `рой · поколение ${swarmInfo?.gen ?? '…'}` : 'считаю…') : done ?? 'готов к пуску'
  const statusKind: '' | 'is-ok' | 'is-bad' | 'is-warn' = done ? (done.startsWith('Перехват') || done.startsWith('Ракета') ? 'is-ok' : 'is-bad') : running ? 'is-warn' : ''
  const subtitle = `${sc.brain === 'full' ? 'полный мозг' : sc.brain === 'connectome' ? 'коннектом' : 'схема'} · ${trained ? 'обучен' : 'без обучения'} · ${__APP_VERSION__}`

  return (
    <AppShell
      subtitle={subtitle}
      workspace={ws}
      onWorkspace={setWs}
      serverOnline={serverOnline}
      statusText={statusText}
      statusKind={statusKind}
      running={running}
      canLaunch={!busy && !swarmRunning && !training}
      onLaunch={() => void run()}
      onStop={() => {
        stop.current = true
        setShtrumCaption(null)
      }}
      day={day}
      onToggleDay={() => setDay((d) => !d)}
      onHelp={() => setHelpOpen(true)}
      cinema={cinema}
      onToggleCinema={() => {
        buzz(HAPTIC.cinemaToggle)
        setCinema((v) => !v)
      }}
      demoBanner={demoBanner}
      onDemoTour={() => {
        setDemoBanner(false)
        setTourOpen(true)
      }}
      onDemoBannerClose={() => setDemoBanner(false)}
      menuItems={[
        {
          label: soundOn ? 'Озвучка: выключить' : 'Озвучка: включить',
          onClick: () => setSoundOn((v) => !v),
        },
        {
          label: voiceKind === 'female' ? 'Голос: мужской' : 'Голос: женский',
          onClick: () => setVoiceKind((v) => (v === 'female' ? 'male' : 'female')),
        },
        { label: egg ? 'Муха за штурвалом: выключить' : 'Муха за штурвалом (X)', onClick: () => setEgg((v) => !v) },
        {
          label: humorOn ? 'Юмор: выключить' : 'Юмор: вторая муха-штурман',
          onClick: () => setHumorOn((v) => !v),
        },
        { label: cinema ? 'Кинорежим: выключить' : 'Кинорежим (K)', onClick: () => setCinema((v) => !v) },
        { label: 'Показать демо заново', onClick: () => { setWs('flight'); runDemo() } },
        { label: 'Короткий тур: с чего начать', onClick: () => setTourOpen(true) },
        { label: 'Сбросить раскладку панелей', onClick: resetLayout },
      ]}
    >
      {ws === 'flight' && (
        <FlightWorkspace
          frame={frame}
          sc={sc}
          set={set}
          onManeuver={onManeuver}
          playback={playback}
          day={day}
          camMode={camMode}
          onCamMode={setCamMode}
          geometryOn={geometryOn}
          onToggleGeometry={() => setGeometryOn((g) => !g)}
          sceneBadge={sceneBadge}
          swarmCurve={swarmCurve}
          swarmValid={swarmValid}
          egg={egg}
          cofly={humorOn ? (voiceKind === 'female' ? 'm' : 'f') : null}
          shtrumCaption={shtrumCaption}
          wendyCaption={wendyCaption}
          silent={swarmRunning || playback?.race === true}
          sceneIdle={training}
          busy={busy}
          done={done}
          log={log}
          onClearLog={clearLog}
        />
      )}
      {ws === 'brain' && (
        <BrainWorkspace
          frame={frame}
          brain={sc.brain}
          onBrain={(b) => set('brain', b)}
          day={day}
          trained={trained}
          trainNote={trainNote}
          training={training}
          trainProgress={trainProgress}
          trainCfg={trainCfg}
          onTrainCfg={(patch) => setTrainCfg((c) => ({ ...c, ...patch }))}
          busy={busy}
          onTrain={() => void train()}
          onTrainStop={() => {
            stop.current = true
          }}
          onSaveBrain={() => void saveBrain()}
          onLoadBrainFile={loadBrainFile}
          onResetBrain={() => void resetBrain()}
          tune={{ pool: tunePool, channels: tuneChannels }}
          onTune={(patch) => {
            if (patch.pool !== undefined) setTunePool(patch.pool)
            if (patch.channels !== undefined) setTuneChannels(patch.channels)
          }}
          onRebuild={(pool, channels) => void rebuildBrain(pool, channels)}
        />
      )}
      {ws === 'swarm' && (
        <SwarmWorkspace
          frame={frame}
          sc={sc}
          playback={playback}
          day={day}
          camMode={camMode}
          onCamMode={setCamMode}
          geometryOn={geometryOn}
          onToggleGeometry={() => setGeometryOn((g) => !g)}
          sceneBadge={sceneBadge}
          swarmCurve={swarmCurve}
          swarmValid={swarmValid}
          egg={egg}
          royCaption={royCaption}
          silent={swarmRunning || playback?.race === true}
          busy={busy}
          swarmRunning={swarmRunning}
          swarmInfo={swarmInfo}
          swarmCfg={swarmCfg}
          onSwarmCfg={(patch) => setSwarmCfg((c) => ({ ...c, ...patch }))}
          onStartSwarm={() => void startSwarm()}
          onStopSwarm={stopSwarm}
          onRookieVsVeteran={rookieVsVeteran}
          onFlyVsPN={() => void flyVsPN()}
          genHistory={labRef.current.gen}
          championToBrain={() => void championToBrain()}
          swarmExport={swarmExport}
          onExportSwarmCsv={exportSwarmCsv}
          expList={expList}
          onRefreshExperiments={() => void refreshExperiments()}
          onSaveExperimentServer={() => void saveExperimentServer()}
          onLoadExperimentServer={(name) => void loadExperimentServer(name)}
        />
      )}
      {ws === 'duel' && (
        <DuelWorkspace
          frame={frame}
          sc={sc}
          set={set}
          playback={playback}
          day={day}
          camMode={camMode}
          onCamMode={setCamMode}
          geometryOn={geometryOn}
          onToggleGeometry={() => setGeometryOn((g) => !g)}
          sceneBadge={sceneBadge}
          silent={swarmRunning || playback?.race === true}
          busy={busy}
          done={done}
          duelVerdict={duelVerdict}
          shtrumCaption={shtrumCaption}
          wendyCaption={wendyCaption}
          duelMatrix={duelMatrix}
          duelBusy={duelBusy}
          duelRepeats={duelRepeats}
          onDuelRepeats={setDuelRepeats}
          onRunMatrix={() => void runDuelMatrix()}
          onExportCsv={exportDuelCsv}
          onDuelNow={(patch) => void run(patch)}
          onShowReplay={showDuelReplay}
          serverOnline={serverOnline}
        />
      )}
      {ws === 'lab' && (
        <LabView
          data={labRef.current}
          tab={labTab}
          onTab={setLabTab}
          onClose={() => setWs('flight')}
          brain={sc.brain}
          compare={cmp ? { rows: cmp.rows, error: cmp.error } : null}
          compareLoading={Boolean(cmp?.loading)}
          onRunCompare={() => void runCompare()}
          onOverlay={() => void overlayTrajectories()}
          overlayBusy={overlayBusy}
          ablation={ablation}
          ablationBusy={ablationBusy}
          onRunAblation={() => void runAblation()}
          mapData={mapData}
          mapBusy={mapBusy}
          onRunMap={() => void runMap()}
          mapRepeats={mapRepeats}
          onMapRepeats={setMapRepeats}
          mapRetina={mapRetina}
          onMapRetina={setMapRetina}
          mapColor={mapColor}
          onMapColor={setMapColor}
          faults={faults}
          faultsBusy={faultsBusy}
          onRunFaults={() => void runFaults()}
          onCoevTrain={() => void coevTrain()}
          coevTraining={coevTraining}
          coevLadder={coevLadder}
          distill={distill}
          distillBusy={distillBusy}
          onRunDistill={() => void runDistill()}
          onApplyDistill={() => void applyDistill()}
          distillTeacher={distillTeacher}
          onDistillTeacher={setDistillTeacher}
          onDistillSave={() => void distillSave()}
          onDistillLoad={() => void distillLoad()}
          distillSaved={distillSaved}
          transfer={transfer}
          transferBusy={transferBusy}
          transferKind={transferKind}
          onTransferKind={setTransferKind}
          onRunTransfer={() => void runTransfer()}
          scaling={scaling}
          scalingBusy={scalingBusy}
          scalingKind={scalingKind}
          onScalingKind={setScalingKind}
          onRunScaling={() => void runScaling()}
          coev={coev}
          coevBusy={coevBusy}
          onCoevStart={() => void coevStart()}
          onCoevStep={() => void coevStep()}
          onCoevReset={() => void coevReset()}
          swarmExport={swarmExport}
          onExportSwarmCsv={exportSwarmCsv}
          championToBrain={() => void championToBrain()}
          expList={expList}
          onRefreshExperiments={() => void refreshExperiments()}
          onSaveExperimentServer={() => void saveExperimentServer()}
          onLoadExperimentServer={(name) => void loadExperimentServer(name)}
          onDeleteExperimentServer={(name) => void deleteExperimentServer(name)}
          onClearLab={clearLab}
          onMatlabExport={() => exportMatlab()}
          onTrajExport={() => exportTrajectories()}
          onExportNeff={exportNeffCsv}
          rob={robData}
          robBusy={robBusy}
          onRunRobustness={() => void runRobustness()}
          mc={mcData}
          mcBusy={mcBusy}
          mcRuns={mcRuns}
          onMcRuns={setMcRuns}
          onRunMc={() => void runMonteCarlo()}
          cz={czData}
          czBusy={czBusy}
          czGmax={czGmax}
          onCzGmax={setCzGmax}
          czRuns={czRuns}
          onCzRuns={setCzRuns}
          onRunCz={() => void runCaptureZone()}
          tune={{ pool: tunePool, channels: tuneChannels }}
          onTune={(patch) => {
            if (patch.pool !== undefined) setTunePool(patch.pool)
            if (patch.channels !== undefined) setTuneChannels(patch.channels)
          }}
          onRebuild={(pool, channels) => void rebuildBrain(pool, channels)}
          trainCfg={trainCfg}
          onTrainCfg={(patch) => setTrainCfg((c) => ({ ...c, ...patch }))}
          training={training}
          trainProgress={trainProgress}
          onTrain={() => void train()}
          onTrainStop={() => {
            stop.current = true
          }}
          smooth={labCfg.smooth}
          showFact={labCfg.showFact}
          onLabCfg={(patch) => setLabCfg((c) => ({ ...c, ...patch }))}
          egg={egg}
          onEgg={() => setEgg((v) => !v)}
          soundOn={soundOn}
          onSound={setSoundOn}
          voiceKind={voiceKind}
          onVoiceKind={setVoiceKind}
          humorOn={humorOn}
          onHumor={setHumorOn}
          serverOnline={serverOnline}
        />
      )}
      {helpOpen && <HelpView onClose={() => setHelpOpen(false)} />}
      {tourOpen && <Tour onClose={() => setTourOpen(false)} />}
    </AppShell>
  )
}
