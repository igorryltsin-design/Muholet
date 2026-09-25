import { useEffect, useRef, useState } from 'react'
import { EngagementView, type Playback } from '../lazyViews'
import { ASPECT_LABEL, EVADER_RU, LAW_RU, fmt } from './labels'
import type { DuelMatrix, FlyGenome, Scenario } from '../types'
import type { CamMode, Frame } from '../types'

/** Рабочее пространство «Дуэль»: муха-ракета против мухи-самолёта.
 *  Цель уходит реактивными законами (navedenie/evader.py) или обучаемой схемой
 *  («школа уклониста», navedenie/evader_train.py), ракета вырабатывает боевую
 *  жизнь fuse_life_s — не сбила за это время, вердикт у цели. */

/** строки «Ринга»: id сторон ракеты (законы — прямые id, мозги — bio:<kind>) */
export const DUEL_MISSILES = ['pn', 'tpn', 'apn', 'pn_gsn', 'bio:stub', 'bio:full', 'bio:connectome']

const mslLabel = (id: string) => (id.startsWith('bio:') ? `мозг: ${id.slice(4)}` : LAW_RU[id] ?? id)
const evLabel = (id: string) => (id === 'straight' ? 'неманёвренная' : EVADER_RU[id] ?? id)

/** единый id текущей стороны ракеты из сценария */
export function missileIdOf(sc: Scenario): string {
  return sc.mode === 'bio' ? `bio:${sc.brain}` : sc.law
}

/** что учим в одном управляющем цикле: цель, обе стороны вкладкой, обе стороны
 *  стендом (фон переживает закрытие вкладки) */
type LearnMode = 'evader' | 'queen' | 'server'
const LEARN_MODES: { id: LearnMode; label: string; tip: string; max: number }[] = [
  { id: 'evader', label: 'цель', tip: 'Школа уклониста: ракета — фиксированный закон, учится только мозг цели. Поколение ~10–15 с на 12 учениках: бои летят на всех ядрах стенда.', max: 40 },
  { id: 'queen', label: 'обе', tip: 'Красная королева: ракета и цель — обе обучаемые схемы, каждая берёт силу у другой. Поколение ~30–60 с.', max: 30 },
  { id: 'server', label: 'обе на сервере', tip: 'Та же гонка, но считает её стенд в фоновом потоке: вкладку можно закрыть и вернуться к готовности. Поколение ~2 с при pop=4.', max: 60 },
]

/** геометрия боя одним кликом: меняет текущий сценарий, а не дефолт вкладки */
const GEO_PRESETS: { id: Scenario['aspect']; label: string; off: number; tip: string }[] = [
  { id: 'head-on', label: 'лоб', off: 420, tip: 'Встречные: цель идёт почти на ракету, боковое смещение 420 м.' },
  { id: 'beam', label: 'пересечение', off: 900, tip: 'Цель идёт поперёк: большое боковое смещение 900 м, точке встречи надо довернуть.' },
  { id: 'tail-chase', label: 'вдогон', off: 120, tip: 'Погоня сзади: ракета догоняет цель по её же курсу — манёвр цели почти ничего не меняет.' },
]

/** настройки вкладки переживают перезагрузку: слой, режим учёбы и показ боёв */
const DUEL_PREFS = 'muholet-duel'
function readDuelPrefs(): { layer: 'school' | 'duels'; mode: LearnMode; show: boolean } {
  try {
    const d = JSON.parse(localStorage.getItem(DUEL_PREFS) || '') as Record<string, unknown>
    return {
      layer: d.layer === 'duels' ? 'duels' : 'school',
      mode: d.mode === 'server' || d.mode === 'queen' ? (d.mode as LearnMode) : 'evader',
      show: d.show === true,
    }
  } catch {
    return { layer: 'school', mode: 'evader', show: false }
  }
}

/** бой, который стенд досчитывает специально для сцены (флаг replay в ответе
 *  поколения): траектории обеих сторон + момент, когда цель начала маневрировать */
export type DuelReplay = {
  label: string
  traj_m: number[][]
  traj_t: number[][]
  hit: boolean
  t_end: number
  cpa_m: number
  missile_n_int: number
  n_target_g: number
  turn: { t_s: number; range_m: number } | null
  scenario: { aspect: string; range_m: number; v_t: number; off_axis_m: number }
  fitness?: number
  missile_fitness?: number
  evader_fitness?: number
}

/** снимок серверного самообучения (navedenie/queen_train.py): фон стенда крутит
 *  цикл поколений, вкладку можно закрыть — форма ответа постоянна и до задачи */
type QueenTrainStatus = {
  running: boolean
  generations: number
  generations_done: number
  pop: number
  seed: number
  log: {
    gen: number
    p_hit_ring: number
    exam_p_hit: number
    missile_best: number
    evader_best: number
    t_survived_median: number
    cpa_m_median: number
    seconds: number
  }[]
  champions: { missile: FlyGenome; evader: FlyGenome } | null
  saved: {
    ring: { id: number; label: string } | null
    weights: unknown
    ring_battle?: { duels: { id: number; label: string }[]; standings: { id: number; label: string; rank: number; score: number; p_attack: number; p_defense: number }[] } | null
  }
  inherit: number[]
  auto_ring?: boolean
  error: string | null
  stop_requested: boolean
  seconds: number
  /** бой последнего досчитанного поколения — только пока задача жива (см. status()) */
  replay?: DuelReplay | null
}

/** «Хроника войн» (navedenie/queen_chronicle.py): доигранная кампания с кривой
 *  взятий — единственная память о самообучении, переживающая рестарт стенда */
type QueenCampaign = {
  id: number
  finished_at: number
  label: string
  generations: number
  planned: number
  pop: number
  seed: number
  seconds: number
  scenario: { aspect?: string }
  curve: { gen: number; p_hit_ring: number; exam_p_hit: number; missile_best: number; evader_best: number }[]
  p_hit_first: number | null
  p_hit_last: number | null
  ring_id: number | null
  shelved: boolean
  applied: boolean
  inherited?: number[]
  stopped: boolean
  error: string | null
}

/** кривая доли взятий в одну строку знаков — рост силы виден без графика */
const SPARK = '▁▂▄▆█'
const spark = (xs: number[] | undefined | null) =>
  (xs ?? [])
    .map((v) => SPARK[Math.max(0, Math.min(SPARK.length - 1, Math.round((Number.isFinite(v) ? v : 0) * (SPARK.length - 1))))])
    .join('')

/** «Ринг чемпионов»: строчка таблицы рангов и сводка пары (атакующий→обороняющийся) */
type RingStanding = {
  rank: number
  id: number
  label: string
  attack_hits: number
  attack_n: number
  defense_saves: number
  defense_n: number
  p_attack: number
  p_defense: number
  score: number
}
type RingCell = { attacker: number; defender: number; p_hit: number; t_survived_median: number }
/** форма полки: кого последний сводный бой считает сильнейшими */
type RingFormRow = { id: number; label: string; rank: number; score: number; p_attack: number; p_defense: number; move?: number | null }
type FormCell = { attacker: number; defender: number; p_hit: number }
type RingSeason = { at: number; champ_id: number; champ_label: string; order: number[] }
type RingForm = { basis: 'форма' | 'свежесть' | 'пусто'; computed_at: number | null; rows: RingFormRow[]; ids: number[]; cells: FormCell[]; seasons: RingSeason[] }

export function DuelWorkspace({
  frame,
  sc,
  set,
  playback,
  day,
  camMode,
  onCamMode,
  geometryOn,
  onToggleGeometry,
  sceneBadge,
  silent,
  busy,
  done,
  duelVerdict,
  shtrumCaption,
  wendyCaption,
  duelMatrix,
  duelBusy,
  duelRepeats,
  onDuelRepeats,
  onRunMatrix,
  onExportCsv,
  onDuelNow,
  onShowReplay,
}: {
  frame: Frame | null
  sc: Scenario
  set: <K extends keyof Scenario>(key: K, value: Scenario[K]) => void
  playback: Playback | null
  day: boolean
  camMode: CamMode
  onCamMode: (m: CamMode) => void
  geometryOn: boolean
  onToggleGeometry: () => void
  sceneBadge: string | null
  silent: boolean
  busy: boolean
  done: string | null
  duelVerdict: { result: 'missile' | 'evader' | null; tSurvived: number | null; fuse: boolean } | null
  /** юмор-режим: субтитры напарницы-штурмана и «Кобры» — мухи-пилота цели */
  shtrumCaption: { text: string; he: boolean } | null
  wendyCaption: { text: string } | null
  duelMatrix: DuelMatrix | null
  duelBusy: boolean
  duelRepeats: number
  onDuelRepeats: (n: number) => void
  onRunMatrix: () => void
  onExportCsv: () => void
  /** запуск прогона «прямо сейчас» с надбавкой к сценарию (не ждёт setState) */
  onDuelNow: (patch: Partial<Scenario>) => void
  /** выпустить учебный бой поколения на сцену: проигрывание траекторий вместо цифр */
  onShowReplay: (r: DuelReplay) => void
}) {
  const cellOf = (row: string, col: string) => duelMatrix?.cells.find((c) => c.row === row && c.col === col) ?? null
  const wins = duelMatrix?.cells.filter((c) => c.win === 'missile').length ?? 0
  const t = frame?.t ?? 0
  const over = duelVerdict ? !busy : t > sc.fuse_life_s + 1e-9

  // ── показ учёбы: стенд досылает бой поколения, он уходит на сцену и в память
  // вкладки (из него же считается честный ответ, когда цель начала крутить)
  const [lesson, setLesson] = useState<DuelReplay | null>(null)
  const busyRef = useRef(busy)
  useEffect(() => {
    busyRef.current = busy
  }, [busy])

  const showLesson = (r?: DuelReplay | null) => {
    if (!r || !(r.traj_m?.length > 0)) return
    // живой прогон не перекрываем: сцена и так играет бой, а учебный кадрился бы поверх
    if (busyRef.current) return
    setLesson(r)
    onShowReplay(r)
  }

  // ── школа уклониста: цикл поколений живёт здесь, стенд отдаёт по одной генерации
  const [schoolBusy, setSchoolBusy] = useState(false)
  const [schoolGens, setSchoolGens] = useState(6)
  const [schoolLog, setSchoolLog] = useState<{ gen: number; best: number; survive: number; t: number }[]>([])
  const [champ, setChamp] = useState<FlyGenome | null>(null)
  const [schoolMsg, setSchoolMsg] = useState<string | null>(null)
  const schoolStop = useRef(false)
  const popRef = useRef<FlyGenome[]>([])

  const runSchool = async () => {
    if (schoolBusy) return
    setSchoolBusy(true)
    setSchoolMsg(null)
    setSchoolLog([])
    schoolStop.current = false
    popRef.current = []
    try {
      for (let g = 0; g < schoolGens && !schoolStop.current; g++) {
        const res = await fetch('/api/evader/gen', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ scenario: sc, population: popRef.current, gen: g, seed: 7, exam_every: 5, replay: true }),
        })
        if (!res.ok) throw new Error(`стенд ${res.status}`)
        const d = (await res.json()) as {
          stats: { best: number; best_idx: number; survive_rate: number; best_t_survived: number }
          results: { fly: FlyGenome }[]
          next_population: FlyGenome[]
          replay?: DuelReplay
        }
        popRef.current = d.next_population
        setChamp(d.results[d.stats.best_idx]?.fly ?? null)
        setSchoolLog((rows) => [...rows, { gen: g, best: d.stats.best, survive: d.stats.survive_rate, t: d.stats.best_t_survived }])
        showLesson(d.replay)
      }
    } catch {
      setSchoolMsg('Стенд не отвечает — школа уклониста считается только на сервере.')
    } finally {
      setSchoolBusy(false)
    }
  }

  const applyChamp = async () => {
    if (!champ) return
    try {
      const res = await fetch('/api/evader/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ w: champ.w, gain: champ.gain }),
      })
      if (!res.ok) throw new Error()
      set('duel', true)
      set('evader_law', 'brain')
      if (sc.n_target <= 0) set('n_target', 8)
      setSchoolMsg('Чемпион применён: цель теперь летает на выученных весах (data/weights_evader.npz).')
    } catch {
      setSchoolMsg('Не удалось применить веса: стенд недоступен.')
    }
  }

  // ── «Красная королева»: обе стороны — мозги, каждый отбор против ВСЕХ живых оппонентов
  const [queenBusy, setQueenBusy] = useState(false)
  const [queenGens, setQueenGens] = useState(4)
  const [queenLog, setQueenLog] = useState<{ gen: number; ring: number; exam: number; m: number; e: number; t: number }[]>([])
  const [queenChamps, setQueenChamps] = useState<{ missile: FlyGenome; evader: FlyGenome } | null>(null)
  const [queenMsg, setQueenMsg] = useState<string | null>(null)
  const queenStop = useRef(false)
  const mPopRef = useRef<FlyGenome[]>([])
  const ePopRef = useRef<FlyGenome[]>([])

  const runQueen = async () => {
    if (queenBusy) return
    setQueenBusy(true)
    setQueenMsg(null)
    setQueenLog([])
    queenStop.current = false
    mPopRef.current = []
    ePopRef.current = []
    try {
      for (let g = 0; g < queenGens && !queenStop.current; g++) {
        const res = await fetch('/api/queen/gen', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ scenario: sc, missile_population: mPopRef.current, evader_population: ePopRef.current, gen: g, seed: 7, replay: true }),
        })
        if (!res.ok) throw new Error(`стенд ${res.status}`)
        const d = (await res.json()) as {
          missile_population: FlyGenome[]
          evader_population: FlyGenome[]
          champions: { missile: FlyGenome; evader: FlyGenome }
          stats: { p_hit_ring: number; missile_best: number; evader_best: number }
          exam: { p_hit: number; t_survived_median: number }
          replay?: DuelReplay
        }
        mPopRef.current = d.missile_population
        ePopRef.current = d.evader_population
        setQueenChamps(d.champions)
        setQueenLog((rows) => [...rows, { gen: g, ring: d.stats.p_hit_ring, exam: d.exam.p_hit, m: d.stats.missile_best, e: d.stats.evader_best, t: d.exam.t_survived_median }])
        showLesson(d.replay)
      }
    } catch {
      setQueenMsg('Стенд не отвечает — королева воюет только на сервере.')
    } finally {
      setQueenBusy(false)
    }
  }

  const applyGains = useRef<{ missile?: number; evader?: number }>({})

  const applyQueens = async (): Promise<boolean> => {
    if (!queenChamps) return false
    try {
      const res = await fetch('/api/queen/apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ missile: { w: queenChamps.missile.w, gain: queenChamps.missile.gain }, evader: { w: queenChamps.evader.w, gain: queenChamps.evader.gain } }),
      })
      if (!res.ok) throw new Error()
      const d = (await res.json().catch(() => null)) as { gains?: { missile?: number; evader?: number } } | null
      applyGains.current = d?.gains ?? {}
      set('mode', 'bio')
      set('brain', 'stub')
      set('duel', true)
      set('evader_law', 'brain')
      if (sc.n_target <= 0) set('n_target', 8)
      // живая ракета берёт усиление из сценария — без этой строки в бой летит
      // чемпион с чужими руками, не тем, с каким он выиграл гонку
      if (applyGains.current.missile !== undefined) set('circuit_gain', applyGains.current.missile)
      setQueenMsg('Чемпионы посажены: ракета — мозг stub, цель — мозг-уклонист. Жми «Пуск».')
      return true
    } catch {
      setQueenMsg('Не удалось применить веса: стенд недоступен.')
      return false
    }
  }

  // один клик: посадить чемпионов И сразу выпустить их в живой бой — патчем,
  // не дожидаясь, пока setState дойдёт до сценария (иначе полетели бы старые стороны)
  const duelChampions = async () => {
    if (!(await applyQueens())) return
    onDuelNow({
      mode: 'bio',
      brain: 'stub',
      duel: true,
      evader_law: 'brain',
      ...(sc.n_target <= 0 ? { n_target: 8 } : {}),
      ...(applyGains.current.missile === undefined ? {} : { circuit_gain: applyGains.current.missile }),
    })
  }

  // ── «Ринг чемпионов»: полка дуэтов + круговой самобой (ранг = атака + оборона)
  const [ringDuels, setRingDuels] = useState<{ id: number; label: string }[]>([])
  const [ringBusy, setRingBusy] = useState(false)
  const [ringLabel, setRingLabel] = useState('')
  const [ringStand, setRingStand] = useState<RingStanding[] | null>(null)
  const [ringCells, setRingCells] = useState<RingCell[] | null>(null)
  // форма полки — память о последнем сводном бое: наследовать сильнейших, а не последних вставших
  const [ringForm, setRingForm] = useState<RingForm | null>(null)

  const refreshForm = async () => {
    try {
      const r = await fetch('/api/queen/ring/form?limit=2')
      if (!r.ok) throw new Error()
      const d = (await r.json()) as RingForm
      setRingForm({ basis: d.basis ?? 'свежесть', computed_at: d.computed_at ?? null, rows: d.rows ?? [], ids: d.ids ?? [], cells: d.cells ?? [], seasons: d.seasons ?? [] })
    } catch {
      setRingForm(null)
    }
  }

  const refreshRing = async () => {
    try {
      const r = await fetch('/api/queen/ring')
      const d = (await r.json()) as { duels: { id: number; label: string; missile: FlyGenome; evader: FlyGenome }[] }
      const list = (d.duels ?? []).map(({ id, label }) => ({ id, label }))
      setRingDuels(list)
      // полка изменилась (✕, новый дуэт, потолок) — и форма обязана пересчитаться
      void refreshForm()
      return list
    } catch {
      return ringDuels
    }
  }
  useEffect(() => {
    void refreshRing()
  }, [])

  const shelfDuel = async () => {
    if (!queenChamps) return
    try {
      const res = await fetch('/api/queen/ring/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label: ringLabel, missile: { w: queenChamps.missile.w, gain: queenChamps.missile.gain }, evader: { w: queenChamps.evader.w, gain: queenChamps.evader.gain } }),
      })
      if (!res.ok) throw new Error()
      setRingLabel('')
      await refreshRing()
      setQueenMsg('Дуэт чемпионов поставлен на полку ринга.')
    } catch {
      setQueenMsg('Полка не приняла дуэт: стенд недоступен или веса кривые.')
    }
  }

  const dropDuel = async (id: number) => {
    try {
      await fetch('/api/queen/ring/delete', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id }) })
      await refreshRing()
    } catch {
      /* сервер недоступен */
    }
  }

  const fightRing = async () => {
    if (ringBusy) return
    setRingBusy(true)
    try {
      // полку освешаем перед боем: дуэты могли поставить в другом окне или
      // после рестарта стенда — иначе кнопка дерёт устаревшее состояние
      const list = await refreshRing()
      if (list.length < 2) throw new Error('мало дуэтов')
      const res = await fetch('/api/queen/ring/battle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario: sc, ids: [] }),
      })
      if (!res.ok) throw new Error(`стенд ${res.status}`)
      const d = (await res.json()) as { standings: RingStanding[]; cells: RingCell[] }
      setRingStand(d.standings)
      setRingCells(d.cells)
      // сводный бой записал форму полки — освежить её сразу: refreshRing в начале
      // боя тянул снимок ДО этого боя, т.е. прошлую расстановку сил
      await refreshForm()
      setQueenMsg(null)
    } catch {
      setQueenMsg('Ринг не свёл: на полке должно стоять минимум два дуэта (стенд жив?).')
    } finally {
      setRingBusy(false)
    }
  }

  // ── вызов с полки: сборная пара (ракета одного дуэта, уклонист другого) — в живой бой
  const [pairAtt, setPairAtt] = useState(0)
  const [pairDef, setPairDef] = useState(0)
  const [pairBusy, setPairBusy] = useState(false)
  const [pairMsg, setPairMsg] = useState<string | null>(null)

  // id с устаревшей полки не держим: после ✕ или перезагрузки select сам
  // возвращается на первого доступного, а не шлёт на стенд несуществующий дуэт
  const attId = ringDuels.some((d) => d.id === pairAtt) ? pairAtt : ringDuels[0]?.id ?? 0
  const defId = ringDuels.some((d) => d.id === pairDef) ? pairDef : ringDuels[1]?.id ?? ringDuels[0]?.id ?? 0

  // пару можно передать явно — так кликабельная ячейка матрицы сажает на штурвалы
  // ровно тех, кого показывает, не дожидаясь, пока списки догонят состояние
  const duelPair = async (att?: number, def?: number) => {
    const a = att ?? attId
    const d = def ?? defId
    if (pairBusy || !a || !d) return
    setPairBusy(true)
    setPairMsg(null)
    try {
      const res = await fetch('/api/queen/ring/pair', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ attacker: a, defender: d }),
      })
      if (!res.ok) {
        const j = await res.json().catch(() => null)
        setPairMsg(`Пара не села: ${typeof j?.detail === 'string' ? j.detail : `стенд ${res.status}`}`)
        return
      }
      const j = (await res.json().catch(() => null)) as { gains?: { missile?: number; evader?: number } } | null
      applyGains.current = j?.gains ?? {}
      // тот же обход гонки setState, что у «Дуэли чемпионов»: летим патчем сразу
      onDuelNow({
        mode: 'bio',
        brain: 'stub',
        duel: true,
        evader_law: 'brain',
        ...(sc.n_target <= 0 ? { n_target: 8 } : {}),
        ...(applyGains.current.missile === undefined ? {} : { circuit_gain: applyGains.current.missile }),
      })
    } catch {
      setPairMsg('Стенд не отвечает — сборная пара сажается на нём, а не в браузере.')
    } finally {
      setPairBusy(false)
    }
  }

  // ── самообучение на сервере: цикл поколений живёт в фоне стенда, вкладку можно закрыть
  const [srv, setSrv] = useState<QueenTrainStatus | null>(null)
  const [srvGens, setSrvGens] = useState(8)
  const [srvMsg, setSrvMsg] = useState<string | null>(null)
  // наследие: война начинается с выученных мозгов полки, а не с врождённого рефлекса
  const [srvHeirs, setSrvHeirs] = useState(false)
  const [srvRing, setSrvRing] = useState(false)
  // родители — сильные: форма последнего сводного боя, а не очередь сохранения.
  // без снимка (бой ещё не сводился) остаёмся честными: берём самых свежих
  const heirIds = (ringForm?.ids.length ? ringForm.ids : [...ringDuels].map((d) => d.id).reverse()).slice(0, 2)
  const heirNames = heirIds.map((id) => ringForm?.rows.find((r) => r.id === id)?.label ?? ringDuels.find((d) => d.id === id)?.label ?? `дуэт ${id}`).join(' · ')
  const heirByRank = ringForm?.basis === 'форма'
  // движение после последнего свода: сверяем ряд таблицы с формой — бой её перезаписал
  const ringMoveTag = (id: number) => {
    const mv = ringForm?.rows.find((r) => r.id === id)?.move
    return mv === undefined || mv === null ? '' : mv > 0 ? ` ▲${mv}` : mv < 0 ? ` ▼${-mv}` : ' ±0'
  }
  const movePhrase = (r: RingFormRow) =>
    r.move === null || r.move === undefined
      ? ' (первый свод для него)'
      : r.move > 0
        ? ` (поднялся на ${r.move})`
        : r.move < 0
          ? ` (сел на ${-r.move})`
          : ' (ранг не сдвинулся)'
  // «кто кого бьёт» из памяти последнего свода: ячейки пар собираются в квадратную
  // матрицу и живут до следующего боя — даже если ринг в этой сессии не сводили руками
  const formCells = ringForm?.cells ?? []
  const formIds = [...new Set(formCells.flatMap((c) => [c.attacker, c.defender]))].sort((a, b) => a - b)
  const formP = new Map(formCells.map((c) => [`${c.attacker}|${c.defender}`, c.p_hit]))
  const ringName = (id: number) => ringDuels.find((d) => d.id === id)?.label ?? `дуэт ${id}`
  // чемпионская лента: исход каждого прошлого свода; метка — у нынешнего лидера
  const seasons = ringForm?.seasons ?? []
  const reigns = seasons.slice(-12)
  // медальный зачёт: места копятся за ВСЕ своды памяти ленты, а не за последний круг
  const medalGold = (n: number) =>
    n % 10 === 1 && n % 100 !== 11 ? 'титул' : n % 10 >= 2 && n % 10 <= 4 && !(n % 100 >= 12 && n % 100 <= 14) ? 'титула' : 'титулов'
  const medalTop = (() => {
    const byId = new Map<number, { pts: number; gold: number }>()
    for (const s of seasons)
      s.order.slice(0, 3).forEach((id, i) => {
        const m = byId.get(id) ?? { pts: 0, gold: 0 }
        m.pts += 3 - i
        if (i === 0) m.gold += 1
        byId.set(id, m)
      })
    const names = new Map<number, string>()
    for (const s of seasons) if (s.champ_label) names.set(s.champ_id, s.champ_label)
    return [...byId.entries()]
      .map(([id, m]) => ({ id, name: names.get(id) ?? ringName(id), ...m }))
      .sort((a, b) => b.pts - a.pts || b.gold - a.gold || a.id - b.id)
      .slice(0, 5)
  })()
  // прогноз на ревانش: та пара уже дрелась в последнем круге — говорим честно из памяти, а не гаданием
  const lastDuel = formCells.find((c) => c.attacker === attId && c.defender === defId) ?? null
  const duelPrediction = (c: FormCell) => {
    const p = Math.round(c.p_hit * 100)
    return p >= 100
      ? `прошлый круг: ${ringName(c.attacker)} брал ${ringName(c.defender)} всухую`
      : p === 0
        ? `прошлый круг: ${ringName(c.defender)} отбивался от ${ringName(c.attacker)} всухую`
        : `прошлый круг: ${ringName(c.attacker)} брал ${ringName(c.defender)} в ${p}% схваток`
  }
  const srvTimer = useRef<number | null>(null)
  const srvSettled = useRef(false)
  // показ поколения из фона: один бой на каждое досчитанное поколение, а не
  // перезапуск одной и той же анимации на каждом опросе раз в 2 с
  const srvShown = useRef('')

  const srvPollStop = () => {
    if (srvTimer.current !== null) window.clearTimeout(srvTimer.current)
    srvTimer.current = null
  }

  const srvApply = (d: QueenTrainStatus) => {
    setSrv(d)
    if (d.champions) setQueenChamps(d.champions)
    // фоновая кампания показывает бой поколения только по тумблеру: цифры идут в
    // любом случае, а сцена по умолчанию не дёргается — вручную учатся видно всегда
    if (showLessons && d.replay && d.replay.label !== srvShown.current) {
      srvShown.current = d.replay.label
      showLesson(d.replay)
    }
    if (d.running) {
      srvSettled.current = false
    } else if (!srvSettled.current && d.generations_done > 0) {
      // готовность одна: финальные действия (полка, сообщение) — ровно один раз,
      // иначе каждый опрос будет заново дерать ring и вписывать реплику
      srvSettled.current = true
      void refreshRing()
      void refreshChron()
      setSrvMsg(
        d.error
          ? `Самообучение споткнулось: ${d.error}`
          : `Готово: ${d.generations_done} поколений на сервере. Чемпионы посажены и поставлены на ринг.`,
      )
    }
    srvPollStop()
    if (d.running) srvTimer.current = window.setTimeout(() => void srvPoll(), 2000)
  }

  const srvPoll = async () => {
    try {
      const res = await fetch('/api/queen/train')
      if (!res.ok) return srvPollStop()
      srvApply((await res.json()) as QueenTrainStatus)
    } catch {
      srvPollStop()
    }
  }

  const srvStart = async () => {
    setSrvMsg(null)
    srvSettled.current = false
    srvShown.current = ''
    try {
      const res = await fetch('/api/queen/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario: sc,
          generations: srvGens,
          pop: 4,
          seed: 7,
          save_duel: true,
          apply: true,
          inherit: srvHeirs ? heirIds : [],
          auto_ring: srvRing,
        }),
      })
      if (res.status === 409) {
        // воюет не моя задача — просто подхватываю её прогресс вместо второй гонки
        setSrvMsg('На стенде уже воюет королева — подхватываю её прогресс.')
        await srvPoll()
        return
      }
      if (!res.ok) {
        const d = await res.json().catch(() => null)
        setSrvMsg(`Война не началась: ${typeof d?.detail === 'string' ? d.detail : `стенд ${res.status}`}`)
        return
      }
      srvApply((await res.json()) as QueenTrainStatus)
    } catch {
      setSrvMsg('Стенд не отвечает — самообучение живёт только на сервере.')
    }
  }

  const srvStop = async () => {
    try {
      const res = await fetch('/api/queen/train/stop', { method: 'POST' })
      if (res.ok) srvApply((await res.json()) as QueenTrainStatus)
    } catch {
      /* сервер недоступен */
    }
  }

  // ── хроника войн: память о доигранных кампаниях, живущая на сервере
  const [chron, setChron] = useState<QueenCampaign[]>([])

  const refreshChron = async () => {
    try {
      const res = await fetch('/api/queen/chronicle')
      if (!res.ok) return
      const d = (await res.json()) as { campaigns: QueenCampaign[] }
      setChron(d.campaigns)
    } catch {
      /* сервер недоступен */
    }
  }

  const dropChron = async (id: number) => {
    try {
      const res = await fetch('/api/queen/chronicle/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id }),
      })
      if (!res.ok) return
      const d = (await res.json()) as { campaigns: QueenCampaign[] }
      setChron(d.campaigns)
    } catch {
      /* сервер недоступен */
    }
  }

  // ── два слоя вместо восьми карточек: сначала смотреть, как учатся две мухи,
  // потом сводить дуэли. Настройки читаются синхронно в инициализаторе useState:
  // effect-restore при двойном маунте StrictMode затирал бы сохранённое дефолтом
  const [layer, setLayer] = useState<'school' | 'duels'>(() => readDuelPrefs().layer)
  const [learnMode, setLearnMode] = useState<LearnMode>(() => readDuelPrefs().mode)
  const [showLessons, setShowLessons] = useState(() => readDuelPrefs().show)
  useEffect(() => {
    try {
      localStorage.setItem(DUEL_PREFS, JSON.stringify({ layer, mode: learnMode, show: showLessons }))
    } catch {
      /* приватный режим — настройка проживёт до перезагрузки */
    }
  }, [layer, learnMode, showLessons])

  // один цикл поколений на троих: поле и пуск общие, считается выбранный режим
  const gens = learnMode === 'evader' ? schoolGens : learnMode === 'queen' ? queenGens : srvGens
  const setGens = (n: number) => {
    const max = LEARN_MODES.find((m) => m.id === learnMode)?.max ?? 40
    const v = Math.max(1, Math.min(max, Math.round(n) || 1))
    if (learnMode === 'evader') setSchoolGens(v)
    else if (learnMode === 'queen') setQueenGens(v)
    else setSrvGens(v)
  }
  const learning = schoolBusy || queenBusy || Boolean(srv?.running)
  const startLabel =
    learnMode === 'evader' ? (schoolBusy ? 'школа учит…' : 'Учить') : learnMode === 'queen' ? (queenBusy ? 'война…' : 'Воевать') : srv?.running ? 'воюет в фоне…' : 'Воевать в фоне'
  const startLearning = () => {
    if (learnMode === 'evader') void runSchool()
    else if (learnMode === 'queen') void runQueen()
    else void srvStart()
  }
  const stopLearning = () => {
    if (learnMode === 'evader') schoolStop.current = true
    else if (learnMode === 'queen') queenStop.current = true
    else void srvStop()
  }
  /** выплата за учёбу: чемпионы последней гонки нужны обоим режимам коэволюции —
   *  и ручному («Красная королева»), и фоновому («Самообучение на сервере») */
  const champBlock = (
    <div className="stat-grid">
      <span data-tip="Чемпионы последнего поколения: обе стороны сажают свои выученные веса.">
        чемпионы · усиление <b>{fmt(queenChamps?.missile.gain, 2)}</b> / <b>{fmt(queenChamps?.evader.gain, 2)}</b>
      </span>
      <button type="button" onClick={applyQueens} data-tip="Ракету — в живой мозг stub (mode=bio), цель — в weights_evader.npz; дальше «Пуск» летает их между собой.">
        Применить чемпионов
      </button>
      <button type="button" onClick={duelChampions} data-tip="Один клик: посадить обоих чемпионов и сразу выпустить живой бой — стрим двух мозгов по websocket, без ожидания «Пуск».">
        Дуэль чемпионов
      </button>
      <input
        type="text"
        value={ringLabel}
        maxLength={40}
        placeholder="имя дуэта"
        onChange={(e) => setRingLabel(e.target.value)}
        data-tip="Имя для полки ринга; пусто — дуэт получит номер."
      />
      <button type="button" onClick={shelfDuel} data-tip="Записать пару выученных мозгов на полку «Ринга чемпионов»: дальше они стреляют друг в друга и против других дуэтов, веса переживают перезапуск.">
        На полку ринга
      </button>
    </div>
  )

  // на входе в космос проверяем фон: королева могла начать войну в прошлой сессии
  useEffect(() => {
    void srvPoll()
    void refreshChron()
    return srvPollStop
  }, [])

  return (
    <div className="twocol-ws">
      <div className="ws-main">
        <EngagementView
          frame={frame}
          scenario={sc}
          playback={playback}
          day={day}
          camMode={camMode}
          geometryOn={geometryOn}
          sceneIdle={false}
          sceneBadge={sceneBadge}
          cockpit={false}
          silent={silent}
        />
        {shtrumCaption && (
          <div className="shtrum-caption">
            <b>{shtrumCaption.he ? '♂ Штруман' : '♀ Штрумана'}</b> {shtrumCaption.text}
          </div>
        )}
        {wendyCaption && (
          <div className="shtrum-caption shtrum-caption--wendy">
            <b>♀ Кобра</b> {wendyCaption.text}
          </div>
        )}
        <div className="scene-overlay scene-overlay--topleft">
          <span className="chip is-warn">дуэль · {sc.duel ? evLabel(sc.evader_law) : 'цель неманёвренная'}</span>
          <span
            className={`chip ${busy ? (over ? 'is-bad' : '') : duelVerdict ? (duelVerdict.result === 'missile' ? 'is-ok' : 'is-bad') : ''}`}
            data-tip="Боевая жизнь ракеты: не сбила до обнуления — вердикт у цели."
          >
            {busy ? `время ${fmt(Math.min(t, sc.fuse_life_s), 1)} / ${fmt(sc.fuse_life_s, 0)} с` : duelVerdict ? (duelVerdict.result === 'missile' ? `ракета взяла · ${fmt(duelVerdict.tSurvived, 1)} с` : duelVerdict.fuse ? `цель пережила · все ${fmt(sc.fuse_life_s, 0)} с` : `цель ушла · ${fmt(duelVerdict.tSurvived, 1)} с`) : `боевая жизнь ${fmt(sc.fuse_life_s, 0)} с`}
          </span>
          {busy && t >= sc.fuse_life_s - 1e-9 && <span className="chip is-bad">время вышло</span>}
        </div>
        {done && (
          <div className="scene-overlay scene-overlay--bottomleft">
            <span className={`chip ${done.startsWith('Ракета') ? 'is-ok' : done.startsWith('Цель') ? 'is-bad' : ''}`}>{done}</span>
            {!busy && (
              <button type="button" className="chip" onClick={() => onDuelNow({})} data-tip="Тот же сценарий, те же веса, тот же seed: дуэль детерминирована, бой повторится кадр в кадр.">
                повторить бой
              </button>
            )}
          </div>
        )}
      </div>

      <div className="ws-side">
        <h2 className="ws-side__title">Дуэль мух</h2>

        {/* два слоя вместо восьми карточек подряд: сначала смотреть, как учатся
            две мухи, потом сводить дуэли. «Стороны» — над слоями: и закон ракеты,
            и включённый манёвр цели нужны обеим половинам вкладки */}
        <div className="ws-tabs">
          <button type="button" className={layer === 'school' ? 'is-on' : undefined} onClick={() => setLayer('school')} data-tip="Учёба: цикл поколений и живой показ того, как мухи учатся друг против друга.">
            Учёба
          </button>
          <button type="button" className={layer === 'duels' ? 'is-on' : undefined} onClick={() => setLayer('duels')} data-tip="Дуэли: готовые стороны, ринг чемпионов и матрица «закон против закона».">
            Дуэли
          </button>
        </div>

        <section className="ws-card">
          <h3>Стороны</h3>
          <div className="fields fields--2" style={{ padding: 0 }}>
            <label data-tip="Чем управляет ракета: закон наведения (точная геометрия или сенсорный) либо био-мозг.">
              ракета
              <select
                value={missileIdOf(sc)}
                onChange={(e) => {
                  const v = e.target.value
                  if (v.startsWith('bio:')) {
                    set('mode', 'bio')
                    set('brain', v.slice(4) as Scenario['brain'])
                  } else {
                    set('mode', 'pn')
                    set('law', v as Scenario['law'])
                  }
                }}
              >
                {DUEL_MISSILES.map((id) => (
                  <option key={id} value={id}>
                    {mslLabel(id)}
                  </option>
                ))}
              </select>
            </label>
            <label data-tip="Как уходит цель: видит ракету (положение и скорость), но не её ускорение. Неманёвренная — базовая линия.">
              цель
              <select
                value={sc.duel ? sc.evader_law : 'straight'}
                onChange={(e) => {
                  const v = e.target.value
                  if (v === 'straight') set('duel', false)
                  else {
                    set('duel', true)
                    set('evader_law', v as Scenario['evader_law'])
                    if (sc.n_target <= 0) set('n_target', 8) // уклонисту нужна перегрузка, иначе он неманёвренный по определению
                  }
                }}
              >
                <option value="straight">неманёвренная</option>
                {(Object.keys(EVADER_RU) as Scenario['evader_law'][]).map((id) => (
                  <option key={id} value={id}>
                    {EVADER_RU[id]}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <p className="lab-hint" style={{ margin: 0 }}>
            цель «видит» ракету как приёмник облучения: положение и скорость — да, ускорение и будущее — нет
          </p>
        </section>

        {layer === 'school' && (
          <>
            <section className="ws-card">
              <h3>Две мухи учатся</h3>
              <div className="fields fields--2" style={{ padding: 0 }}>
                <label data-tip="Один цикл поколений на выбор: учится только цель, учатся обе стороны на вкладке, или ту же гонку считает стенд в фоне.">
                  кто учится
                  <select value={learnMode} onChange={(e) => setLearnMode(e.target.value as LearnMode)}>
                    {LEARN_MODES.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label data-tip={LEARN_MODES.find((m) => m.id === learnMode)?.tip ?? ''}>
                  поколений
                  <input type="number" min={1} max={LEARN_MODES.find((m) => m.id === learnMode)?.max ?? 40} value={gens} onChange={(e) => setGens(Number(e.target.value))} />
                </label>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button type="button" className="primary" data-tip="Прогнать цикл: каждое поколение сразу играется на сцене, если идёт показ боя." disabled={learnMode !== 'server' && (learning || busy)} onClick={startLearning}>
                  {startLabel}
                </button>
                {learning && (
                  <button type="button" data-tip="Цикл сворачивается после текущего поколения; выученное остаётся." onClick={stopLearning}>
                    Стоп
                  </button>
                )}
              </div>
              <div className="duel-geo">
                <span className="chip" data-tip="Геометрия боя, на которой учится поколение: она же задаёт, есть ли у цели смысл маневрировать.">
                  {ASPECT_LABEL[sc.aspect]} · смещение {fmt(sc.off_axis_m, 0)} м · дальность {fmt(sc.range_m, 0)} м
                </span>
                {GEO_PRESETS.map((p) => (
                  <button
                    key={p.id}
                    type="button"
                    className={`chip ${sc.aspect === p.id ? 'is-active' : ''}`}
                    data-tip={p.tip}
                    onClick={() => {
                      set('aspect', p.id)
                      set('off_axis_m', p.off)
                    }}
                  >
                    {p.label}
                  </button>
                ))}
                <label className="chip duel-geo__num" data-tip="Располагаемая перегрузка цели, g: больше — круче разворот, 0 — уклоняться нечем.">
                  перегрузка цели, g
                  <input type="number" min={0} max={30} step={0.5} value={sc.n_target} onChange={(e) => set('n_target', Math.max(0, Number(e.target.value) || 0))} />
                </label>
              </div>
              {(!sc.duel || sc.n_target <= 0) && (
                <p className="lab-hint" style={{ margin: 0 }}>
                  <span className="chip is-warn">{!sc.duel ? 'дуэль выключена: цель идёт по прямой и не поворачивает вовсе' : 'перегрузка цели 0 g → уклонист неманёвренный по определению'}</span>
                </p>
              )}
              {learnMode === 'server' && (
                <label className="duel-geo__opt" data-tip="Цифры поколений приходят в любом случае. С показом стенд досчитывает один бой последнего поколения и играет его на сцене — вручную («Учить», «Воевать») учёба показывается всегда.">
                  показывать бои поколений
                  <input type="checkbox" checked={showLessons} onChange={(e) => setShowLessons(e.target.checked)} />
                </label>
              )}
              {lesson && (
                <p className="lab-hint" style={{ margin: 0 }} data-tip="Промежуток вдали цель не крутит по устройству: её признаки масштабируются близостью (θ·4, ρ·0.4), поэтому реакция появляется только рядом с ракетой. «Нырок + поворот в конце» — законный оптимум фитнеса «выжить», а не поломка.">
                  {lesson.label}: {lesson.hit ? 'цель сбита' : `цель ушла · наименьшее сближение ${fmt(lesson.cpa_m, 0)} м`} ·{' '}
                  {lesson.turn ? `манёвр начался: t = ${fmt(lesson.turn.t_s, 1)} с, до цели ${fmt(lesson.turn.range_m, 0)} м` : lesson.n_target_g > 0 ? 'за весь бой цель не повернула' : 'повернуть было нечем: перегрузка цели 0 g'}
                </p>
              )}
            </section>

            {learnMode === 'evader' && (
              <section className="ws-card">
                <h3>Школа уклониста</h3>
                {schoolLog.length > 0 && (
                  <div className="table-card">
                    <table>
                      <thead>
                        <tr>
                          <th>поколение</th>
                          <th>приспособленность лучшего</th>
                          <th>выживаемость</th>
                          <th>жизнь чемпиона, с</th>
                        </tr>
                      </thead>
                      <tbody>
                        {schoolLog.map((r) => (
                          <tr key={r.gen}>
                            <td>{r.gen + 1}</td>
                            <td className={r.best < 500 ? 'map-win' : undefined} data-tip="Меньше — лучше: отрицательный приспособленность = выжила и сожгла ракету; 1000+ = сбита.">
                              {fmt(r.best, 1)}
                            </td>
                            <td>{Math.round(r.survive * 100)}%</td>
                            <td>{fmt(r.t, 1)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                {champ && (
                  <div className="stat-grid">
                    <span data-tip="Веса DN чемпиона (первая строка — тангаж, вторая — рыскание) и усиление контура.">
                      чемпион · усиление <b>{fmt(champ.gain, 2)}</b>
                    </span>
                    <button type="button" onClick={applyChamp} data-tip="Сохранить веса чемпиона в data/weights_evader.npz и посадить его за штурвал цели (evader_law=brain).">
                      Применить чемпиона
                    </button>
                  </div>
                )}
                {frame?.evader && (
                  <div className="stat-grid">
                    <span data-tip="Выход контура цели-уклониста: команды тангаж/рыскание (−1…1), ограниченные доступной перегрузкой n_target·g.">
                      DN цели <b>{fmt(frame.evader.dn.pitch, 2)} / {fmt(frame.evader.dn.yaw, 2)}</b>
                    </span>
                    <span data-tip="Держит ли сетчатка цели ракету в поле (после задержки сенсора).">
                      {frame.evader.lock ? 'видит ракету' : 'не видит ракету'} · {frame.evader.weights}
                    </span>
                  </div>
                )}
                <p className="lab-hint" style={{ margin: 0 }}>
                  геометрия поколения каждый раз новая, приспособленность — выжить, накрутить ракете перегрузки и дальность;
                  врождённый рефлекс — разворот к пеленгу (бабочка на огонь): эволюция учит цель разворачиваться ОТ ракеты
                </p>
                {schoolMsg && <p className="lab-hint" style={{ margin: 0 }}>{schoolMsg}</p>}
              </section>
            )}

            {learnMode === 'queen' && (
              <section className="ws-card">
                <h3>Красная королева: мозг против мозга</h3>
                {queenLog.length > 0 && (
                  <div className="table-card">
                    <table>
                      <thead>
                        <tr>
                          <th>поколение</th>
                          <th>взятия в бою</th>
                          <th>экзамен</th>
                          <th>приспособленность ракеты</th>
                          <th>приспособленность цели</th>
                          <th>жизнь цели, с</th>
                        </tr>
                      </thead>
                      <tbody>
                        {queenLog.map((r) => (
                          <tr key={r.gen}>
                            <td>{r.gen + 1}</td>
                            <td>{Math.round(r.ring * 100)}%</td>
                            <td data-tip="Чемпион против чемпиона на трёх фиксированных геометриях.">
                              {Math.round(r.exam * 100)}%
                            </td>
                            <td className={r.m < 500 ? 'map-win' : undefined} data-tip="Меньше — лучше: <500 = взял (время перехвата + усилие); 1000+ = промах.">
                              {fmt(r.m, 1)}
                            </td>
                            <td className={r.e < 500 ? 'map-win' : undefined} data-tip="Меньше — лучше: отрицательный = выжила и сожгла ракету; 1000+ = сбита.">
                              {fmt(r.e, 1)}
                            </td>
                            <td>{fmt(r.t, 1)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                {queenChamps && champBlock}
                <p className="lab-hint" style={{ margin: 0 }}>
                  пока обе стороны учатся друг против друга, прогресса не у кого занять — гонка и есть смысл игры
                </p>
                {queenMsg && <p className="lab-hint" style={{ margin: 0 }}>{queenMsg}</p>}
              </section>
            )}

            {learnMode === 'server' && (
              <section className="ws-card">
                <h3>Самообучение на сервере</h3>
                <p className="lab-hint" style={{ margin: 0 }}>
                  та же гонка поколений, но считаем её не вкладкой, а стендом: цикл уходит в фоновый поток,
                  страницу можно закрыть и вернуться к готовности — чемпионы сами сядут за штурвалы и встанут на ринг.
                </p>
                <div className="fields fields--2" style={{ padding: 0 }}>
                  <label data-tip="Первые места стартовой популяции достаются выученным мозгам с полки ринга — сильнейшим по последнему сводному бою (а пока боя не было — самым свежим дуэтам). Остальное — обычный случайный старт. Сила копится от кампании к кампании; выключено — каждая война начинается с врождённого рефлекса.">
                    наследие полки
                    <input type="checkbox" checked={srvHeirs} disabled={ringDuels.length === 0} onChange={(e) => setSrvHeirs(e.target.checked)} />
                  </label>
                  <label data-tip="Сезон без ручных кликов: досчитанная кампания тут же сводит круговой бой — новый чемпион против сильнейших по форме полки (не больше четырёх дуэтов). Форма и её движение обновляются сами, и следующая война стартует уже от свежих рангов. По «Стоп» свод не играется: прерванная кампания — не итог сезона.">
                    свести ринг после войны
                    <input type="checkbox" checked={srvRing} disabled={ringDuels.length === 0} onChange={(e) => setSrvRing(e.target.checked)} />
                  </label>
                </div>
                {srvHeirs && ringDuels.length > 0 && (
                  <p className="lab-hint" style={{ margin: 0 }}>
                    {heirByRank ? 'война начнётся с сильнейших по форме полки: ' : 'сводного боя ещё не было, война начнётся с самых свежих дуэтов: '}
                    {heirNames}
                  </p>
                )}
                {srvHeirs && ringDuels.length === 0 && (
                  <p className="lab-hint" style={{ margin: 0 }}>полка пуста: наследовать нечего, старт пойдёт с врождённого рефлекса</p>
                )}
                <p className="lab-hint" style={{ margin: 0 }}>
                  запуск и «Стоп» — в карточке выше: поколения досчитываются в фоновом потоке сервера, прогресс
                  опрашивается раз в 2 с; по остановке текущее поколение досчитается, и в живые веса уйдёт уже выученное
                </p>
                {srv && (srv.generations_done > 0 || srv.running) && (
                  <>
                    <p className="lab-hint" style={{ margin: 0 }}>
                      {srv.generations_done} / {srv.generations} поколений · {fmt(srv.seconds, 0)} с · seed {srv.seed}
                      {srv.inherit?.length ? ` · наследие дуэтов ${srv.inherit.join(', ')}` : ''}
                      {srv.auto_ring && srv.running && srv.generations_done >= srv.generations ? ' · сводим ринг…' : ''}
                      {srv.stop_requested ? ' · запрошена остановка' : ''}
                    </p>
                    <div className="table-card">
                      <table>
                        <thead>
                          <tr>
                            <th>поколение</th>
                            <th>взятия в бою</th>
                            <th>экзамен</th>
                            <th>приспособленность ракеты</th>
                            <th>приспособленность цели</th>
                          </tr>
                        </thead>
                        <tbody>
                          {srv.log.map((r) => (
                            <tr key={r.gen}>
                              <td>{r.gen + 1}</td>
                              <td>{Math.round(r.p_hit_ring * 100)}%</td>
                              <td>{Math.round(r.exam_p_hit * 100)}%</td>
                              <td className={r.missile_best >= 1000 ? 'is-bad' : ''}>{fmt(r.missile_best, 1)}</td>
                              <td>{fmt(r.evader_best, 1)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    {srv.saved.ring && !srv.running && (
                      <p className="lab-hint" style={{ margin: 0 }}>
                        на полке ринга: «{srv.saved.ring.label}» · weights {srv.saved.weights ? 'записаны' : 'не тронуты'}
                      </p>
                    )}
                    {srv.saved.ring_battle && !srv.running && (
                      <p className="lab-hint" style={{ margin: 0 }}>
                        сезон сведён: {srv.saved.ring_battle.duels.length} дуэта, сильнее всех
                        {srv.saved.ring_battle.standings[0]
                          ? ` «${srv.saved.ring_battle.standings[0].label}» (${srv.saved.ring_battle.standings[0].score} ${srv.saved.ring_battle.standings[0].score % 10 === 1 && srv.saved.ring_battle.standings[0].score % 100 !== 11 ? 'очко' : srv.saved.ring_battle.standings[0].score % 10 >= 2 && srv.saved.ring_battle.standings[0].score % 10 <= 4 && !(srv.saved.ring_battle.standings[0].score % 100 >= 11 && srv.saved.ring_battle.standings[0].score % 100 <= 14) ? 'очка' : 'очков'})`
                          : ''} — форма полки и её движение обновлены
                      </p>
                    )}
                    {queenChamps && champBlock}
                  </>
                )}
                {srvMsg && <p className="lab-hint" style={{ margin: 0 }}>{srvMsg}</p>}
              </section>
            )}

            {/* хроника — память о прошлых войнах, а не текущее действие:
                второстепенное не убираем, но и не держим в первом экране */}
            <details className="ws-fold">
              <summary>Хроника войн</summary>
              <section className="ws-card">
                <p className="lab-hint" style={{ margin: 0 }}>
                  лог поколений живёт в задаче стенда; хроника хранит уже доигранные кампании — их кривые взятий и
                  финал (что ушло в живые веса и на ринг). Память о самообучении переживает перезапуск, потому что
                  лежит в <code>data/queen_chronicle.json</code> под тем же volume, что и веса.
                </p>
                {chron.length === 0 && (
                  <p className="lab-hint" style={{ margin: 0 }}>
                    хронику ещё никто не вёл: выпустите королеву воевать в фоне — доигранная кампания запишется сама
                  </p>
                )}
                {chron.length > 0 && (
                  <div className="table-card">
                    <table>
                      <thead>
                        <tr>
                          <th>кампания</th>
                          <th>поколения</th>
                          <th>кривая взятий</th>
                          <th>итог</th>
                        </tr>
                      </thead>
                      <tbody>
                        {chron
                          .slice()
                          .reverse()
                          .map((c) => (
                            <tr key={c.id}>
                              <td data-tip={`курс ${c.scenario?.aspect ?? '—'} · популяция ${c.pop} · seed ${c.seed} · ${fmt(c.seconds, 0)} с`}>
                                {c.label}
                                {c.error ? ' · ⚠' : ''}
                              </td>
                              <td>
                                {c.generations}
                                {c.generations !== c.planned ? ` / ${c.planned}` : ''}
                                {c.stopped ? ' · стоп' : ''}
                              </td>
                              <td data-tip="Доля взятий ракеты в бою по поколениям: ▁ — провал, █ — берёт всех. Растёт — значит гонка вооружений действительно идёт.">
                                <span className="chip">{spark(c.curve.map((r) => r.p_hit_ring))}</span>
                              </td>
                              <td>
                                {Math.round((c.p_hit_first ?? 0) * 100)}% → {Math.round((c.p_hit_last ?? 0) * 100)}%
                                <div style={{ display: 'flex', gap: 4, marginTop: 4, flexWrap: 'wrap' }}>
                                  {c.applied && <span className="chip is-ok" data-tip="Чемпионы кампании сели за штурвалы живых мозгов.">в веса</span>}
                                  {c.shelved && <span className="chip" data-tip="Дуэт кампании встал на полку ринга чемпионов.">на ринге</span>}
                                  {!!c.inherited?.length && (
                                    <span className="chip" data-tip={`Кампания началась не с врождённого рефлекса: стартовые места достались дуэтам полки ${c.inherited.join(', ')}.`}>
                                      с полки
                                    </span>
                                  )}
                                  <button type="button" onClick={() => void dropChron(c.id)} data-tip="Вычеркнуть запись из хроники.">
                                    ✕
                                  </button>
                                </div>
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </section>
            </details>
          </>
        )}

        {layer === 'duels' && (
          <>
            <section className="ws-card">
              <h3>Ринг чемпионов: дуэты против дуэтов</h3>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                <button type="button" className="primary" disabled={ringBusy || busy} onClick={fightRing} data-tip="Каждый дуэт стреляет по каждому и держит оборону своим выученным уклонистом: три фиксированные геометрии на пару. Ранг = взятия в атаке + отражения в защите.">
                  {ringBusy ? 'сводятся…' : 'Свести на ринге'}
                </button>
                {ringDuels.map((d) => (
                  <button key={d.id} type="button" onClick={() => void dropDuel(d.id)} data-tip="Снять дуэт с полки.">
                    ✕ {d.label}
                  </button>
                ))}
              </div>
              {ringDuels.length === 0 && (
                <p className="lab-hint" style={{ margin: 0 }}>
                  полка пуста: вырастите чемпионов у Красной королевы и поставьте пару на ринг кнопкой «На полку ринга»
                </p>
              )}
              {ringDuels.length > 0 && (
                <>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginTop: 8 }}>
                    <label data-tip="Чья ракета стреляет: берём ракетную сторону дуэта с полки.">
                      ракета
                      <select value={attId} onChange={(e) => setPairAtt(Number(e.target.value))}>
                        {ringDuels.map((d) => (
                          <option key={d.id} value={d.id}>{d.label}</option>
                        ))}
                      </select>
                    </label>
                    <label data-tip="Чей уклонист уходит: берём цель другого дуэта — так на ринге можно сводить стороны в любой комбинации.">
                      цель
                      <select value={defId} onChange={(e) => setPairDef(Number(e.target.value))}>
                        {ringDuels.map((d) => (
                          <option key={d.id} value={d.id}>{d.label}</option>
                        ))}
                      </select>
                    </label>
                    <button type="button" className="primary" disabled={pairBusy || busy} onClick={() => void duelPair()} data-tip="Посадить сборную пару в живые веса и сразу выпустить её в сцену: одним кликом, без «Пуска».">
                      {pairBusy ? 'сажаем…' : 'Свести вживую'}
                    </button>
                  </div>
                  {lastDuel && (
                    <p className="lab-hint" style={{ margin: 0 }} data-tip="Прогноз взят из памяти последнего сводного боя — та же пара, те же три геометрии; «всухую» — 100% или 0% взятий.">
                      {duelPrediction(lastDuel)}
                    </p>
                  )}
                  {pairMsg && <p className="lab-hint" style={{ margin: 0 }}>{pairMsg}</p>}
                </>
              )}
              {ringStand && (
                <div className="table-card">
                  <table>
                    <thead>
                      <tr>
                        <th>ранг</th>
                        <th>дуэт</th>
                        <th>атака</th>
                        <th>оборона</th>
                        <th>очки</th>
                      </tr>
                    </thead>
                    <tbody>
                      {ringStand.map((s) => (
                        <tr key={s.id}>
                          <td>{s.rank}</td>
                          <td data-tip="Движение относительно прежней формы: ▲ поднялся, ▼ сел, ±0 удержал, без знака — новичок или первый свод.">{s.label}{ringMoveTag(s.id)}</td>
                          <td data-tip="Сколько целей взял этот дуэт, атакуя чужие мозги (из всех пар и геометрий).">
                            {s.attack_hits}/{s.attack_n} · {Math.round(s.p_attack * 100)}%
                          </td>
                          <td data-tip="Сколько чужих ракет пережил его уклонист.">
                            {s.defense_saves}/{s.defense_n} · {Math.round(s.p_defense * 100)}%
                          </td>
                          <td className={s.rank === 1 ? 'map-win' : undefined}>{s.score}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {ringCells && (
                <div className="table-card">
                  <table>
                    <thead>
                      <tr>
                        <th>атака → оборона</th>
                        <th>взятия</th>
                        <th>жизнь цели, с</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(() => {
                        const name = new Map(ringDuels.map((d) => [d.id, d.label]))
                        return ringCells.map((c) => (
                          <tr key={`${c.attacker}-${c.defender}`}>
                            <td>
                              {name.get(c.attacker) ?? c.attacker} → {name.get(c.defender) ?? c.defender}
                            </td>
                            <td className={c.p_hit >= 0.5 ? 'map-win' : undefined}>{Math.round(c.p_hit * 100)}%</td>
                            <td>{fmt(c.t_survived_median, 1)}</td>
                          </tr>
                        ))
                      })()}
                    </tbody>
                  </table>
                </div>
              )}
              {!ringCells && formCells.length > 0 && (
                <div className="table-card">
                  <table>
                    <thead>
                      <tr>
                        <th>атака ↓ / оборона →</th>
                        {formIds.map((d) => (
                          <th key={d}>{ringName(d)}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {formIds.map((a) => (
                        <tr key={a}>
                          <td>{ringName(a)}</td>
                          {formIds.map((d) => {
                            const p = a === d ? null : formP.get(`${a}|${d}`) ?? null
                            return (
                              <td
                                key={d}
                                data-tip={
                                  p === null
                                    ? 'Такой пары круг не сводил.'
                                    : `Доля взятий этой парой по всем геометриям последнего свода. Клик — свести её вживую прямо сейчас: стороны сядут на штурвалы и бой полетит на текущей геометрии стенда.`
                                }
                                className={p !== null && p >= 0.5 ? 'map-win' : undefined}
                                style={p !== null ? { cursor: 'pointer' } : undefined}
                                onClick={
                                  p !== null
                                    ? () => {
                                        setPairAtt(a)
                                        setPairDef(d)
                                        void duelPair(a, d)
                                      }
                                    : undefined
                                }
                              >
                                {p === null ? '—' : `${Math.round(p * 100)}%`}
                              </td>
                            )
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {reigns.length > 1 && (
                <p className="lab-hint" style={{ margin: 0 }} data-tip="Лента правлений: каждый сводный бой оставляет чем он кончился; галочкой — нынешний лидер формы. Старшие своды за горизонт памяти (24) не показываются.">
                  {seasons.length > reigns.length ? '… ' : ''}правления: {reigns.map((s, i) => (
                    <span key={i}>
                      {i > 0 ? ' → ' : ''}
                      {s.champ_label || `дуэт ${s.champ_id}`}
                      {i === reigns.length - 1 ? ' ✓' : ''}
                    </span>
                  ))}
                </p>
              )}
              {seasons.length > 1 && medalTop.length > 1 && (
                <p className="lab-hint" style={{ margin: 0 }} data-tip="Медальный зачёт: за каждый свод 1-е место — 3 очка, 2-е — 2, 3-е — 1; в скобках — чемпионские титулы. Считается по всей памяти ленты правлений, показаны первые пять.">
                  медальный зачёт: {medalTop.map((m, i) => (
                    <span key={m.id}>
                      {i > 0 ? ' · ' : ''}
                      {m.name} — {m.pts}
                      {m.gold > 0 ? ` (${m.gold} ${medalGold(m.gold)})` : ''}
                    </span>
                  ))}
                </p>
              )}
              <p className="lab-hint" style={{ margin: 0 }}>
                полка в data/queen_ring.json: чемпионы разных поколений переживают перезапуск и сводятся между собой
                {heirByRank && ringForm && ringForm.rows.length > 0
                  ? ` · форма запомнена: сильнее всех ${ringForm.rows[0].label || `дуэт ${ringForm.rows[0].id}`}${movePhrase(ringForm.rows[0])}, от него и растёт следующая кампания`
                  : ''}
                {formCells.length > 0 ? ' · матрица «кто кого бьёт» — память последнего круга, подсветка — больше половины взятий; клик по числу — свести эту пару вживую' : ''}
              </p>
            </section>

            {/* справочник и «взрослый» ринг законов — под рукой, но не в первом экране */}
            <details className="ws-fold">
              <summary>Оружие и классы мух</summary>
              <section className="ws-card">
                <div className="fields fields--2" style={{ padding: 0 }}>
                  <label data-tip="Боевая жизнь ракеты, с: окно, за которое она обязана сбить. В «Ринге» тоже ограничивает бой.">
                    боевая жизнь, с
                    <input type="number" min={5} max={sc.t_max} value={sc.fuse_life_s} onChange={(e) => set('fuse_life_s', Math.max(5, Number(e.target.value) || 5))} />
                  </label>
                  <label data-tip="Доступная перегрузка цели (уклониста), g. 0 — уклонисту нечем уходить.">
                    перегрузка цели
                    <input type="number" min={0} max={30} step={0.5} value={sc.n_target} onChange={(e) => set('n_target', Math.max(0, Number(e.target.value) || 0))} />
                  </label>
                  <label data-tip="Скорость ракеты-мухи, м/с.">
                    v ракеты
                    <input type="number" min={50} max={2000} step={10} value={sc.v_m} onChange={(e) => set('v_m', Math.max(50, Number(e.target.value) || 50))} />
                  </label>
                  <label data-tip="Скорость цели-самолёта, м/с.">
                    v цели
                    <input type="number" min={50} max={1000} step={10} value={sc.v_t} onChange={(e) => set('v_t', Math.max(50, Number(e.target.value) || 50))} />
                  </label>
                  <label data-tip="Предельная перегрузка ракеты, g.">
                    перегрузка ракеты
                    <input type="number" min={1} max={60} step={0.5} value={sc.n_max} onChange={(e) => set('n_max', Math.max(1, Number(e.target.value) || 1))} />
                  </label>
                  <label data-tip="Стартовая дальность, м.">
                    дальность
                    <input type="number" min={500} max={20000} step={100} value={sc.range_m} onChange={(e) => set('range_m', Math.max(500, Number(e.target.value) || 500))} />
                  </label>
                </div>
                <p className="lab-hint" style={{ margin: 0 }}>
                  честный бой одного класса: скорости сопоставимы (320/260 м/с), перегрузки 10 против 8 — тяжёлая ракета берёт всех закономерно
                </p>
              </section>
            </details>

            <details className="ws-fold">
              <summary>Ринг: закон против закона</summary>
              <section className="ws-card">
                <div className="fields fields--2" style={{ padding: 0 }}>
                  <label data-tip="Сколько прогонов на ячейку (при шуме сенсора — разные реализации; сводка по медиане и доле перехватов).">
                    повторов на бой
                    <input type="number" min={1} max={15} value={duelRepeats} onChange={(e) => onDuelRepeats(Math.max(1, Math.min(15, Number(e.target.value) || 1)))} />
                  </label>
                </div>
                <button type="button" className="primary" data-tip="Посчитать все бои выбранных сторон на текущих условиях. Только на стенде." disabled={duelBusy || busy} onClick={onRunMatrix}>
                  {duelBusy ? 'ринг считает…' : 'Считать ринг'}
                </button>
                <div className="table-card">
                  <table>
                    <thead>
                      <tr>
                        <th>ракета \\ цель</th>
                        {(duelMatrix?.evaders ?? []).map((e) => (
                          <th key={e} data-tip={evLabel(e)}>
                            {e === 'straight' ? 'нема.' : e}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(duelMatrix?.missiles ?? []).map((m) => (
                        <tr key={m}>
                          <td data-tip={mslLabel(m)}>
                            <small>{m}</small>
                          </td>
                          {(duelMatrix?.evaders ?? []).map((e) => {
                            const c = cellOf(m, e)
                            return (
                              <td key={e} className={c ? (c.win === 'missile' ? 'map-win' : '') : ''} data-tip={c ? `${evLabel(e)} · R_min ${fmt(c.cpa_m, 0)} м · усилие ${fmt(c.n_int, 0)} g·с · пик ${fmt(c.n_peak, 1)} g${c.hit_rate > 0 && c.hit_rate < 1 ? ` · перехватов ${Math.round(c.hit_rate * 100)}%` : ''}` : undefined}>
                                {c ? `${c.win === 'missile' ? '✕' : '·'} ${fmt(c.t_survived, 1)} с` : '—'}
                              </td>
                            )
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {!duelMatrix && <p className="lab-hint" style={{ padding: '6px 12px' }}>матрица считается на стенде: строки — стороны ракеты, столбцы — законы уклонения; ✕ — взяла ракета, · — выстояла цель</p>}
                </div>
                {duelMatrix && (
                  <div className="stat-grid">
                    <span data-tip="Ячеек, где победа осталась за ракетой (при повторах — по доле перехватов ≥50%).">
                      ракета взяла <b>{wins} / {duelMatrix.cells.length}</b>
                    </span>
                    <span data-tip="Аспект и боевая жизнь, на которых считался ринг.">
                      условия <b>{duelMatrix.scenario.aspect} · {fmt(duelMatrix.scenario.fuse_life_s, 0)} с · ×{duelMatrix.repeats}</b>
                    </span>
                    <button type="button" onClick={onExportCsv}>CSV ринга</button>
                  </div>
                )}
              </section>
            </details>
          </>
        )}
      </div>
    </div>
  )
}
