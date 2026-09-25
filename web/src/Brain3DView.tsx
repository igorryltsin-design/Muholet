import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { BrainKind, BrainRegion, Frame } from './types'

/**
 * Честный 3D-мозг: одна точка — один нейрон, яркость = реальная активация из кадра
 * (проекция всех нейронов в корзины, см. navedenie/circuit.py::_pack_activity).
 * Позиции точек детерминированы по регионам; ничего не анимируется «по таймеру».
 */

function mulberry32(seed: number) {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
const hash = (s: string) => s.split('').reduce((a, c) => (a * 31 + c.charCodeAt(0)) >>> 0, 7)

/** Анатомия: якорь и радиусы эллипсоида региона в условном мозге (x — латерально, y — вверх, z — вперёд). */
const ANATOMY: Record<string, { c: [number, number, number]; r: [number, number, number]; col: [number, number, number] }> = {
  'глаз': { c: [0, 0.1, 0.95], r: [0.42, 0.3, 0.1], col: [1, 0.82, 0.35] },
  'ламина': { c: [0.55, 0.02, 0.7], r: [0.16, 0.3, 0.1], col: [0.55, 1, 0.78] },
  'медулла': { c: [0.5, -0.05, 0.4], r: [0.2, 0.36, 0.16], col: [0.45, 1, 0.72] },
  'T4': { c: [0.46, -0.05, 0.26], r: [0.1, 0.18, 0.08], col: [0.65, 1, 0.85] },
  'T5': { c: [-0.46, -0.05, 0.26], r: [0.1, 0.18, 0.08], col: [0.6, 0.95, 0.85] },
  'T4/T5': { c: [0.46, -0.05, 0.26], r: [0.1, 0.18, 0.08], col: [0.65, 1, 0.85] },
  'лобула': { c: [0.45, -0.12, 0.05], r: [0.15, 0.24, 0.1], col: [0.4, 0.95, 0.75] },
  'LC11': { c: [0.45, -0.12, 0.05], r: [0.09, 0.14, 0.07], col: [0.95, 0.75, 0.4] },
  'LC10': { c: [-0.45, -0.12, 0.05], r: [0.09, 0.14, 0.07], col: [0.95, 0.7, 0.45] },
  'LPLC2': { c: [0.4, -0.18, -0.08], r: [0.09, 0.13, 0.06], col: [1, 0.45, 0.35] },
  'AOTU': { c: [0.14, 0.28, -0.18], r: [0.07, 0.07, 0.07], col: [0.75, 0.9, 1] },
  'центральный комплекс': { c: [0, 0.18, -0.12], r: [0.09, 0.09, 0.14], col: [0.7, 0.85, 1] },
  'грибовидное тело': { c: [0.2, 0.38, -0.05], r: [0.14, 0.11, 0.18], col: [0.5, 0.8, 1] },
  'пул': { c: [0, 0.3, -0.05], r: [0.1, 0.08, 0.1], col: [0.6, 0.75, 0.95] },
  'DN': { c: [0, -0.5, -0.35], r: [0.08, 0.06, 0.06], col: [1, 0.4, 0.3] },
}

const FALLBACK = { c: [0, 0, 0] as const, r: [0.1, 0.1, 0.1] as const, col: [0.6, 1, 0.8] as const }

/** Канонический состав нейронов по видам мозга — порядок совпадает с python (бины активности). */
function specFor(kind: BrainKind): BrainRegion[] {
  const mk = (spec: Array<[string, number]>): BrainRegion[] => {
    let off = 0
    return spec.map(([name, n]) => {
      const r = { name, start: off, n, mean: 0 }
      off += n
      return r
    })
  }
  if (kind === 'connectome')
    return mk([
      ['ламина', 1600],
      ['медулла', 75000],
      ['T4/T5', 6400],
      ['лобула', 44000],
      ['LPLC2', 1500],
      ['центральный комплекс', 2000],
      ['грибовидное тело', 8000],
      ['DN', 200],
    ])
  if (kind === 'full')
    return mk([
      ['глаз', 64],
      ['T4', 4],
      ['T5', 4],
      ['LC11', 4],
      ['LC10', 2],
      ['LPLC2', 1],
      ['AOTU', 6],
      ['грибовидное тело', 4096],
      ['пул', 32],
      ['DN', 2],
    ])
  return mk([
    ['глаз', 64],
    ['T4', 4],
    ['T5', 4],
    ['LC11', 4],
    ['LC10', 2],
    ['LPLC2', 1],
    ['AOTU', 6],
    ['грибовидное тело', 1],
    ['DN', 2],
  ])
}

/** Региональная активность из энергий слоёв кадра — когда кадр от другого мозга (рой/обучение схемой). */
function regionScalar(name: string, fr: Frame | null): number {
  const L = (fr?.layers || {}) as Record<string, number>
  const g = (a: string, b: string) => Math.min(1.4, Number(L[a] ?? L[b] ?? 0))
  switch (name) {
    case 'глаз':
    case 'ламина':
      return Math.min(1, g('Зрение', 'VISION') * 1.5)
    case 'медулла':
    case 'T4':
    case 'T5':
    case 'T4/T5':
      return Math.min(1, g('Поток', 'FLOW') * 1.6)
    case 'лобула':
    case 'LC11':
    case 'LC10':
      return Math.min(1, 0.5 * g('Решение', 'DECISION') + 0.4 * g('Зрение', 'VISION'))
    case 'LPLC2':
      return Math.min(1, Math.abs(fr?.circuit.lplc2 ?? 0) * 1.6 || g('Приближение', 'LOOM') * 1.6)
    case 'AOTU':
    case 'центральный комплекс':
    case 'грибовидное тело':
    case 'пул':
      return Math.min(1, g('Решение', 'DECISION') * 1.6)
    case 'DN':
      return Math.min(1, g('Мотор', 'MOTOR') * 1.6 + Math.abs(fr?.circuit.dn.yaw ?? 0) * 0.4)
    default:
      return 0
  }
}

const REGION_DESC: Record<string, string> = {
  'глаз': 'Фоторецепторы: кадр ГСН 8×8, светлое пятно — цель.',
  'ламина': 'Первичная оптическая пластинка: контраст и очистка сигнала.',
  'медулла': 'Крупнейший регион оптической доли — элементы движения.',
  'T4': 'T4: направление движения образа (горизонталь и вертикаль).',
  'T5': 'T5: тёмные края, направление движения.',
  'T4/T5': 'Детекторы направления: куда и как быстро ползёт образ цели.',
  'лобула': 'LC-нейроны: положение и признаки цели в кадре.',
  'LC11': 'Четверть кадра, где сидит цель.',
  'LC10': 'Смещение цели по вертикали и горизонтали.',
  'LPLC2': 'Радар столкновения: пятно растёт — цель прёт на муху.',
  'AOTU': 'Передний оптический узел: интеграция с центральным мозгом.',
  'центральный комплекс': 'Интеграция и «решение» о манёвре.',
  'грибовидное тело': 'Ассоциативное обучение: контекст и память.',
  'пул': 'Скрытый слой полного мозга (32 нейрона).',
  'DN': 'Даунины: тангаж и рыскание — команда на рули.',
}

type Built = { geom: THREE.BufferGeometry; count: number; binsPerPoint: Float32Array; gains: Float32Array; bases: Float32Array; deep: Float32Array; regions: BrainRegion[]; regionOf: Int16Array }

// Фоны тем одним цветом: THREE.Color переводит hex из sRGB в линейное рабочее
// пространство — те же значения (r/g/b) используются и как фон сцены, и как
// конец градиента «нейрон → фон» в буфере вершинных цветов (он тоже линейный).
const BG_DAY = new THREE.Color(0xe3ebe5)
const BG_NIGHT = new THREE.Color(0x05090c)
// sRGB-кодированное → линейное (как делает THREE.Color): буферы вершинных
// цветов читаются рендерером как линейные, писать в них sRGB-значения — мимо
// фона и мимо контраста (на светлом фоне такая «глубина» становилась белесой)
const s2l = (x: number) => (x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4))

function buildCloud(regions: BrainRegion[]): Built {
  const total = regions.reduce((s, r) => s + r.n, 0) || 1
  const pos = new Float32Array(total * 3)
  const bases = new Float32Array(total * 3)
  // «глубокая» версия той же палитры (тот же тон, L≈0.26 в sRGB-смысле,
  // записана уже линейно): в светлой теме аддитивное свечение на светлом
  // фоне не читается — активные нейроны красим вглубь
  const deep = new Float32Array(total * 3)
  const gains = new Float32Array(total)
  const binsPerPoint = new Float32Array(total)
  const regionOf = new Int16Array(total)
  const bpb = Math.max(total / 2048, 1e-9)
  let off = 0
  regions.forEach((reg, ri) => {
    const an = ANATOMY[reg.name] ?? FALLBACK
    const rng = mulberry32(hash(reg.name) + reg.n)
    const kd = 0.26 / Math.max(0.299 * an.col[0] + 0.587 * an.col[1] + 0.114 * an.col[2], 1e-3)
    // зеркалим по латерали: у зрительных регионов две копии (полушария)
    const mirrored = /ламина|медулла|лобула|T4\/T5|грибовидное тело|LPLC2/.test(reg.name)
    for (let i = 0; i < reg.n; i += 1) {
      const gx = rng()
      const gy = rng()
      const gz = rng()
      const gr = rng()
      const side = mirrored ? (i % 2 === 0 ? 1 : -1) : 1
      // равномерно внутри эллипсоида: гаусс-нормированный вектор × радиус × u^(1/3)
      let vx = gx * 2 - 1
      let vy = gy * 2 - 1
      let vz = gz * 2 - 1
      const len = Math.max(Math.hypot(vx, vy, vz), 1e-6)
      const rad = Math.cbrt(gr)
      vx = (vx / len) * rad * an.r[0]
      vy = (vy / len) * rad * an.r[1]
      vz = (vz / len) * rad * an.r[2]
      const j = (off + i) * 3
      pos[j] = an.c[0] * side + vx
      pos[j + 1] = an.c[1] + vy
      pos[j + 2] = an.c[2] + vz
      bases[j] = an.col[0]
      bases[j + 1] = an.col[1]
      bases[j + 2] = an.col[2]
      deep[j] = s2l(an.col[0] * kd)
      deep[j + 1] = s2l(an.col[1] * kd)
      deep[j + 2] = s2l(an.col[2] * kd)
      gains[off + i] = 0.55 + 0.45 * rng()
      binsPerPoint[off + i] = Math.min(2047, Math.floor((reg.start + i) / bpb))
      regionOf[off + i] = ri
    }
    off += reg.n
  })
  const geom = new THREE.BufferGeometry()
  geom.setAttribute('position', new THREE.BufferAttribute(pos, 3))
  const colors = new Float32Array(total * 3)
  geom.setAttribute('color', new THREE.BufferAttribute(colors, 3).setUsage(THREE.DynamicDrawUsage))
  return { geom, count: total, binsPerPoint, gains, bases, deep, regions, regionOf }
}

function decodeAct(b64: string): Uint8Array {
  const raw = atob(b64)
  const out = new Uint8Array(raw.length)
  for (let i = 0; i < raw.length; i += 1) out[i] = raw.charCodeAt(i)
  return out
}

/** Мягкая круглая точка вместо квадратного пикселя. */
function dotTexture(): THREE.Texture {
  const c = document.createElement('canvas')
  c.width = 64
  c.height = 64
  const g = c.getContext('2d')!
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32)
  grad.addColorStop(0, 'rgba(255,255,255,1)')
  grad.addColorStop(0.45, 'rgba(255,255,255,0.9)')
  grad.addColorStop(1, 'rgba(255,255,255,0)')
  g.fillStyle = grad
  g.fillRect(0, 0, 64, 64)
  return new THREE.CanvasTexture(c)
}

function makeTag(text: string, color = '#9fd8c8', bg = 'rgba(5,9,12,0.55)') {
  // канвас по ширине текста: длинные названия («грибовидное тело») не обрезаются
  const font = '600 30px "Chakra Petch", sans-serif'
  const meas = document.createElement('canvas').getContext('2d')!
  meas.font = font
  const tw = meas.measureText(text).width
  const c = document.createElement('canvas')
  c.width = Math.ceil(tw + 26)
  c.height = 52
  const g = c.getContext('2d')!
  g.font = font
  g.textAlign = 'center'
  g.textBaseline = 'middle'
  g.fillStyle = bg
  g.fillRect(2, 6, c.width - 4, c.height - 12)
  g.fillStyle = color
  g.fillText(text, c.width / 2, c.height / 2 + 1)
  const tex = new THREE.CanvasTexture(c)
  const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }))
  const h = 0.085
  spr.scale.set(h * (c.width / c.height), h, 1)
  return spr
}

/** Проводка сигнальных путей между регионами (только те, что есть в мозге). */
const LINKS: Array<[string, string]> = [
  ['глаз', 'ламина'],
  ['ламина', 'медулла'],
  ['глаз', 'T4'],
  ['глаз', 'T5'],
  ['медулла', 'T4/T5'],
  ['медулла', 'лобула'],
  ['T4', 'LPLC2'],
  ['LC11', 'LPLC2'],
  ['лобула', 'LPLC2'],
  ['LC11', 'AOTU'],
  ['лобула', 'AOTU'],
  ['AOTU', 'центральный комплекс'],
  ['центральный комплекс', 'грибовидное тело'],
  ['AOTU', 'грибовидное тело'],
  ['грибовидное тело', 'пул'],
  ['грибовидное тело', 'DN'],
  ['центральный комплекс', 'DN'],
]

export function Brain3DView({ frame, kind, day }: { frame: Frame | null; kind: BrainKind; day?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const frameRef = useRef(frame)
  const kindRef = useRef(kind)
  const dayRef = useRef(day ?? false)
  frameRef.current = frame
  kindRef.current = kind
  dayRef.current = day ?? false
  const [hover, setHover] = useState<string | null>(null)
  const [layout, setLayout] = useState<BrainRegion[]>(() => specFor(kind))
  const hoverRef = useRef<string | null>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return

    const scene = new THREE.Scene()
    scene.background = BG_NIGHT.clone()
    const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 50)
    camera.position.set(0, 0.4, 1.75)
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true })
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1))

    const controls = new OrbitControls(camera, canvas)
    controls.enableDamping = true
    controls.dampingFactor = 0.1
    controls.autoRotate = true
    controls.autoRotateSpeed = 0.7
    controls.minDistance = 0.8
    controls.maxDistance = 6

    const group = new THREE.Group()
    scene.add(group)
    let built: Built | null = null
    let points: THREE.Points | null = null
    const extras: Array<{ obj: THREE.Object3D; dispose: () => void }> = []

    const clearExtras = () => {
      extras.forEach((e) => {
        group.remove(e.obj)
        e.dispose()
      })
      extras.length = 0
    }

    const ensureCloud = (spec: BrainRegion[]) => {
      const sig = `${kindRef.current}:${spec.map((r) => r.name + r.n).join(',')}`
      if (built && (built as Built & { sig?: string }).sig === sig) return
      built = buildCloud(spec)
      ;(built as Built & { sig?: string }).sig = sig
      setLayout(spec)
      if (points) {
        group.remove(points)
        points.geometry.dispose()
        ;(points.material as THREE.Material).dispose()
      }
      clearExtras()
      const total = built.count
      const small = total <= 200
      const mid = total <= 5000
      const d0 = dayRef.current // светлая тема: аддитивное свечение на светлом фоне — бледно
      points = new THREE.Points(
        built.geom,
        new THREE.PointsMaterial({
          size: small ? 0.075 : mid ? 0.03 : 0.014,
          map: dotTexture(),
          vertexColors: true,
          transparent: true,
          opacity: d0 ? 1 : mid ? 0.85 : 1,
          alphaTest: 0.02,
          blending: d0 ? THREE.NormalBlending : THREE.AdditiveBlending,
          depthWrite: false,
          sizeAttenuation: true,
        }),
      )
      group.add(points)

      // подписи регионов (с реальными размерами) + проводка путей между якорями
      const present = new Set(spec.map((r) => r.name))
      const dayNow = dayRef.current
      present.forEach((name) => {
        const an = ANATOMY[name] ?? FALLBACK
        const n = spec.find((r) => r.name === name)?.n ?? 0
        const tag = makeTag(
          `${name} · ${n.toLocaleString('ru')}`,
          dayNow ? '#0a4a3a' : '#9fd8c8',
          dayNow ? 'rgba(255,255,255,0.72)' : 'rgba(5,9,12,0.55)',
        )
        // слегка разношу подписи по вертикали, чтобы соседние не слипались
        const lift = 0.09 + (hash(name) % 5) * 0.025
        tag.position.set(an.c[0], an.c[1] + an.r[1] + lift, an.c[2])
        group.add(tag)
        extras.push({
          obj: tag,
          dispose: () => {
            tag.material.map?.dispose()
            tag.material.dispose()
          },
        })
      })
      LINKS.forEach(([a, b]) => {
        const ca = ANATOMY[a]?.c
        const cb = ANATOMY[b]?.c
        if (!ca || !cb || !present.has(a) || !present.has(b)) return
        const line = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(...ca),
            new THREE.Vector3(...cb),
          ]),
          new THREE.LineBasicMaterial({ color: dayNow ? 0x4a7a6c : 0x3d8f88, transparent: true, opacity: dayNow ? 0.5 : 0.35 }),
        )
        group.add(line)
        extras.push({
          obj: line,
          dispose: () => {
            line.geometry.dispose()
            ;(line.material as THREE.Material).dispose()
          },
        })
      })

      // импульсы: бегущие точки вдоль проводки; скорость/яркость — от реальной
      // активности региона-источника (без таймерных «декораций»)
      const activeLinks = LINKS.filter(([a, b]) => present.has(a) && present.has(b) && ANATOMY[a] && ANATOMY[b])
      const pulseGeo = new THREE.BufferGeometry()
      const pulseArr = new Float32Array(activeLinks.length * 3 * 3)
      pulseGeo.setAttribute('position', new THREE.BufferAttribute(pulseArr, 3).setUsage(THREE.DynamicDrawUsage))
      const pulseMat = new THREE.PointsMaterial({
        size: 0.045,
        map: dotTexture(),
        color: dayNow ? 0x0e5a44 : 0xd8ffe9,
        transparent: true,
        opacity: 0.9,
        blending: dayNow ? THREE.NormalBlending : THREE.AdditiveBlending,
        depthWrite: false,
      })
      const pulses = new THREE.Points(pulseGeo, pulseMat)
      group.add(pulses)
      extras.push({
        obj: pulses,
        dispose: () => {
          pulseGeo.dispose()
          pulseMat.dispose()
        },
      })
      pulseState = { pairs: activeLinks, arr: pulseArr, geo: pulseGeo, mat: pulseMat }
    }

    const fit = () => {
      const w = Math.max(canvas.clientWidth, 160)
      const h = Math.max(canvas.clientHeight, 100)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h, false)
    }
    fit()
    const ro = new ResizeObserver(fit)
    ro.observe(canvas)

    let raf = 0
    let lastPaint = 0
    let lastDay: boolean | null = null
    let lastHoverPainted: string | null = null
    let pulseState: { pairs: Array<[string, string]>; arr: Float32Array; geo: THREE.BufferGeometry; mat: THREE.PointsMaterial } | null = null
    // активность региона-источника (0…1) из энергий слоёв кадра
    const regionEnergy = (name: string, fr: Frame | null): number => {
      const L = (fr?.layers || {}) as Record<string, number>
      const g = (a: string, b: string) => Math.min(1.2, Number(L[a] ?? L[b] ?? 0))
      switch (name) {
        case 'глаз':
        case 'ламина':
          return g('Зрение', 'VISION')
        case 'медулла':
        case 'T4':
        case 'T5':
        case 'T4/T5':
          return g('Поток', 'FLOW')
        case 'лобула':
        case 'LC11':
        case 'LC10':
          return Math.min(1, 0.5 * g('Решение', 'DECISION') + 0.4 * g('Зрение', 'VISION'))
        case 'LPLC2':
          return Math.min(1, Math.abs(fr?.circuit.lplc2 ?? 0) * 1.6 || g('Приближение', 'LOOM') * 1.6)
        case 'AOTU':
        case 'центральный комплекс':
        case 'грибовидное тело':
        case 'пул':
          return g('Решение', 'DECISION')
        case 'DN':
          return Math.min(1, g('Мотор', 'MOTOR') * 1.6 + Math.abs(fr?.circuit.dn.yaw ?? 0) * 0.4)
        default:
          return 0
      }
    }
    const tick = () => {
      const fr = frameRef.current
      ensureCloud(specFor(kindRef.current))
      const now = performance.now()
      // тема: фон, blending и палитра облака/импульсов/подписей меняются только при смене;
      // пересборка облака (built=null) даст дневные материалы, теги и проводку на следующем тике
      if (lastDay !== dayRef.current) {
        lastDay = dayRef.current
        scene.background = (dayRef.current ? BG_DAY : BG_NIGHT).clone()
        if (points) {
          const pm = points.material as THREE.PointsMaterial
          pm.blending = dayRef.current ? THREE.NormalBlending : THREE.AdditiveBlending
        }
        if (pulseState) {
          pulseState.mat.color.set(dayRef.current ? 0x0e5a44 : 0xd8ffe9)
          pulseState.mat.blending = dayRef.current ? THREE.NormalBlending : THREE.AdditiveBlending
        }
        built = null
      }
      // без кадра мозг «спит»: вращение медленное, импульсы скрыты
      controls.autoRotateSpeed = fr ? 0.7 : 0.12
      const colors = built?.geom.getAttribute('color') as THREE.BufferAttribute | undefined
      if (built && colors) {
        // перекраска ~25 раз/с: активность меняется ровно с приходом кадров;
        // смена наведённого региона перекрашивает немедленно (подсветка зоны)
        if (now - lastPaint > (fr ? 36 : 600) || lastHoverPainted !== hoverRef.current) {
          lastPaint = now
          lastHoverPainted = hoverRef.current
          const arr = colors.array as Float32Array
          // активность бинами — только от кадра СВОЕГО мозга; чужие кадры (рой/обучение
          // схемой) ложатся регионально, чтобы панель не прыгала между видами мозга
          const matched = fr && fr.circuit.kind === kindRef.current && fr.circuit.act_b64
          const act = matched ? decodeAct(fr.circuit.act_b64) : null
          const scalars = act ? null : built.regions.map((r) => regionScalar(r.name, fr))
          const n = built.count
          const min = fr ? 0.16 : 0.22 // мозг виден даже до пуска
          const dim = built.count > 20000 ? 0.55 : 1 // гасим плотные облака, иначе пересвет
          const hovIdx = hoverRef.current ? built.regions.findIndex((r) => r.name === hoverRef.current) : -1
          for (let i = 0; i < n; i += 1) {
            const b = act ? act[built.binsPerPoint[i]] / 255 : scalars![built.regionOf[i]]
            const v = Math.min(1, b * built.gains[i])
            const j = i * 3
            // наведённый регион подсвечен, остальные притушены — зона читается целиком
            const hov = hovIdx >= 0 ? (built.regionOf[i] === hovIdx ? 1.35 : 0.3) : 1
            const s = Math.min(1.2, (min + v * 0.9) * dim * hov)
            if (dayRef.current) {
              // светлая тема: активность = глубина цвета (mix deep→фон), не яркость;
              // и deep, и фон лежат в буфере линейно (рендерер сам кодирует в sRGB)
              const a = Math.max(0.14, Math.min(1, s))
              const ia = 1 - a
              arr[j] = built.deep[j] * a + BG_DAY.r * ia
              arr[j + 1] = built.deep[j + 1] * a + BG_DAY.g * ia
              arr[j + 2] = built.deep[j + 2] * a + BG_DAY.b * ia
            } else {
              arr[j] = built.bases[j] * s
              arr[j + 1] = built.bases[j + 1] * s
              arr[j + 2] = built.bases[j + 2] * s
            }
          }
          colors.needsUpdate = true
        }
      }
      controls.update()
      // импульсы вдоль проводки: позиция по фазе, скорость/видимость — от активности источника.
      // Без кадра — «сон»: редкие фоновые искры по случайным путям, остальное скрыто.
      if (pulseState) {
        const t = performance.now() / 1000
        const slot = Math.floor(t * 0.8)
        let j = 0
        pulseState.pairs.forEach(([a, b], idx) => {
          const e = regionEnergy(a, fr)
          const ca = ANATOMY[a].c
          const cb = ANATOMY[b].c
          const shown = fr ? e > 0.06 : (idx + slot) % 5 === 0
          const speed = fr ? 0.22 + e * 0.85 : 0.12
          for (let p = 0; p < 3; p += 1) {
            if (shown) {
              const tt = (t * speed + p / 3) % 1
              pulseState!.arr[j] = ca[0] + (cb[0] - ca[0]) * tt
              pulseState!.arr[j + 1] = ca[1] + (cb[1] - ca[1]) * tt + Math.sin(tt * Math.PI) * 0.09
              pulseState!.arr[j + 2] = ca[2] + (cb[2] - ca[2]) * tt
            } else {
              pulseState!.arr[j] = 9999
              pulseState!.arr[j + 1] = 9999
              pulseState!.arr[j + 2] = 9999
            }
            j += 3
          }
        })
        pulseState.geo.attributes.position.needsUpdate = true
        const avgE = pulseState.pairs.reduce((s, [a]) => s + regionEnergy(a, fr), 0) / Math.max(pulseState.pairs.length, 1)
        pulseState.mat.opacity = fr ? 0.15 + avgE * 0.85 : 0.18
      }
      renderer.render(scene, camera)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)

    // наведение: ближайший центр региона в экранных координатах
    const onMove = (ev: MouseEvent) => {
      const rect = canvas.getBoundingClientRect()
      if (!built) return
      const v = new THREE.Vector3()
      let bestName: string | null = null
      let bestD = 70
      for (const reg of built.regions) {
        const an = ANATOMY[reg.name] ?? FALLBACK
        v.set(an.c[0], an.c[1], an.c[2]).applyMatrix4(group.matrixWorld).project(camera)
        const sx = rect.left + ((v.x + 1) / 2) * rect.width
        const sy = rect.top + ((1 - v.y) / 2) * rect.height
        const d = Math.hypot(sx - ev.clientX, sy - ev.clientY)
        if (d < bestD) {
          bestD = d
          bestName = reg.name
        }
      }
      if (bestName !== hoverRef.current) {
        hoverRef.current = bestName
        setHover(bestName)
      }
    }
    const onLeave = () => {
      hoverRef.current = null
      setHover(null)
    }
    canvas.addEventListener('mousemove', onMove)
    canvas.addEventListener('mouseleave', onLeave)

    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
      canvas.removeEventListener('mousemove', onMove)
      canvas.removeEventListener('mouseleave', onLeave)
      controls.dispose()
      renderer.dispose()
      clearExtras()
      if (points) {
        points.geometry.dispose()
        ;(points.material as THREE.Material).dispose()
      }
    }
  }, [])

  const spec = specFor(kind)
  const total = spec.reduce((s, r) => s + r.n, 0)
  const regions = layout
  const hoverReg = regions.find((r) => r.name === hover)
  const syn = frame?.circuit ? (frame.circuit as Frame['circuit'] & { n_synapses?: number }).n_synapses : undefined
  const mismatch = Boolean(frame && frame.circuit.kind !== kind)
  return (
    <div className="brainwrap" ref={wrapRef}>
      {/* на iPhone Fullscreen API для элементов нет — кнопку не показываем */}
      {document.fullscreenEnabled && (
        <button
          type="button"
          className="brain__fs"
          data-tip="Развернуть мозг во весь экран (Esc — выйти)."
          onClick={() => {
            const el = wrapRef.current
            if (!el) return
            if (document.fullscreenElement) void document.exitFullscreen()
            else void el.requestFullscreen()
          }}
        >
          во весь экран
        </button>
      )}
      <canvas className="viewport viewport--brain" ref={ref} />
      <p className="brainhint">
        {hover && hoverReg ? (
          <>
            <b>{hover}</b> — {REGION_DESC[hover] ?? 'регион мозга.'} Активность{' '}
            {((mismatch ? regionScalar(hover, frame) : hoverReg.mean) * 100).toFixed(1)}% · нейронов{' '}
            {hoverReg.n.toLocaleString('ru')}
          </>
        ) : (
          <>
            Каждая точка — нейрон, яркость — его реальная активность.{' '}
            <span className="for-mouse">Вращайте мышью, наводите на регионы.</span>
            <span className="for-touch">Вращайте пальцем, касайтесь регионов.</span>
          </>
        )}
      </p>
      <p className="brainmeta">
        {total.toLocaleString('ru')} нейронов
        {syn ? ` · ${syn.toLocaleString('ru')} синапсов регионов в модели` : ''} · коннектом-модель: регионы по FlyWire, считывание синтетическое
        {mismatch ? ' · сейчас летит схема — активность показана по регионам' : ''}
      </p>
    </div>
  )
}
