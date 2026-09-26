import { useEffect, useRef } from 'react'
import { say, setBuzz } from './voice'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import type { CamMode, FlyGenome, Frame, Scenario } from './types'
import { perfBudget, recordFrameMs } from './perf'
import { buzz, HAPTIC } from './haptics'

const KM = 0.001

/** СК симулятора → мир three.js: X — дальность, Y (мир) — высота, Z (мир) — бок. */
function to3(p: number[]) {
  return new THREE.Vector3(p[0] * KM, p[2] * KM, p[1] * KM)
}

/** Точка следа с реальной нагрузкой в момент прохода: 0 — спокойно, 1 — на пределе. */
type TrailPoint = { p: THREE.Vector3; g: number }

/** Единичная окружность в локальных координатах — круг БЧ масштабируется/двигается, не перестраивается. */
function unitCircleGeometry(segments = 40) {
  const pts: THREE.Vector3[] = []
  for (let i = 0; i < segments; i += 1) {
    const a = (i / segments) * Math.PI * 2
    pts.push(new THREE.Vector3(Math.cos(a), Math.sin(a), 0))
  }
  return new THREE.BufferGeometry().setFromPoints(pts) // LineLoop замыкает последнюю точку на первую сама
}

/** Перестраивает геометрию линии следа с цветом по вершинам (градиент «спокойно → на пределе»).
 * Полный ребилд на ~25 Гц телеметрии для ≤260 точек — дешевле, чем аккуратный incremental-update. */
function setTrailGeometry(line: THREE.Line, trail: TrailPoint[], cold: THREE.Color, hot: THREE.Color) {
  const n = trail.length
  const pos = new Float32Array(n * 3)
  const col = new Float32Array(n * 3)
  const c = new THREE.Color()
  for (let i = 0; i < n; i += 1) {
    const { p, g } = trail[i]
    pos[i * 3] = p.x
    pos[i * 3 + 1] = p.y
    pos[i * 3 + 2] = p.z
    c.copy(cold).lerp(hot, Math.max(0, Math.min(1, g)))
    col[i * 3] = c.r
    col[i * 3 + 1] = c.g
    col[i * 3 + 2] = c.b
  }
  line.geometry.dispose()
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3))
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3))
  line.geometry = geo
}

export type Playback = {
  results: {
    traj_m: number[][]
    traj_t: number[][]
    fitness: number
    hit: boolean
    /** геном и промах — для «Чемпион → в схему» */
    fly?: FlyGenome
    miss_m?: number
    /** идентификатор варианта (kind мозга или закона) — для бейджа наложения */
    kind?: string
    /** подпись траектории на сцене («стажёр», «ветеран», …) */
    label?: string
  }[]
  bestIdx: number
  startedAt: number
  durationMs: number
  /** демо-гонка: ракета[0] — стажёр, ракета[bestIdx] — ветеран, летят одновременно */
  race?: boolean
  /** подпись над веером траекторий: по умолчанию «рой · N мух», у учебного боя
   *  дуэли своя («школа · поколение 3 · tpn») — подписывать её роем было бы неверно */
  caption?: string
  /** инстант-реплей перехвата: те же данные хвоста прогона, просто медленнее показаны —
   *  камера плавно облетает точку удара вместо обычного «веера роя/дуэли» поведения */
  cinematic?: boolean
}

/** Палитры 3D-сцены: ночь (фосфор) и день (пасмурное небо, тёмные метки). */
const THEME_NIGHT = {
  bg: '#0c1418',
  grid1: 0x3d8f88,
  grid2: 0x1a3a40,
  plane: 0x0a1820,
  ambient: 0xb7d8d0,
  ambientI: 0.85,
  sunI: 1.15,
  x: '#ff6b6b',
  y: '#6bb0ff',
  z: '#7dffc8',
  m: '#7dffc8',
  los: 0xe7c15a,
  mLine: 0x7dffc8,
  tLine: 0xff7a55,
  // «горячий» конец градиента трассы при полной перегрузке (n_req/n_lim = 1)
  mLineHot: 0xff7a5e,
  tLineHot: 0xff3b2f,
  req: 0x7dffc8,
  tgt: 0xff7a55,
  zem: 0xe7c15a,
  arrow: 0xff8a5a,
  bloom: 0.9,
  swarmBad: 0xff5a3c,
  swarmGood: 0x7dffc8,
  swarmBest: 0xd2ffe9,
}
const THEME_DAY = {
  bg: '#d9e4df',
  grid1: 0x6f988c,
  grid2: 0xaec4bb,
  plane: 0xc6d5ce,
  ambient: 0xffffff,
  ambientI: 1.05,
  sunI: 1.5,
  x: '#8a2a1c',
  y: '#1c5a8a',
  z: '#0a6a4e',
  m: '#0a6a4e',
  los: 0x8a6a1a,
  mLine: 0x0a6a4e,
  tLine: 0xb03a22,
  mLineHot: 0xb3301c,
  tLineHot: 0x7a1810,
  req: 0x0a6a4e,
  tgt: 0xb03a22,
  zem: 0x8a6a1a,
  arrow: 0xb0452a,
  bloom: 0.3,
  swarmBad: 0x8a2a1c,
  swarmGood: 0x0a6a4e,
  swarmBest: 0x053d2c,
}

function makeLabel(text: string, color: string, scale = 1.6) {
  const c = document.createElement('canvas')
  c.width = 320
  c.height = 80
  const g = c.getContext('2d')!
  const tex = new THREE.CanvasTexture(c)
  const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }))
  const draw = (col: string, newText?: string) => {
    if (newText !== undefined) text = newText
    g.font = '600 40px "Chakra Petch", sans-serif'
    // холст по ширине текста с запасом: длинные подписи («ПН (эталон) · 41 м»)
    // на фиксированных 320px резались по краям — пропадала ножка у «П»
    const w = Math.max(64, Math.ceil(g.measureText(text).width) + 32)
    if (c.width !== w) c.width = w // ресайз сбрасывает состояние контекста — шрифт задаём заново
    g.clearRect(0, 0, c.width, c.height)
    g.font = '600 40px "Chakra Petch", sans-serif'
    g.fillStyle = col
    g.textAlign = 'center'
    g.textBaseline = 'middle'
    g.fillText(text, c.width / 2, c.height / 2)
    tex.needsUpdate = true
    spr.userData.text = text // сверка подписей живой проверкой по графу сцены
    // мир: высота подписи постоянна (scale·0.25), ширина растёт с текстом;
    // размер глифов совпадает со старым поведением (320-холст, scale-ширина)
    spr.scale.set((scale * c.width) / 320, scale * 0.25, 1)
  }
  draw(color)
  return { spr, set: draw }
}

function ringSprite(color = 'rgba(255,192,112,0.9)') {
  const c = document.createElement('canvas')
  c.width = 128
  c.height = 128
  const g = c.getContext('2d')!
  g.strokeStyle = color
  g.lineWidth = 5
  g.beginPath()
  g.arc(64, 64, 56, 0, Math.PI * 2)
  g.stroke()
  const spr = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: new THREE.CanvasTexture(c),
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      opacity: 0,
    }),
  )
  return spr
}

/** Пусковая расстановка — зеркало sim.py::spawn, чтобы сцена до пуска совпадала с прогоном. */
function spawnPair(sc: Scenario): { m: number[]; t: number[] } {
  if (sc.aspect === 'free') return { m: [0, 0, sc.alt_m], t: [sc.free_tx, sc.free_ty, sc.free_talt] }
  if (sc.aspect === 'beam') return { m: [0, 0, 4000], t: [sc.range_m * 0.55, -sc.range_m * 0.65, 4000] }
  if (sc.aspect === 'tail-chase') return { m: [0, 0, 4000], t: [sc.range_m * 0.45, sc.off_axis_m * 0.4, 4040] }
  return { m: [0, 0, 4000], t: [sc.range_m, sc.off_axis_m, 4080] }
}

function scKey(sc: Scenario) {
  const free = sc.aspect === 'free' ? `|${sc.free_tx}|${sc.free_ty}|${sc.free_talt}` : ''
  return `${sc.aspect}|${sc.range_m}|${sc.off_axis_m}|${sc.alt_m}${free}`
}

/** Стартовый ракурс, базовый масштаб сцены и моделей под сценарий (единица мира — 1 км). */
function homeView(sc: Scenario) {
  const pair = spawnPair(sc)
  const m0 = to3(pair.m)
  const t0 = to3(pair.t)
  const span = Math.max(m0.distanceTo(t0), 1.2)
  const tgt = m0.clone().lerp(t0, 0.42)
  const dist = Math.min(Math.max(span * 1.6, 4), 120)
  const pos = tgt.clone().addScaledVector(new THREE.Vector3(-0.55, 0.35, 0.75).normalize(), dist)
  const modelScale = Math.min(1, Math.max(0.6, 6 / span))
  return { pos, tgt, span, dist, modelScale }
}

function sampleTraj(traj: number[][], k: number): number[] {
  if (traj.length === 0) return [0, 0, 0]
  if (traj.length === 1) return traj[0]
  const x = k * (traj.length - 1)
  const i = Math.min(traj.length - 2, Math.floor(x))
  const f = x - i
  const a = traj[i]
  const b = traj[i + 1]
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
}

function velAt(traj: number[][], k: number): number[] {
  if (traj.length < 2) return [0, 0, 0]
  const x = k * (traj.length - 1)
  const i = Math.min(traj.length - 2, Math.floor(x))
  const a = traj[i]
  const b = traj[i + 1]
  const dt = 0.25
  return [(b[0] - a[0]) / dt, (b[1] - a[1]) / dt, (b[2] - a[2]) / dt]
}

/** Стилизованный истребитель из примитивов, носом вдоль +Z. */
function buildJet(): THREE.Group {
  const g = new THREE.Group()
  const paint = new THREE.MeshStandardMaterial({ color: 0xff7a55, emissive: 0x5a1808, emissiveIntensity: 0.9, metalness: 0.2, roughness: 0.5, side: THREE.DoubleSide, transparent: true })
  const dark = new THREE.MeshStandardMaterial({ color: 0xd94f30, emissive: 0x401004, emissiveIntensity: 0.8, roughness: 0.6, side: THREE.DoubleSide, transparent: true })
  const glass = new THREE.MeshStandardMaterial({ color: 0x9fd8ff, emissive: 0x1a3a50, emissiveIntensity: 0.6, roughness: 0.2, metalness: 0.4, transparent: true })
  g.userData.mats = [paint, dark, glass]

  const fuselage = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.075, 0.42, 10), paint)
  fuselage.rotation.x = Math.PI / 2
  g.add(fuselage)
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.055, 0.16, 10), dark)
  nose.rotation.x = Math.PI / 2
  nose.position.z = 0.29
  g.add(nose)
  const canopy = new THREE.Mesh(new THREE.SphereGeometry(0.05, 10, 8), glass)
  canopy.scale.set(0.8, 0.6, 1.4)
  canopy.position.set(0, 0.045, 0.1)
  g.add(canopy)

  // плоскость: форма в XY (размах +x, назад −y), поворотом Rx(π/2) ложится в XZ
  const wingShape = new THREE.Shape()
  wingShape.moveTo(0, 0)
  wingShape.lineTo(0.46, -0.2)
  wingShape.lineTo(0.46, -0.32)
  wingShape.lineTo(0.05, -0.16)
  wingShape.lineTo(0, 0)
  const wingGeo = new THREE.ExtrudeGeometry(wingShape, { depth: 0.014, bevelEnabled: false })
  // правое крыло: +x, назад
  const wingR = new THREE.Mesh(wingGeo, paint)
  wingR.rotation.x = Math.PI / 2
  g.add(wingR)
  // левое крыло: оборачиваем в группу с поворотом на 180° вокруг продольной оси —
  // размах уходит в −x, стреловидность и толщина остаются на месте
  const wingLWrap = new THREE.Group()
  wingLWrap.rotation.z = Math.PI
  const wingLInner = new THREE.Mesh(wingGeo, paint)
  wingLInner.rotation.x = Math.PI / 2
  wingLWrap.add(wingLInner)
  g.add(wingLWrap)

  // стабилизаторы — тем же способом
  const tailShape = new THREE.Shape()
  tailShape.moveTo(0, 0)
  tailShape.lineTo(0.16, -0.12)
  tailShape.lineTo(0.16, -0.18)
  tailShape.lineTo(0.02, -0.1)
  tailShape.lineTo(0, 0)
  const tailGeo = new THREE.ExtrudeGeometry(tailShape, { depth: 0.01, bevelEnabled: false })
  const tailR = new THREE.Mesh(tailGeo, dark)
  tailR.rotation.x = Math.PI / 2
  tailR.position.z = -0.24
  g.add(tailR)
  const tailLWrap = new THREE.Group()
  tailLWrap.rotation.z = Math.PI
  tailLWrap.position.z = -0.24
  const tailL = new THREE.Mesh(tailGeo, dark)
  tailL.rotation.x = Math.PI / 2
  tailLWrap.add(tailL)
  g.add(tailLWrap)

  // киль: плоская плоскость поворачивается из XZ в вертикаль YZ
  const fin = new THREE.Mesh(tailGeo, dark)
  fin.rotation.x = Math.PI / 2
  fin.rotation.y = Math.PI / 2
  fin.position.set(0.005, 0, -0.24)
  g.add(fin)

  // сопло
  const nozzle = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.05, 0.06, 10), dark)
  nozzle.rotation.x = Math.PI / 2
  nozzle.position.z = -0.23
  g.add(nozzle)
  return g
}

export function EngagementView({
  frame,
  scenario,
  playback,
  day,
  camMode,
  geometryOn,
  sceneIdle,
  sceneBadge,
  swarmCurve,
  swarmValid,
  cockpit,
  cofly,
  silent,
  outcome,
  onFreeGeom,
}: {
  frame: Frame | null
  scenario: Scenario
  playback: Playback | null
  day: boolean
  camMode: CamMode
  geometryOn: boolean
  /** кадр синтетический (обучение): сцена держит пусковую расстановку, панели живут */
  sceneIdle: boolean
  /** подпись поверх сцены: эпизод обучения / поколение роя / демо */
  sceneBadge?: string | null
  /** кривая обучения роя: наименьшее сближение по поколениям */
  swarmCurve?: number[] | null
  /** индексы валидационных поколений в кривой (где мерили эталонное трио) */
  swarmValid?: number[]
  /** пасхалка: муха за штурвалом (разрез корпуса + рычаги по командам DN) */
  cockpit?: boolean
  /** юмор-режим: кабина вскрывается и за первой мухой садится вторая-пассажирка
   *  (клон, ничего не крутит); палитра по полу напарника: 'm' — он, 'f' — она */
  cofly?: 'm' | 'f' | null
  /** беззвучный режим: рой и его подрежимы — короткие циклы, фразы не успевают */
  silent?: boolean
  /** итог завершённого живого прогона: взял или нет и наименьшее сближение (м).
   *  Нужен потому, что у проигрыша по истечении боевой жизни терминального события
   *  в кадре нет — без него финальная подпись зависала живой дистанцией («0 м») */
  outcome?: { hit: boolean; missM: number } | null
  /** «свободная расстановка»: перетаскивание цели прямо в сцене (до прогона) */
  onFreeGeom?: (patch: { free_tx: number; free_ty: number; free_talt: number }) => void
}) {
  const host = useRef<HTMLDivElement>(null)
  const chipRef = useRef<HTMLDivElement>(null)
  const frameRef = useRef(frame)
  const scRef = useRef(scenario)
  const playbackRef = useRef(playback)
  const dayRef = useRef(day)
  const camRef = useRef(camMode)
  const geoRef = useRef(geometryOn)
  const idleRef = useRef(sceneIdle)
  const cockpitOnRef = useRef(false)
  const coflyRef = useRef<'m' | 'f' | null>(null)
  const silentRef = useRef(false)
  const outcomeRef = useRef<{ hit: boolean; missM: number } | null>(outcome ?? null)
  const freeGeomRef = useRef(onFreeGeom)
  frameRef.current = frame
  scRef.current = scenario
  playbackRef.current = playback
  dayRef.current = day
  camRef.current = camMode
  geoRef.current = geometryOn
  idleRef.current = sceneIdle
  cockpitOnRef.current = cockpit ?? false
  coflyRef.current = cofly ?? null
  silentRef.current = silent ?? false
  outcomeRef.current = outcome ?? null
  freeGeomRef.current = onFreeGeom

  useEffect(() => {
    const el = host.current
    if (!el) return

    const home0 = homeView(scRef.current)
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(THEME_NIGHT.bg)
    scene.fog = new THREE.Fog(THEME_NIGHT.bg, home0.span * 3, home0.span * 10)

    const camera = new THREE.PerspectiveCamera(42, 1, 0.05, Math.max(400, home0.span * 20))
    camera.position.copy(home0.pos)

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setPixelRatio(Math.min(1.5, window.devicePixelRatio || 1))
    renderer.setClearColor(THEME_NIGHT.bg, 1)
    el.appendChild(renderer.domElement)

    const composer = new EffectComposer(renderer)
    composer.addPass(new RenderPass(scene, camera))
    // сила bloom — по тиру устройства (perfBudget().bloomStrength, 0.5 на low): дешёвый
    // пост-эффект держит GPU весь кадр, а не только в момент вспышки/искр
    const bloomBudget = perfBudget().bloomStrength
    const bloom = new UnrealBloomPass(new THREE.Vector2(800, 600), THEME_NIGHT.bloom * bloomBudget, 0.5, 0.85)
    composer.addPass(bloom)

    // ─── камера: авто-дистанция и перелёт при смене сценария ───
    let userZoom = 1 // относительный масштаб, заданный пользователем колесом
    let flight = false
    const flightPos = new THREE.Vector3()
    const flightTgt = new THREE.Vector3()
    let spanBase = home0.span
    let spanBaseWant = home0.span
    let lastScKey = scKey(scRef.current)
    let recenterUntil = 0
    let interacting = false
    let lastInteract = -1e9
    // ручное вращение во время прогона отдаёт камеру пользователю ДО конца
    // прогона (не на 1.5с, как раньше) — иначе терминальный наезд отбирал её
    // назад при первой же паузе в движении мыши, и покрутить сцену было нельзя
    let userFreeUntilNextRun = false

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.maxDistance = Math.max(40, home0.span * 8)
    controls.minDistance = 0.1 // можно приблизиться, чтобы рассмотреть зазор промаха
    controls.target.copy(home0.tgt)

    // двойной клик — плавно вернуть точку взгляда на действие, ракурс не меняется
    renderer.domElement.addEventListener('dblclick', () => {
      recenterUntil = performance.now() + 1200
    })
    controls.addEventListener('start', () => {
      interacting = true
      flight = false
      userFreeUntilNextRun = true
    })
    controls.addEventListener('end', () => {
      interacting = false
      lastInteract = performance.now()
    })

    const ambient = new THREE.AmbientLight(THEME_NIGHT.ambient, THEME_NIGHT.ambientI)
    scene.add(ambient)
    const sun = new THREE.DirectionalLight(0xffffff, THEME_NIGHT.sunI)
    sun.position.set(-6, 12, 8)
    scene.add(sun)

    let grid = new THREE.GridHelper(2, 8, THEME_NIGHT.grid1, THEME_NIGHT.grid2)
    scene.add(grid)
    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({ color: THEME_NIGHT.plane, transparent: true, opacity: 0.1, depthWrite: false, side: THREE.DoubleSide }),
    )
    plane.rotation.x = -Math.PI / 2
    scene.add(plane)

    const axes = new THREE.AxesHelper(1)
    scene.add(axes)
    // короткие подписи осей; легенда — в подсказке сцены
    const labXl = makeLabel('X', THEME_NIGHT.x, 0.9)
    const labYl = makeLabel('Y', THEME_NIGHT.y, 0.9)
    const labZl = makeLabel('Z', THEME_NIGHT.z, 0.9)
    const labX = labXl.spr
    const labY = labYl.spr
    const labZ = labZl.spr
    scene.add(labX, labY, labZ)

    const missile = new THREE.Group()
    const nose = new THREE.Mesh(
      new THREE.ConeGeometry(0.09, 0.38, 10),
      new THREE.MeshStandardMaterial({ color: 0xb8ffe0, emissive: 0x1a6048, emissiveIntensity: 0.9 }),
    )
    nose.rotation.x = Math.PI / 2
    const body = new THREE.Mesh(
      new THREE.CylinderGeometry(0.055, 0.08, 0.42, 10),
      new THREE.MeshStandardMaterial({ color: 0x7dffc8, emissive: 0x0b3a30 }),
    )
    body.rotation.x = Math.PI / 2
    body.position.z = -0.32
    missile.add(nose, body)
    scene.add(missile)

    // ─── пасхалка: муха за штурвалом (сидит на корпусе, задние лапки тянут рычаги) ───

    const cockpit = new THREE.Group()
    cockpit.position.set(0, 0.058, -0.1) // муха сидит сверху на корпусе
    cockpit.visible = false
    missile.add(cockpit)
    // весь «сектор кабины» (муха, рычаги, лапы) масштабируется целиком
    const rig = new THREE.Group()
    rig.scale.setScalar(3.5)
    cockpit.add(rig)
    // площадка с рычагами позади мухи: тангаж (фосфор) и рыскание (янтарь)
    const desk = new THREE.Mesh(
      new THREE.BoxGeometry(0.09, 0.004, 0.045),
      new THREE.MeshStandardMaterial({ color: 0x14332c, emissive: 0x0b3a30 }),
    )
    desk.position.set(0, 0.002, -0.06)
    rig.add(desk)
    // рычаги позади мухи: тангаж (фосфор) и рыскание (янтарь) — их тянут задние лапки
    const mkLever = (x: number, col: number) => {
      const base = new THREE.Group()
      base.position.set(x, 0.004, -0.06)
      const stick = new THREE.Mesh(
        new THREE.CylinderGeometry(0.004, 0.004, 0.05, 6),
        new THREE.MeshStandardMaterial({ color: col, emissive: col, emissiveIntensity: 0.7 }),
      )
      stick.position.y = 0.025
      const knob = new THREE.Mesh(
        new THREE.SphereGeometry(0.0095, 8, 6),
        new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: col, emissiveIntensity: 0.9 }),
      )
      knob.position.y = 0.05
      base.add(stick, knob)
      rig.add(base)
      return base
    }
    const leverPitch = mkLever(-0.026, 0x7dffc8)
    const leverYaw = mkLever(0.026, 0xe7c15a)

    // ── муха: голова с глазами и хоботком, грудь, полосатое брюшко, крылья, шесть лап ──
    const flyGrp = new THREE.Group()
    flyGrp.position.set(0, 0.008, 0.055)
    flyGrp.scale.setScalar(2)
    rig.add(flyGrp)
    const chitin = new THREE.MeshStandardMaterial({ color: 0x3f6f5f, emissive: 0x1d4438, emissiveIntensity: 1, roughness: 0.45 })
    const glow = new THREE.MeshStandardMaterial({ color: 0xd2ffe9, emissive: 0x1a6048, emissiveIntensity: 1.1 })
    const eyeMat = new THREE.MeshStandardMaterial({ color: 0xff6b6b, emissive: 0x7a2015, emissiveIntensity: 1.3 })
    const legMat = new THREE.MeshStandardMaterial({ color: 0x35594d, emissive: 0x1d4438, emissiveIntensity: 0.9 })
    // брюшко из трёх сегментов
    const abdomen = new THREE.Group()
    abdomen.position.z = -0.024
    for (let i = 0; i < 3; i++) {
      const seg = new THREE.Mesh(new THREE.SphereGeometry(0.0115 - i * 0.002, 8, 6), i % 2 ? chitin : glow)
      seg.scale.set(1, 0.85, 1.2)
      seg.position.z = -i * 0.0135
      abdomen.add(seg)
    }
    flyGrp.add(abdomen)
    // грудь
    const thorax = new THREE.Mesh(new THREE.SphereGeometry(0.0125, 10, 8), chitin)
    thorax.scale.set(1, 0.9, 1.25)
    flyGrp.add(thorax)
    // голова: составные глаза + хоботок
    const head = new THREE.Group()
    head.position.z = 0.021
    head.add(new THREE.Mesh(new THREE.SphereGeometry(0.0088, 10, 8), chitin))
    const eyeGeo = new THREE.SphereGeometry(0.0058, 8, 6)
    ;[-0.0052, 0.0052].forEach((x) => {
      const e = new THREE.Mesh(eyeGeo, eyeMat)
      e.position.set(x, 0.002, 0.0042)
      head.add(e)
    })
    const proboscis = new THREE.Mesh(new THREE.ConeGeometry(0.0018, 0.009, 6), glow)
    proboscis.rotation.x = Math.PI / 2
    proboscis.position.z = 0.013
    head.add(proboscis)
    // усики
    ;[-1, 1].forEach((s) => {
      const ant = new THREE.Mesh(new THREE.CylinderGeometry(0.0009, 0.0014, 0.012, 4), glow)
      ant.position.set(s * 0.0035, 0.006, 0.006)
      ant.rotation.z = -s * 0.5
      ant.rotation.x = 0.5
      head.add(ant)
    })
    flyGrp.add(head)
    // крылья (эллипс)
    const wingMat = new THREE.MeshBasicMaterial({ color: 0xbfeee0, transparent: true, opacity: 0.38, side: THREE.DoubleSide, depthWrite: false })
    const wingGeo = new THREE.CircleGeometry(0.016, 10)
    wingGeo.scale(0.62, 1.9, 1)
    wingGeo.translate(0, 0.016, 0)
    const wingL = new THREE.Mesh(wingGeo, wingMat)
    wingL.position.set(-0.009, 0.013, -0.005)
    wingL.rotation.x = -0.45
    const wingR = new THREE.Mesh(wingGeo, wingMat)
    wingR.position.set(0.009, 0.013, -0.005)
    wingR.rotation.x = -0.45
    flyGrp.add(wingL, wingR)
    // лапы: цилиндры единичной длины, растягиваются между суставами
    const legCyl = (r: number) => {
      const g = new THREE.CylinderGeometry(r, r * 0.75, 1, 5)
      g.translate(0, 0.5, 0)
      return g
    }
    const YAX = new THREE.Vector3(0, 1, 0)
    const stretch = (seg: THREE.Mesh, a: THREE.Vector3, b: THREE.Vector3) => {
      const dir = b.clone().sub(a)
      const len = Math.max(dir.length(), 1e-4)
      seg.scale.set(1, len, 1)
      seg.position.copy(a)
      seg.quaternion.setFromUnitVectors(YAX, dir.divideScalar(len))
    }
    type Leg = { hip: THREE.Vector3; foot: THREE.Vector3; upper: THREE.Mesh; knee: THREE.Mesh; lower: THREE.Mesh; tip: THREE.Mesh }
    const mkLeg = (hip: THREE.Vector3, foot: THREE.Vector3): Leg => {
      const upper = new THREE.Mesh(legCyl(0.0018), legMat)
      const lower = new THREE.Mesh(legCyl(0.0013), legMat)
      const knee = new THREE.Mesh(new THREE.SphereGeometry(0.003, 6, 5), legMat)
      const tip = new THREE.Mesh(new THREE.SphereGeometry(0.0028, 6, 5), glow)
      rig.add(upper, knee, lower, tip)
      return { hip: hip.clone(), foot: foot.clone(), upper, knee, lower, tip }
    }
    const legLenA = 0.03
    const legLenB = 0.033
    // двухкостная лапа: бедро + голень, «колено» выгнуто вверх
    const solveLeg = (leg: Leg, footOverride?: THREE.Vector3) => {
      const foot = footOverride ?? leg.foot
      const dirF = foot.clone().sub(leg.hip)
      let d = dirF.length()
      const dMax = (legLenA + legLenB) * 0.98
      if (d > dMax) {
        dirF.multiplyScalar(dMax / d)
        d = dMax
      }
      if (d < 1e-4) d = 1e-4
      const F = leg.hip.clone().add(dirF)
      const nd = dirF.normalize()
      const h = Math.sqrt(Math.max(legLenA * legLenA - (d / 2) * (d / 2), 1e-6))
      const K = leg.hip.clone().addScaledVector(nd, d / 2).add(new THREE.Vector3(0, h, 0))
      stretch(leg.upper, leg.hip, K)
      leg.knee.position.copy(K)
      stretch(leg.lower, K, F)
      leg.tip.position.copy(F)
    }
    // задние — до рычагов, средние и передние — опорные на корпусе
    const hindL = mkLeg(new THREE.Vector3(-0.012, 0.014, -0.028), new THREE.Vector3(-0.026, 0.056, -0.06))
    const hindR = mkLeg(new THREE.Vector3(0.012, 0.014, -0.028), new THREE.Vector3(0.026, 0.056, -0.06))
    const midL = mkLeg(new THREE.Vector3(-0.015, 0.002, 0.022), new THREE.Vector3(-0.028, 0.004, 0.022))
    const midR = mkLeg(new THREE.Vector3(0.015, 0.002, 0.022), new THREE.Vector3(0.028, 0.004, 0.022))
    const foreL = mkLeg(new THREE.Vector3(-0.011, 0.002, 0.07), new THREE.Vector3(-0.02, 0.004, 0.092))
    const foreR = mkLeg(new THREE.Vector3(0.011, 0.002, 0.07), new THREE.Vector3(0.02, 0.004, 0.092))
    // «Zzz» — муха спит, пока нет полёта
    const zzz = makeLabel('Zzz', '#9fc3ba', 0.07)
    zzz.spr.position.set(0.06, 0.09, -0.02)
    zzz.spr.visible = false
    rig.add(zzz.spr)
    let flapPhase = 0

    // ── юмор-режим: вторая муха-штурман в тандеме за спиной первой ──
    // клон готовой mesh-мухи (лапы клон не наследует — они висят на rig),
    // свой хитин по полу: янтарная самка / синеватый самец; взмах — в противофазе.
    // Пассажирка: к рычагам не касается, траекторию не считает.
    const mkCofly = (which: 'm' | 'f') => {
      const g = flyGrp.clone(true)
      g.name = `cofly_${which}`
      const male = which === 'm'
      const chitinC = new THREE.MeshStandardMaterial({
        color: male ? 0x4f6f8a : 0x8a6f3f,
        emissive: male ? 0x22354a : 0x44351a,
        emissiveIntensity: 1,
        roughness: 0.45,
      })
      const glowC = new THREE.MeshStandardMaterial({
        color: male ? 0xd2ecff : 0xfff0d2,
        emissive: male ? 0x1a4860 : 0x604818,
        emissiveIntensity: 1.1,
      })
      g.traverse((o) => {
        const mesh = o as THREE.Mesh
        if (!mesh.isMesh) return
        const src = mesh.material as THREE.Material
        if (src === chitin) mesh.material = chitinC
        else if (src === glow) mesh.material = glowC
        else if (src === legMat) mesh.material = chitinC
      })
      g.scale.setScalar(1.7) // чуть меньше первой — сидит в тандемном кресле
      g.visible = false
      rig.add(g)
      g.position.set(0.032, 0.004, -0.078) // сзади и чуть вбок: рычаги и лапы первой не загораживает
      const wings = g.children.filter((c) => (c as THREE.Mesh).isMesh && (c as THREE.Mesh).material === wingMat)
      return { grp: g, wingL: wings[0] as THREE.Mesh, wingR: wings[1] as THREE.Mesh }
    }
    const coflyF = mkCofly('f')
    const coflyM = mkCofly('m')

    // демо-гонка: вторая ракета со своей (необученной) мухой — клон каркаса
    flyGrp.name = 'fly'
    const rocketR = missile.clone(true)
    rocketR.visible = false
    rocketR.traverse((o) => {
      const m = o as THREE.Mesh
      if (m.isMesh) {
        m.material = (m.material as THREE.Material).clone()
        ;(m.material as THREE.MeshStandardMaterial).color = new THREE.Color(0xffc46b)
        ;(m.material as THREE.MeshStandardMaterial).emissive = new THREE.Color(0x7a4a10)
      }
    })
    scene.add(rocketR)
    // подписи ракет в гонке
    const mkRaceLabel = (color: string) => {
      const l = makeLabel('', color, 0.55)
      l.spr.visible = false
      scene.add(l.spr)
      return l
    }
    const raceLabelFly = { ...mkRaceLabel('#d2ffe9'), text: '' }
    const raceLabelOther = { ...mkRaceLabel('#ffc46b'), text: '' }
    let raceStack = 0 // текущий подъём «чужой» подписи при наложении (сглаженный)

    const target = buildJet()
    scene.add(target)

    // обломки самолёта: части модели разлетаются от эпицентра подрыва
    type JetPart = { mesh: THREE.Mesh; home: THREE.Vector3; homeRot: THREE.Euler; vel: THREE.Vector3; spin: THREE.Vector3 }
    const jetParts: JetPart[] = []
    target.traverse((o) => {
      const m = o as THREE.Mesh
      if (m.isMesh) jetParts.push({ mesh: m, home: m.position.clone(), homeRot: m.rotation.clone(), vel: new THREE.Vector3(), spin: new THREE.Vector3() })
    })
    const jetMats = (target.userData.mats ?? []) as THREE.MeshStandardMaterial[]
    // базовая яркость материалов — чтобы короткую вспышку подрыва можно было отпустить обратно
    const jetMatsBaseEmissive = jetMats.map((m) => m.emissiveIntensity)

    // искры при подрыве: короткий Points-всплеск с тем же затуханием, что у обломков
    // (age 0…1.1с), число частиц — по тиру устройства (perfBudget().sparkCount, 0 на low)
    const SPARK_MAX = 18
    const sparkPos = new Float32Array(SPARK_MAX * 3)
    const sparkGeo = new THREE.BufferGeometry()
    sparkGeo.setAttribute('position', new THREE.BufferAttribute(sparkPos, 3).setUsage(THREE.DynamicDrawUsage))
    const sparkMat = new THREE.PointsMaterial({ color: 0xffe0a0, size: 0.045, transparent: true, opacity: 0, depthWrite: false })
    const sparks = new THREE.Points(sparkGeo, sparkMat)
    sparks.visible = false
    scene.add(sparks)
    const sparkVel: THREE.Vector3[] = Array.from({ length: SPARK_MAX }, () => new THREE.Vector3())
    let activeSparks = 0

    // ── ручная расстановка прямо в сцене (аспект free, до прогона) ──
    // тяга цели: по горизонтальной плоскости её высоты — X/Y; с зажатым Shift — вертикально, высота.
    // невидимая сфера-хитбокс щедро ловит клик вокруг самолёта; rehoming камеры на время тяги глушится
    const grab = new THREE.Mesh(new THREE.SphereGeometry(0.45, 10, 10), new THREE.MeshBasicMaterial({ visible: false }))
    target.add(grab)
    const ray = new THREE.Raycaster()
    const ndcV = new THREE.Vector2()
    let freeDrag: { shift: boolean } | null = null
    const snap50 = (m: number) => Math.round(m / 50) * 50
    const canGrab = () => scRef.current.aspect === 'free' && !playbackRef.current && (!frameRef.current || idleRef.current)
    const setRay = (e: PointerEvent) => {
      const r = renderer.domElement.getBoundingClientRect()
      ray.setFromCamera(ndcV.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera)
    }
    const grabDown = (e: PointerEvent) => {
      if (e.button !== 0 || !canGrab()) return
      setRay(e)
      if (ray.intersectObject(grab, false).length === 0) return
      freeDrag = { shift: e.shiftKey }
      controls.enabled = false
      renderer.domElement.setPointerCapture(e.pointerId)
      renderer.domElement.style.cursor = 'grabbing'
      buzz(HAPTIC.grab)
      e.stopPropagation()
    }
    const grabMove = (e: PointerEvent) => {
      if (!freeDrag) {
        if (canGrab()) {
          setRay(e)
          renderer.domElement.style.cursor = ray.intersectObject(grab, false).length > 0 ? 'grab' : ''
        }
        return
      }
      const sc0 = scRef.current
      freeDrag.shift = e.shiftKey
      if (e.shiftKey) {
        const r = renderer.domElement.getBoundingClientRect()
        const dist = camera.position.distanceTo(target.position)
        const dhM = (-e.movementY * (2 * Math.tan(((camera.fov * Math.PI) / 180) / 2) * dist)) / r.height / KM
        freeGeomRef.current?.({ free_tx: sc0.free_tx, free_ty: sc0.free_ty, free_talt: Math.max(0, snap50(sc0.free_talt + dhM)) })
        return
      }
      setRay(e)
      const d = ray.ray.direction
      if (Math.abs(d.y) < 1e-6) return
      const t = (target.position.y - ray.ray.origin.y) / d.y
      if (t <= 0) return
      freeGeomRef.current?.({
        free_tx: snap50((ray.ray.origin.x + d.x * t) / KM),
        free_ty: snap50((ray.ray.origin.z + d.z * t) / KM),
        free_talt: sc0.free_talt,
      })
    }
    const grabUp = () => {
      if (!freeDrag) return
      freeDrag = null
      controls.enabled = true
      renderer.domElement.style.cursor = ''
    }
    renderer.domElement.addEventListener('pointerdown', grabDown, true)
    renderer.domElement.addEventListener('pointermove', grabMove)
    renderer.domElement.addEventListener('pointerup', grabUp)
    renderer.domElement.addEventListener('pointercancel', grabUp)

    const los = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineDashedMaterial({ color: THEME_NIGHT.los, dashSize: 0.25, gapSize: 0.12 }))
    scene.add(los)
    // цвет следа — по вершинам (см. setTrailGeometry): материал держит белый, чтобы не искажать градиент
    const mLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffffff, vertexColors: true }))
    const tLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffffff, vertexColors: true }))
    scene.add(mLine, tLine)

    // ─── геометрия наведения ───
    const geoGroup = new THREE.Group()
    scene.add(geoGroup)
    const reqLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: THEME_NIGHT.req, transparent: true, opacity: 0.9 }))
    const tgtToPip = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineDashedMaterial({ color: THEME_NIGHT.tgt, dashSize: 0.1, gapSize: 0.06, transparent: true, opacity: 0.8 }))
    const aArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), 0.5, THEME_NIGHT.arrow, 0.1, 0.05)
    // круг реального радиуса срабатывания БЧ (sc.kill_radius_m) — только у настоящей позиции
    // цели, только в режиме «геометрия»; не декоративное кольцо у прогнозной точки
    const killRing = new THREE.LineLoop(unitCircleGeometry(40), new THREE.LineBasicMaterial({ color: THEME_NIGHT.y, transparent: true, opacity: 0.8 }))
    let lastKillR = -1
    const killLabel = makeLabel('', THEME_NIGHT.y, 0.55)
    geoGroup.add(reqLine, tgtToPip, aArrow, killRing, killLabel.spr)
    // стрелка перегрузки цели: куда самолёт тянет манёвр. Живёт вне geo-режима —
    // манёвр должен быть виден в любом режиме камеры и панелей
    const tArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), 0.5, THEME_NIGHT.tgt, 0.12, 0.06)
    tArrow.visible = false
    scene.add(tArrow)

    let wasHit = false
    let hitAt = 0
    const partRng = (() => {
      let a = 991
      return () => {
        a = (a * 16807) % 2147483647
        return a / 2147483647
      }
    })()
    const swarmCenter = new THREE.Vector3()
    let swarmSpan = 0 // размах веера траекторий роя

    let mTrail: TrailPoint[] = []
    let tTrail: TrailPoint[] = []
    const trailCold = new THREE.Color()
    const trailHot = new THREE.Color()
    let modelScale = 0.75
    let effScale = 0.75 // сглаженный текущий масштаб (в терминальной фазе уменьшается)
    const missLabel = makeLabel('промах', '#ff6a4a', 0.5)
    missLabel.spr.visible = false
    scene.add(missLabel.spr)
    const distLabel = makeLabel('0.00', '#e7c15a', 0.7)
    distLabel.spr.visible = false
    scene.add(distLabel.spr)
    let lastStamp = -2
    let intro = 0
    let gridSpan = 0.6
    const anchor = new THREE.Vector3(2, 4, 0)
    let gridSpanWant = 8
    /** Сетка ставится один раз на прогон: центр фиксируется в мире, шаг клеток «красивый»
     * (ряд 1–2–2.5–5). Если же сетка каждый кадр доводится к живому расстоянию пары,
     * она весь полёт «дышит» и ползает — и разом пропадает ощущение движения. */
    const placeGrid = (center: THREE.Vector3, spanWorld: number) => {
      const cells = [0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20]
      const wantCells = Math.max(spanWorld * 1.25, 4) / 8
      gridSpanWant = (cells.find((c) => c >= wantCells) ?? 20) * 8
      anchor.copy(center)
    }
    // сглаженная телеметрия: кадры приходят ~25 Гц, рендер — 60–120 Гц; без интерполяции
    // модели движутся ступеньками и «дрожат»
    const smM = new THREE.Vector3()
    const smT = new THREE.Vector3()
    const smMv = new THREE.Vector3()
    const smTv = new THREE.Vector3()
    const smA = new THREE.Vector3()
    const smRp = new THREE.Vector3()
    const smRv = new THREE.Vector3()
    const qAim = new THREE.Quaternion()
    let smInit = false
    let smRInit = false
    let lastTickMs = performance.now()
    // самокалибровка тира устройства: средний интервал первых 60 рендер-кадров
    // оседает в localStorage (perf.ts::recordFrameMs) и учитывается со следующей сессии
    let calibFrames = 0
    let calibSumMs = 0
    let calibDone = false
    let lastTrailStamp = -2
    let wasInRun = false
    let terminalCam = false
    // манёвр из кинематики: поперечное ускорение = производная сглаженной скорости
    // минус продольная составляющая. Из него — крен самолёта/ракеты и стрелка цели
    const smTa = new THREE.Vector3()
    const prevTv = new THREE.Vector3()
    const qRoll = new THREE.Quaternion()
    const Z_AXIS = new THREE.Vector3(0, 0, 1)
    const G_KM = 0.00981 // g в единицах сцены (км/с²)
    let bankT = 0 // визуальный крен цели, рад
    let bankM = 0 // визуальный крен ракеты, рад

    // линии роя: перестраиваются при смене поколения
    let shownPlayback: Playback | null = null
    type SwarmLine = { line: THREE.Line; total: number; hit: boolean; best: boolean; k: number }
    let swarmLines: SwarmLine[] = []
    let swarmLabels: THREE.Sprite[] = []
    let tgtTrack: THREE.Line | null = null

    const disposeSwarm = () => {
      swarmLines.forEach((s) => {
        scene.remove(s.line)
        s.line.geometry.dispose()
        ;(s.line.material as THREE.Material).dispose()
      })
      swarmLines = []
      swarmLabels.forEach((spr) => {
        spr.removeFromParent()
        const m = spr.material as THREE.SpriteMaterial
        m.map?.dispose()
        m.dispose()
      })
      swarmLabels = []
      if (tgtTrack) {
        scene.remove(tgtTrack)
        tgtTrack.geometry.dispose()
        ;(tgtTrack.material as THREE.Material).dispose()
        tgtTrack = null
      }
    }

    let appliedDay = false
    const restoreJet = () => {
      jetParts.forEach((p) => {
        p.mesh.position.copy(p.home)
        p.mesh.rotation.copy(p.homeRot)
      })
      jetMats.forEach((m) => (m.opacity = 1))
    }
    const applyTheme = (isDay: boolean) => {
      const T = isDay ? THEME_DAY : THEME_NIGHT
      ;(scene.background as THREE.Color).set(T.bg)
      ;(scene.fog as THREE.Fog).color.set(T.bg)
      scene.remove(grid)
      grid.geometry.dispose()
      ;(grid.material as THREE.Material).dispose()
      grid = new THREE.GridHelper(2, 8, T.grid1, T.grid2)
      scene.add(grid)
      ;(plane.material as THREE.MeshBasicMaterial).color.set(T.plane)
      ambient.color.set(T.ambient)
      ambient.intensity = T.ambientI
      sun.intensity = T.sunI
      labXl.set(T.x)
      labYl.set(T.y)
      labZl.set(T.z)
      ;(los.material as THREE.LineDashedMaterial).color.set(T.los)
      // mLine/tLine: цвет по вершинам (см. setTrailGeometry) — материал остаётся белым,
      // новая тема подхватывается следующим пушем точки, старый след не перекрашивается
      if (mTrail.length > 1) setTrailGeometry(mLine, mTrail, trailCold.set(T.mLine), trailHot.set(T.mLineHot))
      if (tTrail.length > 1) setTrailGeometry(tLine, tTrail, trailCold.set(T.tLine), trailHot.set(T.tLineHot))
      ;(reqLine.material as THREE.LineBasicMaterial).color.set(T.req)
      ;(tgtToPip.material as THREE.LineDashedMaterial).color.set(T.tgt)
      aArrow.setColor(new THREE.Color(T.arrow))
      ;(killRing.material as THREE.LineBasicMaterial).color.set(T.y)
      lastKillR = -1 // форсируем перерисовку подписи круга БЧ новым цветом темы на следующем кадре
      tArrow.setColor(new THREE.Color(T.tgt))
      bloom.strength = T.bloom * bloomBudget
      swarmLines.forEach((s) => {
        const col = s.best
          ? new THREE.Color(T.swarmBest)
          : new THREE.Color(T.swarmBad).lerp(new THREE.Color(T.swarmGood), s.k)
        ;(s.line.material as THREE.LineBasicMaterial).color.copy(col)
      })
    }

    const fit = () => {
      const w = Math.max(el.clientWidth, 16)
      const h = Math.max(el.clientHeight, 16)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h, false)
      composer.setSize(w, h)
    }
    fit()
    const ro = new ResizeObserver(fit)
    ro.observe(el)

    let raf = 0
    const tick = () => {
      if (dayRef.current !== appliedDay) {
        appliedDay = dayRef.current
        applyTheme(appliedDay)
      }
      const fr = frameRef.current
      const sc = scRef.current
      const pb = playbackRef.current

      let mp: THREE.Vector3
      let tp: THREE.Vector3
      let mv: THREE.Vector3
      let tv: THREE.Vector3
      let stamp: number
      let event: string | null = null
      let aCmd: THREE.Vector3 | null = null
      let prog = 0
      // веер траекторий: взял ли лучший бой и каково в нём наименьшее сближение —
      // у проигрыша терминального события нет, а финальная подпись обязана быть
      let bestHit: boolean | null = null
      let bestMiss: number | null = null
      let rp: THREE.Vector3 | null = null // позиция ракеты-стажёра в демо-гонке

      if (pb && pb.results.length > 0) {
        const best = pb.results[Math.min(pb.bestIdx, pb.results.length - 1)]
        prog = Math.min(1, Math.max(0, (performance.now() - pb.startedAt) / pb.durationMs))
        if (pb.race === true && pb.results.length >= 2) {
          // вторая ракета — та, что не является лидером кадра
          rp = to3(sampleTraj(pb.results[pb.bestIdx === 0 ? 1 : 0].traj_m, prog))
        }
        mp = to3(sampleTraj(best.traj_m, prog))
        tp = to3(sampleTraj(best.traj_t, prog))
        // камера держит центр всего веера траекторий, а не скользит за лидером
        swarmCenter.set(0, 0, 0)
        let cn = 0
        for (let i = 0; i < best.traj_m.length; i += 8) {
          swarmCenter.add(to3(best.traj_m[i]))
          cn += 1
        }
        if (cn > 0) {
          swarmCenter.divideScalar(cn)
          let rMax = 0
          for (let i = 0; i < best.traj_m.length; i += 8) rMax = Math.max(rMax, to3(best.traj_m[i]).distanceTo(swarmCenter))
          swarmSpan = rMax * 2
        }
        mv = to3(velAt(best.traj_m, prog))
        tv = to3(velAt(best.traj_t, prog))
        stamp = pb.startedAt
        bestHit = best.hit
        bestMiss = best.miss_m ?? null
        if (best.hit && prog >= 1) event = 'перехват'
      } else if (fr && !idleRef.current) {
        mp = to3(fr.missile)
        tp = to3(fr.target)
        mv = to3(fr.missile_v)
        tv = to3(fr.target_v)
        aCmd = to3(fr.a_cmd)
        stamp = fr.t
        event = fr.event
      } else {
        const pair = spawnPair(sc)
        mp = to3(pair.m)
        tp = to3(pair.t)
        mv = to3([sc.v_m, 0, 0])
        tv = to3([0, 0, 0])
        stamp = -1
      }

      // новый прогон или простой: собрать самолёт обратно
      const freshRun = stamp < 0 || stamp < 0.05 || stamp < lastStamp
      if (freshRun) {
        mTrail = []
        tTrail = []
        intro = 0
        lastTrailStamp = -2
        restoreJet()
        missLabel.spr.visible = false
        const home = homeView(scRef.current)
        modelScale = home.modelScale
        effScale = modelScale
        const key = scKey(scRef.current)
        if (key !== lastScKey) {
          lastScKey = key
          // во время ручной тяги цели камера не перелетает: ключ молча догоняем, джампа нет
          if (!freeDrag) {
            spanBaseWant = home.span
            flightPos.copy(home.pos)
            flightTgt.copy(home.tgt)
            flight = camRef.current !== 'chase'
            userZoom = 1
            recenterUntil = 0
          }
        }
      }
      lastStamp = stamp

      // телеметрия приходит реже, чем кадры рендера: экспоненциальное сглаживание
      // (τ≈70 мс) превращает ступеньки в непрерывный ход; на старте прогона — мгновенный щелчок
      const nowMs = performance.now()
      const rawDtMs = nowMs - lastTickMs
      if (!calibDone) {
        calibSumMs += rawDtMs
        calibFrames += 1
        if (calibFrames >= 60) {
          recordFrameMs(calibSumMs / calibFrames)
          calibDone = true
        }
      }
      const dtS = Math.min(0.1, Math.max(0.001, rawDtMs / 1000))
      lastTickMs = nowMs
      const kPos = 1 - Math.exp(-dtS / 0.07)
      const kRot = 1 - Math.exp(-dtS / 0.05)
      const snap = !smInit || freshRun
      if (snap) {
        smM.copy(mp)
        smT.copy(tp)
        smMv.copy(mv)
        smTv.copy(tv)
        smA.copy(aCmd ?? new THREE.Vector3())
        prevTv.copy(tv)
        smTa.set(0, 0, 0)
        bankT = 0
        bankM = 0
        smInit = true
        // сетка нового прогона — по фактическим стартовым точкам пары: сценарий прогона
        // может быть шире базового (рой семплирует дальности)
        placeGrid(mp.clone().lerp(tp, 0.5), mp.distanceTo(tp))
      } else {
        smM.lerp(mp, kPos)
        smT.lerp(tp, kPos)
        smMv.lerp(mv, kPos)
        smTv.lerp(tv, kPos)
        if (aCmd) smA.lerp(aCmd, kPos)
      }
      mp = smM
      tp = smT
      mv = smMv
      tv = smTv

      // манёвр цели: поперечное ускорение из производной сглаженной скорости.
      // Из него считаются крен самолёта и стрелка перегрузки — без них вираж
      // на широкой сцене не отличить от прямолинейного полёта
      if (!snap && tv.lengthSq() > 1e-9 && prevTv.lengthSq() > 1e-9 && dtS > 0) {
        const aRaw = tv.clone().sub(prevTv).divideScalar(dtS)
        const fwdT = tv.clone().normalize()
        aRaw.addScaledVector(fwdT, -aRaw.dot(fwdT))
        smTa.lerp(aRaw, 1 - Math.exp(-dtS / 0.25))
        const rightT = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), fwdT)
        if (rightT.lengthSq() > 1e-6) {
          const rollWant = Math.max(-1.3, Math.min(1.3, -Math.atan2(smTa.dot(rightT.normalize()), G_KM)))
          bankT += (rollWant - bankT) * (1 - Math.exp(-dtS / 0.3))
        }
      }
      prevTv.copy(tv)
      // крен ракеты — по её сглаженной команде управления
      if (smA.lengthSq() > 1e-9 && mv.lengthSq() > 1e-9) {
        const fwdM = mv.clone().normalize()
        const aM = smA.clone().addScaledVector(fwdM, -smA.dot(fwdM))
        const rightM = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), fwdM)
        if (rightM.lengthSq() > 1e-6) {
          const rollWantM = Math.max(-1.3, Math.min(1.3, -Math.atan2(aM.dot(rightM.normalize()), G_KM)))
          bankM += (rollWantM - bankM) * (1 - Math.exp(-dtS / 0.3))
        }
      }

      // масштаб среды (туман, горизонт, лимиты масштаба) плывёт вслед за масштабом сценария
      spanBase += (spanBaseWant - spanBase) * 0.04
      const fog = scene.fog as THREE.Fog
      fog.near = spanBase * 3
      fog.far = spanBase * 10
      const wantFar = Math.max(400, spanBase * 20)
      if (Math.abs(camera.far - wantFar) / wantFar > 0.02) {
        camera.far = wantFar
        camera.updateProjectionMatrix()
      }
      controls.maxDistance = Math.max(40, spanBase * 8)

      const span = Math.max(mp.distanceTo(tp), 1.2)
      const sepRaw = mp.distanceTo(tp) // без клампа — для терминальной фазы и подписи
      intro = Math.min(1, intro + 0.012)
      // размер и центр сетки заданы на весь прогон (placeGrid); здесь она лишь плавно
      // выходит на цель после пуска — и замирает, давая миру неподвижную опору.
      // На простое (до пуска) сетка показывается сразу полной — по дистанции сценария:
      // иначе каждый кадр простоя считается «новым прогоном» и анимация ввода
      // обнуляется, оставляя сетку на доле целевого размера.
      const gridIntro = stamp < 0 ? 1 : 0.15 + 0.85 * intro
      gridSpan += (gridSpanWant * gridIntro - gridSpan) * 0.08
      const s = Math.max(gridSpan, 0.5)
      grid.scale.set(s / 2, 1, s / 2)
      plane.scale.set(s / 2, s / 2, 1)
      grid.position.copy(anchor)
      plane.position.set(anchor.x, anchor.y - 0.01, anchor.z)

      axes.position.copy(anchor)
      axes.scale.setScalar(Math.min(s * 0.35, 4))
      const showAxisLabels = span < spanBase * 1.2
      labX.visible = showAxisLabels
      labY.visible = showAxisLabels
      labZ.visible = showAxisLabels
      labX.position.set(anchor.x - Math.min(s * 0.38, 4.2), anchor.y + 0.15, anchor.z)
      labZ.position.set(anchor.x, anchor.y + Math.min(s * 0.32, 3.4), anchor.z)
      labY.position.set(anchor.x, anchor.y + 0.15, anchor.z + Math.min(s * 0.32, 3.4))

      missile.position.copy(mp)
      target.position.copy(tp)
      // нос — по вектору скорости, крен — по горизонту: на горке/пике самолёт не ложится боком.
      // Доворот через slerp, а не мгновенная установка: команда наведения дрожит от шага к шагу.
      // Крен подмешивается в ЦЕЛЕВОЙ кватернион до slerp — если применять его после,
      // он накапливается от кадра к кадру (целевой угол делится на k) и модель крутится юлой
      const aimAlong = (obj: THREE.Object3D, v: THREE.Vector3, k: number, roll = 0) => {
        if (v.lengthSq() < 1e-6) return
        const fwd = v.clone().normalize()
        const right = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), fwd)
        if (right.lengthSq() < 1e-6) right.set(1, 0, 0)
        right.normalize()
        const up = new THREE.Vector3().crossVectors(fwd, right).normalize()
        qAim.setFromRotationMatrix(new THREE.Matrix4().makeBasis(right, up, fwd))
        if (roll !== 0) qAim.multiply(qRoll.setFromAxisAngle(Z_AXIS, roll))
        if (k >= 1) obj.quaternion.copy(qAim)
        else obj.quaternion.slerp(qAim, k)
      }
      aimAlong(missile, mv, snap ? 1 : kRot, bankM)
      aimAlong(target, tv, snap ? 1 : kRot, bankT)
      // базовый масштаб держится весь прогон, но в терминальной фазе модели
      // плавно уменьшаются вместе с зазором — иначе «390 м» выглядит как контакт
      const rawScale = Math.min(modelScale, Math.max(0.15, sepRaw * 0.55))
      effScale += (rawScale - effScale) * 0.1
      missile.scale.setScalar(effScale)
      target.scale.setScalar(effScale)
      // демо-гонка: ракета-стажёр летит параллельно ветерану по своей траектории
      if (rp) {
        const pbR = pb as Playback
        const rv = to3(velAt(pbR.results[pbR.bestIdx === 0 ? 1 : 0].traj_m, prog))
        if (!smRInit || snap) {
          smRp.copy(rp)
          smRv.copy(rv)
          smRInit = true
        } else {
          smRp.lerp(rp, kPos)
          smRv.lerp(rv, kPos)
        }
        rocketR.visible = true
        rocketR.position.copy(smRp)
        // ориентация — обязательно по СВОЕМУ вектору скорости: без этого клон
        // летит «замороженным» — боком и задом, как только траектория поворачивает
        aimAlong(rocketR, smRv, snap ? 1 : kRot)
        rocketR.scale.setScalar(effScale)
        const iBest = pbR.bestIdx
        const iOther = pbR.bestIdx === 0 ? 1 : 0
        // ракеты сошлись — живые подписи налагаются. Наложение проверяем в координатах
        // кадра (NDC) по БАЗОВЫМ позициям, а «чужую» подпись плавно поднимаем над лидером
        const baseFly = mp.clone().add(new THREE.Vector3(0, 0.32, 0))
        const baseOther = rp.clone().add(new THREE.Vector3(0, 0.32, 0))
        const ndcA = baseFly.clone().project(camera)
        const ndcB = baseOther.clone().project(camera)
        const overlapped = Math.abs(ndcA.x - ndcB.x) < 0.12 && Math.abs(ndcA.y - ndcB.y) < 0.045
        raceStack += ((overlapped ? 0.62 : 0) - raceStack) * 0.18
        raceLabelFly.spr.position.copy(baseFly)
        raceLabelOther.spr.position.copy(baseOther).add(new THREE.Vector3(0, raceStack, 0))
        raceLabelFly.spr.visible = true
        raceLabelOther.spr.visible = true
        const lblFly = pbR.results[iBest]?.label ?? ''
        const lblOther = pbR.results[iOther]?.label ?? ''
        if (raceLabelFly.text !== lblFly) {
          raceLabelFly.set('#d2ffe9', lblFly)
          raceLabelFly.text = lblFly
        }
        if (raceLabelOther.text !== lblOther) {
          raceLabelOther.set('#ffc46b', lblOther)
          raceLabelOther.text = lblOther
        }
      } else {
        rocketR.visible = false
        smRInit = false
        raceLabelFly.spr.visible = false
        raceLabelOther.spr.visible = false
      }

      // пасхалка: муха за штурвалом — задние лапки тянут рычаги реальными командами DN.
      // юмор-режим (cofly) вскрывает кабину принудительно: вторая муха должна быть видна
      const cof = coflyRef.current
      cockpit.visible = cockpitOnRef.current || cof !== null
      coflyF.grp.visible = cof === 'f'
      coflyM.grp.visible = cof === 'm'
      if (cockpit.visible) {
        const dnCmd = (fr?.circuit?.dn) ?? { pitch: 0, yaw: 0 }
        const load = Math.min(1, (Math.abs(dnCmd.pitch) + Math.abs(dnCmd.yaw)) / 1.4)
        const sleeping = !fr
        const tNow = performance.now() / 1000
        flapPhase += (sleeping ? 2 : 14 + load * 26) * 0.016
        const flap = Math.sin(flapPhase * Math.PI * 2) * (sleeping ? 0.04 : 0.5 + load * 0.4)
        wingL.rotation.z = 0.3 + flap
        wingR.rotation.z = -0.3 - flap
        // пассажирка: противофаза; на перегрузке у первой машет чаще — «пищит громче»
        const coSpd = sleeping ? 1.7 : 2.3 + load * 1.5
        const coFlap = Math.sin(flapPhase * Math.PI * 2 * coSpd) * (sleeping ? 0.04 : 0.45 + load * 0.4)
        for (const cf of [coflyF, coflyM]) {
          if (cf.wingL) cf.wingL.rotation.z = 0.3 - coFlap
          if (cf.wingR) cf.wingR.rotation.z = -0.3 + coFlap
        }
        leverPitch.rotation.x = -dnCmd.pitch * 0.7
        leverYaw.rotation.z = -dnCmd.yaw * 0.7
        flyGrp.rotation.x = dnCmd.pitch * 0.25
        if (sleeping) {
          flyGrp.position.y = 0.004
          flyGrp.rotation.z = 0.18
        } else if (load > 0.8) {
          flyGrp.position.y = 0.009 + Math.sin(tNow * 60) * 0.002 // трясёт на большой перегрузке
          flyGrp.rotation.z = 0
        } else {
          flyGrp.position.y = 0.009 + Math.sin(tNow * 9) * 0.0012
          flyGrp.rotation.z = 0
        }
        zzz.spr.visible = sleeping
        // задние лапки: ступни на рычагах (дёргают с рывком по нагрузке), в сне — на корпусе
        const pull = sleeping ? 0 : Math.sin(tNow * 34) * 0.0015 * (0.3 + load)
        const knobP = new THREE.Vector3(0, 0.05, 0).applyEuler(leverPitch.rotation).add(leverPitch.position)
        const knobY = new THREE.Vector3(0, 0.05, 0).applyEuler(leverYaw.rotation).add(leverYaw.position)
        const restL = new THREE.Vector3(-0.021, 0.004, 0.014)
        const restR = new THREE.Vector3(0.021, 0.004, 0.014)
        solveLeg(hindL, sleeping ? restL : knobP.add(new THREE.Vector3(0, pull, 0)))
        solveLeg(hindR, sleeping ? restR : knobY.add(new THREE.Vector3(0, pull, 0)))
        solveLeg(midL)
        solveLeg(midR)
        solveLeg(foreL)
        solveLeg(foreR)
      }
      // озвучка: жужжание в тон манёвра — независимо от пасхалки; на резком вираже муха шутит.
      // Жужжит только одиночная муха в «Пуске»: рой и его подрежимы молчат (короткие циклы —
      // фразы не успевают), после финала (перехват/промах/пролёт) — тоже тишина
      const dnNow = fr?.circuit?.dn ?? { pitch: 0, yaw: 0 }
      const flyLoad = Math.min(1, (Math.abs(dnNow.pitch) + Math.abs(dnNow.yaw)) / 1.4)
      const finalEvent =
        fr?.event === 'перехват' || fr?.event === 'hit' || fr?.event === 'промах' || fr?.event === 'miss_pass' || fr?.event === 'отказ перехвата'
      const flyingNow = Boolean(!silentRef.current && !finalEvent && (pb ? pb.race && prog < 1 : fr && !idleRef.current))
      setBuzz(flyLoad, flyingNow)
      if (flyingNow && flyLoad > 0.78) say('overload')
      // в финале промаха — явная подпись наименьшего сближения на отрезке «ракета—цель».
      // Прогон может кончиться и без терминального события (в дуэли цель уходит по
      // истечении боевой жизни): тогда итог даёт проп `outcome` живого пуска или `hit`
      // лучшего боя веера, иначе над парой висела бы залипшая живая дистанция
      const endedNoKill = pb
        ? prog >= 1 && bestHit === false
        : outcomeRef.current !== null && !outcomeRef.current.hit && !idleRef.current
      const missedEnd =
        Boolean(fr && (fr.event === 'промах' || fr.event === 'miss_pass' || fr.event === 'отказ перехвата')) || endedNoKill
      missLabel.spr.visible = missedEnd
      if (missedEnd) {
        // цифра та же, что в вердикте: итог прогона из метрик, а кадрный минимум —
        // только пока прогон ещё идёт (исхода нет)
        const cpa = pb ? bestMiss ?? fr?.miss ?? 0 : outcomeRef.current?.missM ?? fr?.miss ?? 0
        missLabel.spr.position.copy(mp.clone().lerp(tp, 0.5)).add(new THREE.Vector3(0, 0.18, 0))
        missLabel.set(dayRef.current ? '#b03a22' : '#ff6a4a', `промах · наименьшее ${Math.round(cpa)} м`)
      }

      los.geometry.setFromPoints([mp, tp])
      ;(los.material as THREE.LineDashedMaterial).needsUpdate = true
      los.computeLineDistances()

      // ─── геометрия наведения: треугольник, прогноз h_cv, команда ───
      const showGeo = geoRef.current && !pb && Boolean(fr) && !idleRef.current
      geoGroup.visible = showGeo
      const chip = chipRef.current
      if (showGeo) {
        const r = tp.clone().sub(mp)
        const vRel = tv.clone().sub(mv)
        const vv2 = vRel.lengthSq()
        const tgo = vv2 > 1e-9 ? Math.max(0, -r.dot(vRel) / vv2) : 0
        const pip = tp.clone().addScaledVector(tv, tgo) // точка встречи (нулевое усилие цели)
        const mZe = mp.clone().addScaledVector(mv, tgo) // куда придёт ракета без манёвра
        const zem = pip.clone().sub(mZe)
        const course = pip.clone().sub(mp)
        const gamma = mv.lengthSq() > 1e-9 && course.lengthSq() > 1e-9
          ? (Math.acos(Math.max(-1, Math.min(1, mv.clone().normalize().dot(course.clone().normalize())))) * 180) / Math.PI
          : 0
        reqLine.geometry.setFromPoints([mp, pip])
        tgtToPip.geometry.setFromPoints([tp, pip])
        ;(tgtToPip.material as THREE.LineDashedMaterial).needsUpdate = true
        tgtToPip.computeLineDistances()
        // круг БЧ: реальный радиус sc.kill_radius_m, у текущей позиции цели, лицом к камере
        killRing.position.copy(tp)
        killRing.scale.setScalar(Math.max(1e-4, sc.kill_radius_m * KM))
        killRing.quaternion.copy(camera.quaternion)
        if (sc.kill_radius_m !== lastKillR) {
          lastKillR = sc.kill_radius_m
          killLabel.set(dayRef.current ? THEME_DAY.y : THEME_NIGHT.y, `БЧ R ${Math.round(sc.kill_radius_m)} м`)
        }
        killLabel.spr.position.copy(tp).add(new THREE.Vector3(0, sc.kill_radius_m * KM * 1.15, 0))
        if (aCmd && aCmd.lengthSq() > 1e-9) {
          aArrow.visible = true
          aArrow.position.copy(mp)
          aArrow.setDirection(smA.clone().normalize())
          const ms = missile.scale.x
          // длина честно по доле израсходованной перегрузки (a_cmd в км/с², n_max·g в м/с²)
          const gFrac = Math.min(1, smA.length() / (sc.n_max * 9.81 * KM))
          aArrow.setLength(ms * (0.35 + 0.45 * gFrac), ms * 0.16, ms * 0.08)
        } else {
          aArrow.visible = false
        }
        if (chip) {
          chip.textContent = `t_cpa ${tgo.toFixed(1)} с · прогноз h_cv ${(zem.length() * 1000).toFixed(0)} м · угол упреждения ${gamma.toFixed(0)}°`
          chip.style.display = 'block'
        }
      } else if (chip) {
        chip.style.display = 'none'
      }

      // стрелка манёвра цели: тянется от самолёта в сторону перегрузки, длина — по g.
      // Порог полграмма, чтобы прямолинейный полёт не рисовался «дёрганием»
      const targetFlying = Boolean(fr && !idleRef.current) && event !== 'перехват' && event !== 'hit' && target.visible
      tArrow.visible = targetFlying && smTa.length() > 0.5 * G_KM
      if (tArrow.visible) {
        tArrow.position.copy(tp)
        tArrow.setDirection(smTa.clone().normalize())
        const ts = target.scale.x
        tArrow.setLength(ts * (0.5 + 0.35 * Math.min(2, smTa.length() / G_KM)), ts * 0.16, ts * 0.08)
      }

      if (pb !== shownPlayback) {
        shownPlayback = pb
        disposeSwarm()
        restoreJet() // новое поколение — самолёт целый, даже если прошлый подрыв был
        if (pb) {
          // сетка — под весь веер траекторий поколения: стартовый прогон мог быть
          // по другому сценарию, и центр от старой пары веер бы не накрыл
          const bb = new THREE.Box3()
          pb.results.forEach((res) => {
            for (let i = 0; i < res.traj_m.length; i += 6) bb.expandByPoint(to3(res.traj_m[i]))
            for (let i = 0; i < res.traj_t.length; i += 6) bb.expandByPoint(to3(res.traj_t[i]))
          })
          if (!bb.isEmpty()) {
            const sz = bb.getSize(new THREE.Vector3())
            placeGrid(bb.getCenter(new THREE.Vector3()), Math.max(sz.x, sz.z))
            // численность роя — подписью над веером траекторий поколения
            const pop = makeLabel(pb.caption ?? `рой · ${pb.results.length} мух`, dayRef.current ? THEME_DAY.z : THEME_NIGHT.z, 0.5)
            pop.spr.position.copy(bb.getCenter(new THREE.Vector3())).add(new THREE.Vector3(0, 1.1, 0))
            scene.add(pop.spr)
            swarmLabels.push(pop.spr)
          }
          const T = dayRef.current ? THEME_DAY : THEME_NIGHT
          const fits = pb.results.map((res) => res.fitness)
          const sorted = fits.slice().sort((a, b) => a - b)
          const n = Math.max(pb.results.length - 1, 1)
          pb.results.forEach((res, i) => {
            const rank = sorted.indexOf(res.fitness)
            const k = 1 - rank / n
            const best = i === pb.bestIdx
            const col = best
              ? new THREE.Color(T.swarmBest)
              : new THREE.Color(T.swarmBad).lerp(new THREE.Color(T.swarmGood), k)
            // длинные траектории прореживаем вдвое: 48 линий полной истории
            // перестраиваются каждое поколение — геометрия вдвое легче, глаз не видит
            const src = res.traj_m
            const stride = src.length > 400 ? 2 : 1
            const pts = src.filter((_, i) => i % stride === 0 || i === src.length - 1).map(to3)
            if (pts.length < 2) pts.push(pts[0].clone())
            const line = new THREE.Line(
              new THREE.BufferGeometry().setFromPoints(pts),
              new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: best ? 1 : res.hit ? 0.8 : 0.5 }),
            )
            scene.add(line)
            swarmLines.push({ line, total: pts.length, hit: res.hit, best, k })
            // подписи особых траекторий («стажёр», «ветеран») — разносим по высоте
            if (res.label) {
              const spr = makeLabel(res.label, `#${col.getHexString()}`, 0.55)
              spr.spr.position
                .copy(pts[Math.min(pts.length - 1, Math.round(pts.length * 0.45))])
                .add(new THREE.Vector3(0, 0.3 + swarmLabels.length * 0.28, 0))
              scene.add(spr.spr)
              swarmLabels.push(spr.spr)
            }
          })
          // геометрия поколения: полная траектория цели лидера
          const bestRes = pb.results[Math.min(pb.bestIdx, pb.results.length - 1)]
          const tPts = bestRes.traj_t.map(to3)
          if (tPts.length >= 2) {
            tgtTrack = new THREE.Line(
              new THREE.BufferGeometry().setFromPoints(tPts),
              new THREE.LineDashedMaterial({ color: T.tgt, dashSize: 0.12, gapSize: 0.08, transparent: true, opacity: 0.4 }),
            )
            tgtTrack.computeLineDistances()
            scene.add(tgtTrack)
          }
        }
      }
      if (pb) {
        const prog = Math.min(1, Math.max(0, (performance.now() - pb.startedAt) / pb.durationMs))
        swarmLines.forEach((sl) => {
          sl.line.geometry.setDrawRange(0, Math.max(2, Math.ceil(prog * (sl.total - 1)) + 1))
          if (prog >= 0.98 && !sl.best && !sl.hit) (sl.line.material as THREE.LineBasicMaterial).opacity = 0.22
        })
      }

      // в след пишем точку на кадр телеметрии, а не на кадр рендера: на 120 Гц
      // иначе след «съедается» одинаковыми точками в два раза быстрее
      if (!pb && fr && intro > 0.35 && stamp !== lastTrailStamp) {
        lastTrailStamp = stamp
        // ракета: реальная доля выработанной перегрузки; цель: кинематическая оценка
        // манёвра (smTa уже используется для крена/стрелки цели выше) — честно
        // помечено как оценка в подсказке чипа «геометрия», не точный замер
        const gM = Math.max(0, Math.min(1, fr.n_req / Math.max(1, fr.n_lim)))
        const gT = Math.max(0, Math.min(1, smTa.length() / (G_KM * Math.max(sc.n_target, 1))))
        mTrail.push({ p: mp.clone(), g: gM })
        tTrail.push({ p: tp.clone(), g: gT })
        const cap = perfBudget().trailPoints
        if (mTrail.length > cap) mTrail.shift()
        if (tTrail.length > cap) tTrail.shift()
        const T = dayRef.current ? THEME_DAY : THEME_NIGHT
        if (mTrail.length > 1) setTrailGeometry(mLine, mTrail, trailCold.set(T.mLine), trailHot.set(T.mLineHot))
        if (tTrail.length > 1) setTrailGeometry(tLine, tTrail, trailCold.set(T.tLine), trailHot.set(T.tLineHot))
      }

      // перехват: ракета дошла до цели; самолёт без вспышки разлетается на части
      const hit = event === 'перехват' || event === 'hit'
      // живая дистанция на линии визирования: на экране пара может выглядеть «рядом»,
      // подпись показывает реальную дальность на конечном участке
      const showDist = !hit && !missedEnd && sepRaw < 3
      distLabel.spr.visible = showDist
      if (showDist) {
        distLabel.set(dayRef.current ? '#8a6a1a' : '#e7c15a', sepRaw < 1 ? `${Math.round(sepRaw * 1000)} м` : `${sepRaw.toFixed(2)} км`)
        distLabel.spr.position.copy(mp.clone().lerp(tp, 0.5)).add(new THREE.Vector3(0, 0.2 + effScale * 0.25, 0))
      }
      if (hit && !wasHit) {
        wasHit = true
        hitAt = performance.now()
        const mid = mp.clone().lerp(tp, 0.5)
        jetParts.forEach((p) => {
          const wp = p.mesh.getWorldPosition(new THREE.Vector3())
          const away = wp.sub(mid)
          if (away.lengthSq() < 1e-9) away.set(0, 1, 0)
          away.normalize()
          p.vel.set(
            away.x * (0.5 + partRng() * 0.8) + (partRng() - 0.5) * 0.5,
            0.5 + partRng() * 1.0,
            away.z * (0.5 + partRng() * 0.8) + (partRng() - 0.5) * 0.5,
          )
          p.spin.set((partRng() - 0.5) * 7, (partRng() - 0.5) * 7, (partRng() - 0.5) * 7)
        })
        // искры: короткий Points-всплеск от эпицентра, число — по тиру устройства
        activeSparks = perfBudget().sparkCount
        for (let i = 0; i < SPARK_MAX; i += 1) {
          if (i < activeSparks) {
            const dir = new THREE.Vector3(partRng() - 0.5, 0.25 + partRng() * 0.65, partRng() - 0.5).normalize()
            sparkVel[i].copy(dir).multiplyScalar(0.9 + partRng() * 1.4)
            sparkPos[i * 3] = mid.x
            sparkPos[i * 3 + 1] = mid.y
            sparkPos[i * 3 + 2] = mid.z
          } else {
            sparkPos[i * 3] = mid.x
            sparkPos[i * 3 + 1] = mid.y
            sparkPos[i * 3 + 2] = mid.z
          }
        }
        sparkGeo.attributes.position.needsUpdate = true
        sparks.visible = activeSparks > 0
      }
      if (!hit) {
        wasHit = false
        sparks.visible = false
      }
      missile.visible = !hit
      if (hit) {
        // разлёт: кувырки и гравитация, части плавно гаснут; короткая (120-160мс) вспышка
        // на материалах обломков — подхватывается существующим UnrealBloomPass, никакого
        // нового полноэкранного слоя/света. HUD — DOM над canvas, вспышка его не перекрывает
        const age = (performance.now() - hitAt) / 1000
        const dts = 0.016
        jetParts.forEach((p) => {
          p.vel.y -= 0.55 * dts
          p.mesh.position.addScaledVector(p.vel, dts)
          p.mesh.rotation.x += p.spin.x * dts
          p.mesh.rotation.y += p.spin.y * dts
          p.mesh.rotation.z += p.spin.z * dts
        })
        const flash = Math.max(0, 1 - age / 0.14)
        jetMats.forEach((m, i) => {
          m.opacity = Math.max(0, Math.min(1, 1 - Math.max(0, age - 1.0) / 1.3))
          m.emissiveIntensity = jetMatsBaseEmissive[i] + flash * 2.4
        })
        if (activeSparks > 0) {
          for (let i = 0; i < activeSparks; i += 1) {
            sparkVel[i].y -= 0.6 * dts
            sparkPos[i * 3] += sparkVel[i].x * dts
            sparkPos[i * 3 + 1] += sparkVel[i].y * dts
            sparkPos[i * 3 + 2] += sparkVel[i].z * dts
          }
          sparkGeo.attributes.position.needsUpdate = true
          sparkMat.opacity = Math.max(0, 1 - age / 1.1)
        }
      }
      target.visible = true

      // старт прогона (фронт «простой → полёт»): один раз перелетаем к пусковому ракурсу,
      // дальше на маршевом участке камера стоит в мире неподвижно — см. ветку «авто».
      // Если пользователь как раз крутит камеру в этот момент, откладываем перелёт
      // на следующий кадр (wasInRun не трогаем) — раньше при активном вращении переход
      // терялся НАВСЕГДА, и новый прогон стартовал с камерой, застрявшей в терминальном
      // кадре предыдущего (цель не попадала в кадр с самого начала)
      const inRun = Boolean(pb) || Boolean(fr && !idleRef.current)
      if (inRun && !wasInRun) {
        if (!interacting) {
          const home = homeView(scRef.current)
          flightPos.copy(home.pos)
          flightTgt.copy(home.tgt)
          flight = camRef.current !== 'chase'
          userZoom = 1
          terminalCam = false
          userFreeUntilNextRun = false
          wasInRun = true
        }
      } else {
        wasInRun = inRun
      }

      const mid = mp.clone().lerp(tp, 0.42)
      const desired = pb ? swarmCenter : mid
      const spanScene = Math.max(sepRaw, pb ? swarmSpan : 0, 0.5)
      const wantDist = Math.min(Math.max(spanScene * 1.6, 0.8), 120)
      if (rp && camRef.current !== 'free' && prog < 1 && !interacting) {
        // демо-гонка: обе ракеты в кадре, камера сбоку и дальше —
        // пока идёт показ; ручное вращение имеет приоритет
        controls.enabled = true
        const mid = mp.clone().lerp(rp, 0.5)
        const dirRace = mv.lengthSq() > 1e-9 ? mv.clone().normalize() : new THREE.Vector3(1, 0, 0)
        const rightR = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), dirRace).normalize()
        const wantPos = mid.clone().addScaledVector(dirRace, -1.7).addScaledVector(rightR, 1.35).add(new THREE.Vector3(0, 0.75, 0))
        camera.position.lerp(wantPos, 0.08)
        controls.target.lerp(mid.clone().addScaledVector(dirRace, 0.35), 0.12)
      } else if (camRef.current === 'chase') {
        controls.enabled = false
        const dir = mv.lengthSq() > 1e-9 ? mv.clone().normalize() : new THREE.Vector3(1, 0, 0)
        const want = mp.clone().addScaledVector(dir, -2.4).add(new THREE.Vector3(0, 0.55, 0))
        camera.position.lerp(want, 0.06)
        controls.target.lerp(mp.clone().addScaledVector(dir, 1.2), 0.08)
      } else if (camRef.current === 'free') {
        // свободная: ракурс и дистанция полностью у пользователя
        controls.enabled = true
        if (nowMs < recenterUntil) controls.target.lerp(desired, 0.12) // двойной клик возвращает взгляд на действие
      } else {
        // авто: маршевый участок прогона камера стоит в мире (ниже), терминал — наезд,
        // веер роя и простой — слежение за центром действия
        controls.enabled = true
        const dirAuto = mv.lengthSq() > 1e-9 ? mv.clone().normalize() : new THREE.Vector3(1, 0, 0)
        if (cockpitOnRef.current || cof !== null) {
          // пасхалка/юмор: муха на ракете крупно, но кадр шире — видно и цель, и сцену вперёд
          const right = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), dirAuto).normalize()
          const wantPos = mp.clone().addScaledVector(dirAuto, -1.3).addScaledVector(right, 1.15).add(new THREE.Vector3(0, 0.55, 0))
          const wantTgt = mp.clone().addScaledVector(dirAuto, 0.9)
          if (freshRun) {
            camera.position.copy(wantPos)
            controls.target.copy(wantTgt)
          } else {
            camera.position.lerp(wantPos, 0.08)
            controls.target.lerp(wantTgt, 0.12)
          }
        } else if (pb?.cinematic) {
          // инстант-реплей перехвата: неполный (~1.15π) облёт точки удара — те же данные,
          // просто медленнее и с новым ракурсом; первое касание/вращение отменяет облёт
          // так же, как обычную «авто»-камеру (interacting уже общий для всех веток)
          const manual = interacting || nowMs - lastInteract < 1500
          if (!manual) {
            const angle = prog * Math.PI * 1.15
            const dist = Math.max(1.2, sepRaw * 2.6, spanBase * 0.05)
            camera.position.set(mid.x + Math.cos(angle) * dist, mid.y + dist * 0.3, mid.z + Math.sin(angle) * dist)
          }
          controls.target.lerp(mid, 0.15)
        } else if (flight) {
          // перелёт к пусковому ракурсу сценария или нового прогона
          controls.target.lerp(flightTgt, 0.05)
          camera.position.lerp(flightPos, 0.05)
          const tol = Math.max(0.03, spanBase * 0.008)
          if (camera.position.distanceTo(flightPos) < tol && controls.target.distanceTo(flightTgt) < tol) flight = false
        } else if (fr && !idleRef.current && !pb) {
          // одиночный прогон, маршевый участок: камера ЗАФИКСИРОВАНА в мире.
          // Если она едет за парой и наезжает пропорционально сближению, пара всегда
          // занимает одну долю кадра — самолёт кажется стоящим относительно осей.
          // Единственное исключение — терминальный участок: мягкий наезд на точку встречи.
          // Ручное вращение отдаёт камеру пользователю ДО КОНЦА прогона (не на 1.5с) —
          // правило «ручное вращение → свободная камера», а не временная уступка
          const manual = interacting || userFreeUntilNextRun
          // манёвренная цель — камера приходит раньше: уклонение должно быть видно
          const evasive = sc.maneuver !== 'straight' && sc.n_target > 0
          // второе условие — по времени до встречи (R/Vсбл), а не только по дистанции:
          // терминальный наезд гарантированно укладывается в последние ~1.6с, а не во всю
          // дистанцию. tRadial офлайн-фолбэк (localSim.ts) не считает — обязателен fallback на tgo
          const tGoNow = fr.tRadial ?? fr.tgo
          if (sepRaw < Math.max(1.2, spanBase * (evasive ? 0.45 : 0.3)) || (tGoNow !== null && tGoNow < 1.6)) terminalCam = true
          if (terminalCam && !manual) {
            controls.target.lerp(mid, 0.08)
            const dist = camera.position.distanceTo(controls.target)
            const wantD = Math.min(Math.max(sepRaw * 3, 1.0), 7.5)
            if (dist > 1e-4)
              camera.position.sub(controls.target).multiplyScalar(1 + ((wantD - dist) * 0.06) / dist).add(controls.target)
          } else if (nowMs < recenterUntil) {
            controls.target.lerp(desired, 0.12)
          }
        } else {
          // веер роя и простой: точка внимания следует за центром действия,
          // дистанция мягко держится (ручной масштаб учитывается через userZoom)
          controls.target.lerp(desired, nowMs < recenterUntil ? 0.12 : 0.04)
          const manual = interacting || nowMs - lastInteract < 1500
          const dist = camera.position.distanceTo(controls.target)
          if (dist > 1e-4) {
            if (manual) {
              userZoom = Math.min(4, Math.max(0.2, dist / wantDist))
            } else {
              const want = Math.min(Math.max(wantDist * userZoom, controls.minDistance * 2), controls.maxDistance)
              camera.position.sub(controls.target).multiplyScalar(1 + ((want - dist) * 0.03) / dist).add(controls.target)
            }
          }
        }
      }
      controls.update()
      composer.render()
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)

    // скрытая вкладка: rAF не крутится вхолостую — все тайминги завязаны на
    // performance.now() напрямую (hitAt/age, pb.startedAt, lastInteract), так
    // что пауза не требует пересчёта состояния при возврате
    const onVisibility = () => {
      if (document.hidden) cancelAnimationFrame(raf)
      else raf = requestAnimationFrame(tick)
    }
    document.addEventListener('visibilitychange', onVisibility)

    return () => {
      document.removeEventListener('visibilitychange', onVisibility)
      cancelAnimationFrame(raf)
      missLabel.spr.removeFromParent()
      distLabel.spr.removeFromParent()
      killLabel.spr.removeFromParent()
      killRing.geometry.dispose()
      ;(killRing.material as THREE.Material).dispose()
      sparkGeo.dispose()
      sparkMat.dispose()
      ro.disconnect()
      disposeSwarm()
      controls.dispose()
      composer.dispose()
      renderer.dispose()
      if (renderer.domElement.parentNode === el) el.removeChild(renderer.domElement)
    }
  }, [])

  const curve = swarmCurve && swarmCurve.length >= 2 ? swarmCurve : null
  const validIdx = curve && swarmValid ? swarmValid.filter((i) => i >= 0 && i < curve.length) : []
  const cw = 214
  const ch = 86
  const pad = 8
  const lo = curve ? Math.min(...curve) : 0
  const hi = curve ? Math.max(...curve) : 1
  const span = Math.max(hi - lo, 1)
  const px = (i: number) => pad + (i * (cw - 2 * pad)) / Math.max(curve!.length - 1, 1)
  const py = (v: number) => ch - pad - ((v - lo) / span) * (ch - 2 * pad)
  const curvePts = curve ? curve.map((v, i) => `${px(i)},${py(v)}`).join(' ') : ''

  return (
    <div className="viewport viewport--3d" ref={host}>
      <div className="geo-chip" ref={chipRef} />
      {sceneBadge && <div className="scene-badge">{sceneBadge}</div>}
      {curve && (
        <svg className="swarm-curve" viewBox={`0 0 ${cw} ${ch}`} data-tip="Эволюция роя: наименьшее сближение (м) по поколениям; пунктирные тики — валидация на эталонном трио.">
          <text x={pad} y={12} fill="#7dffc8" fontSize="9">
            эволюция: наименьшее сближение {Math.round(lo)} → {Math.round(curve[curve.length - 1])} м
          </text>
          {validIdx.map((i) => (
            <line
              key={`v${i}`}
              x1={px(i)}
              x2={px(i)}
              y1={16}
              y2={ch - pad + 2}
              stroke="#7dffc8"
              strokeWidth="0.8"
              strokeDasharray="2 2"
              opacity="0.45"
            />
          ))}
          <polyline points={curvePts} fill="none" stroke="#e7c15a" strokeWidth="1.6" />
          {curve.map((v, i) => (
            <circle
              key={i}
              cx={px(i)}
              cy={py(v)}
              r={i === curve.length - 1 ? 2.4 : 1.6}
              fill={i === curve.length - 1 ? '#d2ffe9' : '#e7c15a'}
              opacity={i === curve.length - 1 ? 1 : 0.55}
            />
          ))}
          {validIdx.length > 0 && (
            <text x={cw - pad} y={ch - 1} textAnchor="end" fill="#7dffc8" fontSize="7.5" opacity="0.8">
              тик — валидация
            </text>
          )}
        </svg>
      )}
      <p className="view-hint">
        <span className="for-mouse">левая кнопка — вращать · колесо — масштаб · </span>
        <span className="for-touch">палец — вращать · щипок — масштаб · </span>X красная — дальность · Y синяя — бок · Z зелёная — высота, км
      </p>
    </div>
  )
}
