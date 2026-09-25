/** Рой мух на клиенте: популяция, эволюция, прогон поколения (сервер → фолбэк в окне). */

import { features } from './brain'
import { decodeImage, integrate, packActivity, rasterFoveal, stepEncounter } from './localSim'
import type { FlyGenome, FlyResult, Frame, Scenario, SwarmGenResponse } from './types'

const G = 9.81
const FEAT = 10 // схема признаков v2 (+theta, +rho)
const W_LIMIT = 4
const GAIN_RANGE: [number, number] = [0.4, 2.5]
const PN_RANGE: [number, number] = [1.5, 8]

export function defaultW(): number[][] {
  // врождённый рефлекс v2 = v1 (новые столбцы theta/rho — нули)
  const w = Array.from({ length: 2 }, () => Array(FEAT).fill(0))
  w[0][1] = 1.8
  w[0][3] = 0.35
  w[0][5] = 0.25
  w[0][8] = 0.4
  w[1][0] = 1.8
  w[1][2] = 0.35
  w[1][5] = 0.25
  w[1][7] = 0.4
  return w
}

/** Детерминированный ГПСЧ, чтобы эволюция в окне совпадала при одинаковых seed. */
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

function gauss(rng: () => number) {
  const u = Math.max(rng(), 1e-9)
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng())
}

const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x))

export function sanitizeFly(blob: Partial<FlyGenome>): FlyGenome {
  const kind = blob.kind === 'pn' ? 'pn' : 'bio'
  if (kind === 'pn') return { kind, w: [], gain: 1, pn_n: clamp(Number(blob.pn_n) || 4, ...PN_RANGE) }
  let w: number[]
  if (Array.isArray(blob.w) && blob.w.length === 16) {
    // старый геном схемы v1: расширяем нулями на theta/rho (поведение сохранено)
    const src = blob.w.map((x) => clamp(Number(x) || 0, -W_LIMIT, W_LIMIT))
    const map = [0, 1, 2, 3, 5, 7, 8, 9]
    w = Array(FEAT * 2).fill(0)
    map.forEach((newIdx, oldIdx) => {
      w[newIdx] = src[oldIdx]
      w[FEAT + newIdx] = src[8 + oldIdx]
    })
  } else if (Array.isArray(blob.w) && blob.w.length === FEAT * 2) w = blob.w.map((x) => clamp(Number(x) || 0, -W_LIMIT, W_LIMIT))
  else w = defaultW().flat()
  return { kind, w, gain: clamp(Number(blob.gain) || 1, ...GAIN_RANGE), pn_n: 4 }
}

export function initPopulation(n: number, seed: number): FlyGenome[] {
  const rng = mulberry32(seed)
  const flies: FlyGenome[] = []
  for (let i = 0; i < Math.max(n, 2); i += 1) {
    if (i % 2 === 0) {
      const base = defaultW().flat()
      const w = base.map((x) => clamp(x + gauss(rng) * 0.35, -W_LIMIT, W_LIMIT))
      flies.push({ kind: 'bio', w, gain: clamp(GAIN_RANGE[0] + rng() * (GAIN_RANGE[1] - GAIN_RANGE[0]), ...GAIN_RANGE), pn_n: 4 })
    } else {
      flies.push({ kind: 'pn', w: [], gain: 1, pn_n: clamp(PN_RANGE[0] + rng() * (PN_RANGE[1] - PN_RANGE[0]), ...PN_RANGE) })
    }
  }
  return flies
}

const BIO_W = 20 // 2×10, схема признаков v2
const LEGACY_BIO_W = 16 // 2×8, схема v1

/** Миграция генома: старые 16 весов расширяются нулями на столбцах theta/rho. */
function migrateBioW(w: number[]): number[] {
  if (w.length >= BIO_W) return w
  if (w.length !== LEGACY_BIO_W) return w.concat(Array(Math.max(0, BIO_W - w.length)).fill(0))
  const map = [0, 1, 2, 3, 5, 7, 8, 9]
  const out = Array(BIO_W).fill(0)
  map.forEach((newIdx, oldIdx) => {
    out[newIdx] = w[oldIdx]
    out[10 + newIdx] = w[8 + oldIdx]
  })
  return out
}

function bioDn(fly: FlyGenome, feat: number[]): { pitch: number; yaw: number } {
  const w = migrateBioW(fly.w)
  const tanh = Math.tanh
  return {
    pitch: tanh(w.slice(0, 10).reduce((s, v, i) => s + v * (feat[i] ?? 0), 0)),
    yaw: tanh(w.slice(10, 20).reduce((s, v, i) => s + v * (feat[i] ?? 0), 0)),
  }
}

/** Точка телеметрии полёта мухи — зеркалит TelPoint из types.ts. */
type Tel = {
  t: number
  az: number
  el: number
  lock: boolean
  size: number
  sizeDot: number
  azDot: number
  elDot: number
  pitch: number
  yaw: number
  nReq: number
  miss: number
  rng: number
  v: number[]
}

/** Любой перехват лучше любого промаха; среди промахов — градиент по кратчайшему расстоянию (зеркало swarm.py). */
export const HIT_BASE = 1000
const CPA_CAP = 5000

/** Лёгкий прогон одной мухи в окне — ДЕМО-зеркало navedenie/swarm.py::rollout (без задержек ГСН). */
export function flyRollout(sc: Scenario, fly: FlyGenome): FlyResult {
  const dt = 0.02
  const tMax = Math.min(sc.t_max, 20)
  const { pM, vM, pT, vT } = spawnBody(sc)
  const mB: Body = { p: pM, v: vM }
  const tB: Body = { p: pT, v: vT }

  const sub = (a: number[], b: number[]) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
  const add = (a: number[], b: number[], s = 1) => [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s]
  const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
  const cross = (a: number[], b: number[]): number[] => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
  const norm = (a: number[]) => Math.hypot(a[0], a[1], a[2])
  const unit = (a: number[]) => {
    const n = norm(a) || 1e-12
    return [a[0] / n, a[1] / n, a[2] / n]
  }
  const perp = (a: number[], v: number[]) => {
    const k = dot(a, v) / (dot(v, v) + 1e-12)
    return sub(a, [v[0] * k, v[1] * k, v[2] * k])
  }
  const clip = (a: number[], nMax: number) => {
    const mag = norm(a)
    const lim = nMax * G
    return mag > lim ? [a[0] * (lim / mag), a[1] * (lim / mag), a[2] * (lim / mag)] : a
  }

  let t = 0
  let cpa = 1e9
  let effort = 0
  let hit = false
  let tHit: number | null = null
  let prevAz = 0
  let prevThetaSw = 0 // сглаженный theta предыдущего шага — для rho в рое
  let prevEl = 0
  let prevSize = 0
  let first = true
  const trajM: number[][] = []
  const trajT: number[][] = []
  const tel: Tel[] = []
  let nextSample = 0
  const sample = (pitch: number, yaw: number, nReq: number) => {
    trajM.push([...mB.p])
    trajT.push([...tB.p])
    tel.push({
      t,
      az: prevAz,
      el: prevEl,
      lock: telLock,
      size: prevSize,
      sizeDot: telSizeDot,
      azDot: telAzDot,
      elDot: telElDot,
      pitch,
      yaw,
      nReq,
      miss: cpa,
      rng: norm(sub(tB.p, mB.p)),
      v: [...mB.v],
    })
  }
  let telLock = false
  let telSizeDot = 0
  let telAzDot = 0
  let telElDot = 0
  sample(0, 0, 0)

  const halfFovBio = ((Math.max(sc.bio_fov_deg, 2) * Math.PI) / 180) / 2
  while (t < tMax) {
    const r = sub(tB.p, mB.p)
    const rng = norm(r)

    let aCmd: number[]
    let pitch = 0
    let yaw = 0
    let lock = true
    if (fly.kind === 'pn') {
      // pn-муха — эталонный закон с постоянной N
      const vRel = sub(tB.v, mB.v)
      const omega = cross(r, vRel).map((x) => x / (rng * rng + 1e-12))
      aCmd = clip(perp(cross(omega, mB.v).map((x) => x * fly.pn_n), mB.v), sc.n_max)
      const scale = sc.n_max * G + 1e-9
      const fwd = unit(mB.v)
      let right = cross([0, 0, 1], fwd)
      if (norm(right) < 0.05) right = cross([0, 1, 0], fwd)
      right = unit(right)
      const up = unit(cross(fwd, right))
      pitch = Math.max(-1, Math.min(1, dot(aCmd, up) / scale))
      yaw = Math.max(-1, Math.min(1, dot(aCmd, right) / scale))
    } else {
      // био: изображение → декодирование → команды (без задержек в демо-зеркале)
      const fwd = unit(mB.v)
      let right = cross([0, 0, 1], fwd)
      if (norm(right) < 0.05) right = cross([0, 1, 0], fwd)
      right = unit(right)
      const up = unit(cross(fwd, right))
      const u = unit(r)
      const ahead = dot(u, fwd)
      const truthAz = Math.atan2(dot(u, right), ahead)
      const truthEl = Math.atan2(dot(u, up), Math.hypot(ahead, dot(u, right)))
      const size = Math.min(0.35, 12 / rng)
      const img = rasterFoveal(truthAz, truthEl, size, halfFovBio)
      const dec = decodeImage(img, halfFovBio)
      lock = dec.peak > 0.25 && Math.abs(truthAz) <= halfFovBio && Math.abs(truthEl) <= halfFovBio
      const az = lock ? dec.az : 0
      const el = lock ? dec.el : 0
      const azDot = first ? 0 : (az - prevAz) / dt
      const elDot = first ? 0 : (el - prevEl) / dt
      const sizeDot = first ? 0 : (dec.size - prevSize) / dt
      first = false
      prevAz = az
      prevEl = el
      prevSize = dec.size
      telLock = lock
      telSizeDot = sizeDot
      telAzDot = azDot
      telElDot = elDot
      if (lock) {
        const thetaSw = dec.theta ?? 0
        const rhoSw = first ? 0 : thetaSw > 1e-4 ? ((thetaSw - prevThetaSw) / dt) / Math.max(thetaSw, 1e-4) : 0
        prevThetaSw = thetaSw
        const feat = features(az, el, azDot, elDot, thetaSw, sizeDot * 40, Math.max(-10, Math.min(30, rhoSw)), Math.tanh(azDot), Math.tanh(elDot), true)
        const { pitch: p2, yaw: y2 } = bioDn(fly, feat)
        pitch = p2
        yaw = y2
        aCmd = clip(
          perp(add(up.map((x) => x * pitch * sc.n_max * G * fly.gain), right.map((x) => x * yaw * sc.n_max * G * fly.gain)), mB.v),
          sc.n_max,
        )
      } else {
        aCmd = [0, 0, 0]
      }
    }

    const nReq = norm(aCmd) / G
    effort += nReq * dt

    const pM0 = [...mB.p]
    const pT0 = [...tB.p]
    integrate(tB, targetLiftOf(tB.v, sc, t), dt)
    integrate(mB, aCmd, dt)
    const enc = stepEncounter(pM0, mB.p, pT0, tB.p, sc.kill_radius_m)
    cpa = Math.min(cpa, enc.cpa)

    if (enc.hit) {
      hit = true
      tHit = t + (enc.alphaIn ?? 0) * dt
      sample(pitch, yaw, nReq)
      break
    }
    const vRel = sub(tB.v, mB.v)
    const vc = -dot(unit(r), vRel)
    if (t > 0.6 && vc < 0 && rng > cpa + 50) break

    t += dt
    if (t >= nextSample) {
      sample(pitch, yaw, nReq)
      nextSample += 0.25
    }
  }

  const fitness = hit ? tHit! + 0.02 * effort : HIT_BASE + Math.min(cpa, CPA_CAP)
  return {
    fly: sanitizeFly(fly),
    fitness,
    miss_m: cpa,
    hit,
    n_int: effort,
    traj_m: trajM,
    traj_t: trajT,
    tel,
  }
}

type Body = { p: number[]; v: number[] }

function spawnBody(sc: Scenario) {
  const pM: number[] = [0, 0, sc.alt_m]
  const vM: number[] = [sc.v_m, 0, 0]
  let raw: number[]
  let vT: number[]
  if (sc.aspect === 'beam') {
    raw = [0.55, -0.65, 0]
    vT = [0, sc.v_t, 0]
  } else if (sc.aspect === 'tail-chase') {
    raw = [0.45, (0.4 * sc.off_axis_m) / Math.max(sc.range_m, 1), 40 / Math.max(sc.range_m, 1)]
    vT = [sc.v_t, 0, 0]
  } else {
    raw = [1, sc.off_axis_m / Math.max(sc.range_m, 1), 80 / Math.max(sc.range_m, 1)]
    vT = [-sc.v_t, 0, 0]
  }
  const n = Math.hypot(raw[0], raw[1], raw[2])
  const pT = [pM[0] + (raw[0] * sc.range_m) / n, pM[1] + (raw[1] * sc.range_m) / n, pM[2] + (raw[2] * sc.range_m) / n]
  return { pM, vM, pT, vT }
}

function targetLiftOf(vT: number[], sc: Scenario, t: number): number[] {
  if (sc.maneuver === 'straight' || sc.n_target <= 0) return [0, 0, 0]
  const cross = (a: number[], b: number[]): number[] => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
  const norm = (a: number[]) => Math.hypot(a[0], a[1], a[2])
  const unit = (a: number[]) => {
    const n = norm(a) || 1e-12
    return [a[0] / n, a[1] / n, a[2] / n]
  }
  const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
  const perp = (a: number[], v: number[]) => {
    const k = dot(a, v) / (dot(v, v) + 1e-12)
    return [a[0] - v[0] * k, a[1] - v[1] * k, a[2] - v[2] * k]
  }
  let lift = unit(cross(vT, [0, 0, 1]))
  if (norm(lift) < 0.2) lift = unit(cross(vT, [0, 1, 0]))
  const n = sc.n_target * G
  const vert = [0, 0, n]
  let a = lift.map((x) => x * n)
  if (sc.maneuver === 'weave') a = a.map((x) => x * Math.sin((2 * Math.PI * t) / 4))
  const mag = norm(a)
  if (mag > n && n > 0) a = a.map((x) => (x * n) / mag)
  return perp(a, vT)
}

/** Кадр из телеметрии мухи: глаз, мозг, рули и параметры оживают синхронно с проигрышем поколения. */
export function synthFrameFromTel(sc: Scenario, tel: Tel[], k: number): Frame {
  const n = tel.length
  const x = Math.max(0, Math.min(1, k)) * (n - 1)
  const i = Math.min(n - 2, Math.max(0, Math.floor(x)))
  const f = n > 1 ? x - i : 0
  const mix = (a: number, b: number) => a + (b - a) * f
  const p = tel[i]
  const q = tel[Math.min(n - 1, i + 1)]
  const az = mix(p.az, q.az)
  const el = mix(p.el, q.el)
  const lock = p.lock || q.lock
  const size = mix(p.size, q.size)
  const sizeDot = mix(p.sizeDot, q.sizeDot)
  const azDot = mix(p.azDot, q.azDot)
  const elDot = mix(p.elDot, q.elDot)
  const pitch = mix(p.pitch, q.pitch)
  const yaw = mix(p.yaw, q.yaw)
  const nReq = mix(p.nReq, q.nReq)
  const miss = mix(p.miss, q.miss)
  const rng = mix(p.rng, q.rng)
  const t = mix(p.t, q.t)
  const v0 = p.v ?? q.v ?? [sc.v_m, 0, 0]
  const v1 = q.v ?? p.v ?? v0
  const v = [mix(v0[0], v1[0]), mix(v0[1], v1[1]), mix(v0[2], v1[2])]

  const seeHalf = ((165 * Math.PI) / 180) / 2
  const img = lock ? rasterFoveal(az, el, size, seeHalf) : Array.from({ length: 16 }, () => Array(16).fill(0))
  const activity = packActivity([
    ['глаз', img.flat()],
    ['DN', [pitch, yaw]],
  ])
  const scale = (sc.n_max || 1) * G
  const nu = (a: number[]) => {
    const nn = Math.hypot(a[0], a[1], a[2]) || 1e-9
    return [a[0] / nn, a[1] / nn, a[2] / nn]
  }
  const c3 = (a: number[], b: number[]): number[] => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
  const fu = nu(v.length ? v : [1, 0, 0])
  let rt = c3([0, 0, 1], fu)
  if (Math.hypot(rt[0], rt[1], rt[2]) < 0.05) rt = c3([0, 1, 0], fu)
  rt = nu(rt)
  const up = nu(c3(fu, rt))
  const aCmd = [
    up[0] * pitch * scale + rt[0] * yaw * scale,
    up[1] * pitch * scale + rt[1] * yaw * scale,
    up[2] * pitch * scale + rt[2] * yaw * scale,
  ]
  const layers = {
    Зрение: lock ? 0.4 + size : 0.05,
    Поток: Math.min(1, Math.abs(azDot) + Math.abs(elDot)),
    Приближение: Math.min(1, size * 3),
    Решение: Math.min(1, Math.abs(az) + Math.abs(el)),
    Мотор: Math.min(1, nReq / Math.max(sc.n_max, 1)),
  }
  return {
    t,
    missile: [0, 0, 0],
    target: [0, 0, 0],
    missile_v: v,
    target_v: [0, 0, 0],
    a_cmd: aCmd,
    a_pn: aCmd,
    n_req: nReq,
    n_lim: sc.n_max,
    range_m: rng,
    v_c: 0,
    omega_los: 0,
    az,
    el,
    lock,
    miss,
    seeker: { image: img, size, az_dot: azDot, el_dot: elDot },
    circuit: {
      kind: 'stub',
      photo: img,
      t4: [0, 0, 0, 0],
      dn: { pitch, yaw },
      lplc2: Math.min(1, Math.max(0, sizeDot * 40)),
      layers,
      weights: 'не обучен',
      n_cells: 279,
      ...activity,
    },
    layers,
    event: null,
    theta: 0,
    theta_dot: 0,
    rho: 0,
    tau_contact: 30,
    n_eff: null,
    n_eff_valid: false,
    n_eff_reason: null,
    n_eff_yaw: null,
    n_eff_pitch: null,
    sat: false,
    tgo: null,
    speed_mode: (sc as Partial<Scenario>).target_speed_mode ?? 'constant',
    target_speed: 0,
  }
}

export function evolve(
  flies: FlyGenome[],
  fits: number[],
  eliteK: number,
  mutation: number,
  seed: number,
): FlyGenome[] {
  const n = flies.length
  if (n < 2) return flies.slice()
  const rng = mulberry32(seed)
  const order = fits.map((f, i) => [f, i] as const).sort((a, b) => a[0] - b[0])
  const eliteKc = clamp(eliteK, 1, n - 1)
  const next: FlyGenome[] = order.slice(0, eliteKc).map(([, i]) => sanitizeFly(flies[i]))

  const tournament = () => {
    const i = Math.floor(rng() * n)
    const j = Math.floor(rng() * n)
    return fits[i] <= fits[j] ? flies[i] : flies[j]
  }
  const crossover = (a: FlyGenome, b: FlyGenome): FlyGenome => {
    const kind = a.kind === b.kind || rng() < 0.5 ? a.kind : b.kind
    if (kind === 'pn') return { kind: 'pn', w: [], gain: 1, pn_n: rng() < 0.5 ? a.pn_n : b.pn_n }
    const w = a.w.map((x, i) => (rng() < 0.5 ? x : b.w[i]))
    return { kind: 'bio', w, gain: rng() < 0.5 ? a.gain : b.gain, pn_n: 4 }
  }
  const mutate = (fly: FlyGenome): FlyGenome => {
    const out = sanitizeFly(fly)
    if (out.kind === 'bio') {
      out.w = out.w.map((x) => clamp(x + gauss(rng) * mutation, -W_LIMIT, W_LIMIT))
      out.gain = clamp(out.gain + gauss(rng) * mutation * 0.3, ...GAIN_RANGE)
    } else {
      out.pn_n = clamp(out.pn_n + gauss(rng) * mutation * 1.5, ...PN_RANGE)
    }
    return out
  }

  while (next.length < n) {
    if (rng() < 0.08) {
      next.push(initPopulation(2, Math.floor(rng() * 1e9))[Math.floor(rng() * 2)])
      continue
    }
    next.push(mutate(crossover(tournament(), tournament())))
  }
  return next.slice(0, n)
}

export type GenerationResult = SwarmGenResponse & { local: boolean }

const ASPECTS: Scenario['aspect'][] = ['head-on', 'beam', 'tail-chase']
const EASY: Scenario['maneuver'][] = ['straight', 'turn', 'weave']
const HARD: Scenario['maneuver'][] = ['weave_var', 'break', 'scissors', 'dive', 'combo']

/** Геометрия поколения — детерминированно новая каждый раз, чтобы рой не переобучался. */
export function sampleGenerationScenario(base: Scenario, gen: number, seed: number): Scenario {
  const rng = mulberry32((seed * 7919 + gen * 104729) >>> 0)
  const pick = <T,>(arr: T[]): T => arr[Math.floor(rng() * arr.length)]
  // первые поколения — простые манёвры, дальше рой встречает весь арсенал
  const pool = gen < 4 ? EASY : EASY.concat(HARD)
  return {
    ...base,
    aspect: ASPECTS[gen % 3],
    range_m: 3500 + rng() * 7500,
    v_m: 700 + rng() * 200,
    v_t: 160 + rng() * 180,
    off_axis_m: 50 + rng() * 1150,
    maneuver: pick(pool),
    n_target: pick([0, 0, 2, 3, 4]),
    alt_m: 3500 + rng() * 1000,
    // редкие эпизоды с шумами: рой учится наводить в неидеальных условиях
    noise_az_deg: pick([0, 0, 0, 1.5, 3]),
    lock_drop_p: pick([0, 0, 0.1]),
    fov_deg: 10 + rng() * 6,
  }
}

/** Эталонное трио для валидации чемпиона. */
export function canonicalScenarios(base: Scenario): Scenario[] {
  return (['head-on', 'beam', 'tail-chase'] as const).map((aspect) => ({
    ...base,
    aspect,
    range_m: 6000,
    off_axis_m: 250,
    v_t: 240,
    maneuver: 'straight' as const,
    n_target: 0,
    alt_m: 4000,
  }))
}

export function scenarioLabel(sc: Scenario): string {
  const aspect = sc.aspect === 'head-on' ? 'встречные' : sc.aspect === 'beam' ? 'пересечение' : 'вдогон'
  const man =
    sc.maneuver === 'straight'
      ? 'прямо'
      : sc.maneuver === 'turn'
        ? 'вираж'
        : sc.maneuver === 'weave'
          ? 'змейка'
          : sc.maneuver === 'weave_var'
            ? 'перемен. змейка'
            : sc.maneuver === 'break'
              ? 'форс. вираж'
              : sc.maneuver === 'scissors'
                ? 'ножницы'
                : sc.maneuver === 'dive'
                  ? 'горка'
                  : 'комбо'
  return `${aspect}, ${(sc.range_m / 1000).toFixed(1)} км, ${man}${sc.n_target > 0 ? ` ${sc.n_target}g` : ''}`
}

const median = (xs: number[]) => {
  const s = xs.slice().sort((a, b) => a - b)
  return s.length % 2 ? s[(s.length - 1) / 2] : 0.5 * (s[s.length / 2 - 1] + s[s.length / 2])
}

/** Одно поколение: список сценариев (обычно 1; при валидации + эталонное трио).
 * Фитнес — медиана по сценариям, траектории для повтора — из первого. */
export async function runGeneration(
  scenarios: Scenario[],
  population: FlyGenome[],
  cfg: { eliteK: number; mutation: number; seed: number },
): Promise<GenerationResult> {
  const local = (pop: FlyGenome[]): GenerationResult => {
    const perSc = scenarios.map((sc) => pop.map((fly) => flyRollout(sc, fly)))
    const fits = pop.map((_, i) => median(perSc.map((res) => res[i].fitness)))
    const misses = pop.map((_, i) => median(perSc.map((res) => res[i].miss_m)))
    const bestIdx = fits.reduce((b, f, i) => (f < fits[b] ? i : b), 0)
    const bioWs = pop.filter((f) => f.kind === 'bio').map((f) => f.w)
    const diversity = bioWs.length > 1
      ? bioWs[0].map((_, j) => {
          const col = bioWs.map((w) => w[j])
          const m = col.reduce((s, v) => s + v, 0) / col.length
          return Math.sqrt(col.reduce((s, v) => s + (v - m) ** 2, 0) / col.length)
        }).reduce((s, v) => s + v, 0) / bioWs[0].length
      : 0
    const stats: SwarmGenResponse['stats'] = {
      best: fits[bestIdx],
      avg: fits.reduce((s, f) => s + f, 0) / fits.length,
      worst: Math.max(...fits),
      hit_rate: perSc[0].filter((res) => res.hit).length / Math.max(pop.length, 1),
      diversity,
      best_idx: bestIdx,
    }
    if (scenarios.length > 1) {
      const canonMiss = pop.map((_, i) => median(perSc.slice(1).map((res) => res[i].miss_m)))
      stats.canon_best = canonMiss[canonMiss.reduce((b, f, i) => (f < canonMiss[b] ? i : b), 0)]
    }
    return {
      results: pop.map((fly, i) => ({ ...perSc[0][i], fitness: fits[i], miss_m: misses[i], fly: sanitizeFly(fly) })),
      next_population: evolve(pop, fits, cfg.eliteK, cfg.mutation, cfg.seed),
      stats,
      local: true,
    }
  }
  try {
    const res = await fetch('/api/swarm/gen', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        scenario: scenarios[0],
        scenarios,
        population: population.map(sanitizeFly),
        elite_k: cfg.eliteK,
        mutation: cfg.mutation,
        seed: cfg.seed,
      }),
    })
    if (!res.ok) throw new Error(`сервер ${res.status}`)
    const data = (await res.json()) as SwarmGenResponse
    return { ...data, local: false }
  } catch {
    return local(population)
  }
}
