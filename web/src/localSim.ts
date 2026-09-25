import { features, isTrained, nCells, readout, sgd } from './brain'
import type { BrainRegion, Frame, Scenario } from './types'

const G = 9.81

/** Честная упаковка активности нейронов в корзины (зеркало navedenie/circuit.py::_pack_activity). */
export function packActivity(vecs: [string, number[]][]): { n_neurons: number; regions: BrainRegion[]; act_b64: string } {
  const parts = vecs.map(([, v]) => v)
  const total = parts.reduce((s, p) => s + p.length, 0)
  const bins = 2048
  const bpb = Math.max(total / bins, 1e-9)
  const binsVal = new Uint8Array(bins)
  let off = 0
  const regions: BrainRegion[] = []
  for (const [name, v] of vecs) {
    let mean = 0
    for (let i = 0; i < v.length; i += 1) {
      const gi = Math.floor((off + i) / bpb)
      const q = Math.min(255, Math.round(Math.abs(v[i]) * 255))
      if (q > binsVal[gi]) binsVal[gi] = q
      mean += Math.abs(v[i])
    }
    regions.push({ name, start: off, n: v.length, mean: v.length ? mean / v.length : 0 })
    off += v.length
  }
  let bin = ''
  for (let i = 0; i < binsVal.length; i += 1) bin += String.fromCharCode(binsVal[i])
  return { n_neurons: total, regions, act_b64: btoa(bin) }
}

function add(a: number[], b: number[], s = 1): number[] {
  return [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s]
}
function sub(a: number[], b: number[]): number[] {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
function dot(a: number[], b: number[]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
function cross(a: number[], b: number[]): number[] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
function norm(a: number[]) {
  return Math.hypot(a[0], a[1], a[2])
}
function unit(a: number[]): number[] {
  const n = norm(a) || 1e-12
  return [a[0] / n, a[1] / n, a[2] / n]
}
function perp(a: number[], v: number[]): number[] {
  const s2 = dot(v, v) + 1e-12
  const k = dot(a, v) / s2
  return sub(a, [v[0] * k, v[1] * k, v[2] * k])
}
function clip(a: number[], nMax: number): number[] {
  const mag = norm(a)
  const lim = nMax * G
  if (mag > lim && mag > 0) return [a[0] * (lim / mag), a[1] * (lim / mag), a[2] * (lim / mag)]
  return a
}

type Body = { p: number[]; v: number[] }

/** Шаг кинематической точки-массы: |v| сохраняется, ускорение поворачивает скорость,
 *  позиция — трапеция. Зеркало navedenie/sim.py::integrate. */
export function integrate(b: Body, a: number[], dt: number) {
  const speed = norm(b.v)
  const aMag = norm(a)
  if (speed < 1e-9) return
  if (aMag < 1e-12) {
    b.p = add(b.p, b.v, dt)
    return
  }
  const axis = cross(b.v, a).map((x) => x / (aMag * speed))
  const theta = Math.min((aMag * dt) / speed, Math.PI)
  const c = Math.cos(theta)
  const s = Math.sin(theta)
  const axv = cross(axis, b.v)
  const axd = dot(axis, b.v) * (1 - c)
  const v1: number[] = [
    b.v[0] * c + axv[0] * s + axis[0] * axd,
    b.v[1] * c + axv[1] * s + axis[1] * axd,
    b.v[2] * c + axv[2] * s + axis[2] * axd,
  ]
  b.p = add(b.p, add(b.v, v1), dt / 2)
  b.v = v1
}

/** Межшаговая встреча: CPA, пересечение сферы БЧ, доля шага входа. Зеркало sim.step_encounter. */
export function stepEncounter(pM0: number[], pM1: number[], pT0: number[], pT1: number[], kill: number) {
  const r0 = sub(pT0, pM0)
  const dr = sub(sub(pT1, pM1), r0)
  const a = dot(dr, dr)
  const b = 2 * dot(r0, dr)
  const c0 = dot(r0, r0)
  const r0In = c0 <= kill * kill
  let alphaStar = 0
  if (a > 1e-18) alphaStar = Math.min(1, Math.max(0, -b / (2 * a)))
  const rStar = add(r0, dr, alphaStar)
  const cpa = norm(rStar)
  let hit = false
  let alphaIn: number | null = null
  if (r0In) {
    hit = true
    alphaIn = 0
  } else if (a > 1e-18) {
    const disc = b * b - 4 * a * (c0 - kill * kill)
    if (disc >= 0) {
      const sq = Math.sqrt(disc)
      const a1 = (-b - sq) / (2 * a)
      if (a1 >= 0 && a1 <= 1) {
        hit = true
        alphaIn = a1
      }
    }
  }
  return { cpa: a > 1e-18 || cpa > 0 ? cpa : norm(r0), hit, alphaIn }
}

function velFromAngles(speed: number, hdgDeg: number, climbDeg: number): number[] {
  // зеркало sim.py::vel_from_angles: V·(cosγ·cosψ, cosγ·sinψ, sinγ)
  const hdg = (hdgDeg * Math.PI) / 180
  const climb = (climbDeg * Math.PI) / 180
  const ch = speed * Math.cos(climb)
  return [ch * Math.cos(hdg), ch * Math.sin(hdg), speed * Math.sin(climb)]
}

function spawn(sc: Scenario) {
  // range_m — ТОЧНАЯ начальная наклонная дальность для всех аспектов;
  // v_t — фактический модуль скорости цели ВО ВСЕХ аспектах (скрытых множителей нет);
  // исключение — 'free' («свободная расстановка»): цель и курсы задаются числами
  const pM: number[] = [0, 0, sc.alt_m]
  let vM: number[] = [sc.v_m, 0, 0]
  if (sc.aspect === 'free') {
    const pT = [sc.free_tx, sc.free_ty, sc.free_talt]
    vM = velFromAngles(sc.v_m, sc.free_mhdg, sc.free_mclimb)
    const vT = velFromAngles(sc.v_t, sc.free_thdg, sc.free_tclimb)
    return { pM, vM, pT, vT }
  }
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
  const n = norm(raw)
  const pT = add(pM, raw, sc.range_m / n)
  return { pM, vM, pT, vT }
}

function emptyCircuit(): Frame['circuit'] {
  return {
    kind: 'stub',
    photo: Array.from({ length: 16 }, () => Array(16).fill(0)),
    t4: [0, 0, 0, 4].slice(0, 4) as number[],
    dn: { pitch: 0, yaw: 0 },
    lplc2: 0,
    layers: { Зрение: 0, Поток: 0, Приближение: 0, Решение: 0, Мотор: 0 },
    weights: isTrained() ? 'обучен' : 'не обучен',
    n_cells: nCells(),
    n_neurons: 0,
    regions: [],
    act_b64: '',
  }
}

// ── фовеальная сетчатка: зеркало navedenie/seeker.py ─────────────────────────

export const FOVEA_K = 3.5
export const SEEKER_N = 16

export function foveaAngles(halfFov: number, n = SEEKER_N, k = FOVEA_K): number[] {
  const out: number[] = []
  for (let i = 0; i < n; i += 1) {
    const q = -1 + (2 * (i + 0.5)) / n
    out.push((halfFov * Math.sinh(k * q)) / Math.sinh(k))
  }
  return out
}

function foveaCellWidth(halfFov: number, n = SEEKER_N, k = FOVEA_K): number[] {
  const out: number[] = []
  for (let i = 0; i < n; i += 1) {
    const q = -1 + (2 * (i + 0.5)) / n
    out.push(((halfFov * k * Math.cosh(k * q)) / Math.sinh(k)) * (2 / n))
  }
  return out
}

/** Изображение цели на фовеальной сетчатке (зеркало raster_foveal).
 *  Локальная планка PSF: в фовеа пятно узкое — угловой размер цели измерим. */
export function rasterFoveal(az: number, el: number, size: number, halfFov: number, n = SEEKER_N): number[][] {
  const ax = foveaAngles(halfFov, n)
  const wx = foveaCellWidth(halfFov, n)
  const sig = wx.map((w) => Math.max(0.7 * w, size * halfFov * 0.9))
  const maxA = ax[ax.length - 1]
  const azC = Math.max(-maxA, Math.min(maxA, az))
  const elC = Math.max(-maxA, Math.min(maxA, el))
  const gx = ax.map((a, i) => Math.exp(-0.5 * ((a - azC) / sig[i]) ** 2))
  const gy = ax.map((a, i) => Math.exp(-0.5 * ((a - elC) / sig[i]) ** 2))
  const img: number[][] = []
  for (let j = 0; j < n; j += 1) img.push(gx.map((g, i) => gy[j] * g))
  return img
}

/** Суммарная яркость пятна с ядром sigma (та же модель, что raster) — для θ-решателя. */
function blobTotal(az: number, el: number, sigma: number, halfFov: number, ax: number[], wx: number[]): number {
  const sig = wx.map((w) => Math.max(0.7 * w, sigma))
  const maxA = ax[ax.length - 1]
  const azC = Math.max(-maxA, Math.min(maxA, az))
  const elC = Math.max(-maxA, Math.min(maxA, el))
  let sum = 0
  const gx = ax.map((a, i) => Math.exp(-0.5 * ((a - azC) / sig[i]) ** 2))
  const gy = ax.map((a, i) => Math.exp(-0.5 * ((a - elC) / sig[i]) ** 2))
  for (let j = 0; j < ax.length; j += 1) for (let i = 0; i < ax.length; i += 1) sum += gy[j] * gx[i]
  return sum
}

/** Измеренный угловой размер цели (рад): обратная задача оптики бисекцией по σ.
 *  Зеркало navedenie/seeker.py::measure_theta. Цель много меньше PSF → θ = 0. */
export function measureTheta(total: number, az: number, el: number, halfFov: number, n = SEEKER_N): number {
  if (total <= 0) return 0
  const ax = foveaAngles(halfFov, n)
  const wx = foveaCellWidth(halfFov, n)
  const tLo = blobTotal(az, el, 0, halfFov, ax, wx)
  if (total <= tLo * 1.01) return 0
  let lo = 0
  let hi = halfFov
  for (let i = 0; i < 14; i += 1) {
    const mid = 0.5 * (lo + hi)
    if (blobTotal(az, el, mid, halfFov, ax, wx) < total) lo = mid
    else hi = mid
  }
  return 0.5 * (lo + hi)
}

/** Декодирование признаков из изображения — единственный источник углов агента.
 *  theta — измеренный угловой размер (калибровка PSF собственной сетчатки). */
export function decodeImage(img: number[][], halfFov: number, n = SEEKER_N) {
  const ax = foveaAngles(halfFov, n)
  let total = 0
  const wx = new Array(n).fill(0)
  const wy = new Array(n).fill(0)
  let peak = 0
  for (let j = 0; j < n; j += 1)
    for (let i = 0; i < n; i += 1) {
      const v = img[j][i]
      total += v
      wx[i] += v
      wy[j] += v
      if (v > peak) peak = v
    }
  if (total < 0.15) return { az: 0, el: 0, size: 0, peak: 0, theta: 0 }
  let az = 0
  let el = 0
  for (let i = 0; i < n; i += 1) {
    az += wx[i] * ax[i]
    el += wy[i] * ax[i]
  }
  az /= Math.max(wx.reduce((s, v) => s + v, 0), 1e-9)
  el /= Math.max(wy.reduce((s, v) => s + v, 0), 1e-9)
  const size = Math.max(0, Math.min(0.35, total / (n * n * 0.08)))
  const theta = measureTheta(total, az, el, halfFov, n)
  return { az, el, size, peak, theta }
}

/** 16×16 «омматидии» для ДЕМО-кадров в окне (равномерная сетка — только визуализация). */
export function raster(az: number, el: number, size: number, fov: number) {
  const n = 16
  const half = fov / 2
  const img: number[][] = []
  const sig = Math.max(size * half * 4, half * 0.08)
  for (let j = 0; j < n; j += 1) {
    const row: number[] = []
    const y = -half + (2 * half * j) / (n - 1)
    for (let i = 0; i < n; i += 1) {
      const x = -half + (2 * half * i) / (n - 1)
      row.push(Math.exp(-0.5 * (((x - az) / sig) ** 2 + ((y - el) / sig) ** 2)))
    }
    img.push(row)
  }
  return img
}

function targetAccel(vT: number[], sc: Scenario, t: number): number[] {
  if (sc.maneuver === 'straight' || sc.n_target <= 0) return [0, 0, 0]
  let lift = unit(cross(vT, [0, 0, 1]))
  if (norm(lift) < 0.2) lift = unit(cross(vT, [0, 1, 0]))
  const n = sc.n_target * G
  const vert = [0, 0, n]
  const m = sc.maneuver
  // зеркало navedenie/sim.py::target_accel
  let a: number[]
  if (m === 'turn') {
    a = [lift[0] * n, lift[1] * n, lift[2] * n]
  } else if (m === 'weave') {
    const s = Math.sin((2 * Math.PI * t) / 4)
    a = [lift[0] * n * s, lift[1] * n * s, lift[2] * n * s]
  } else if (m === 'weave_var') {
    const phase = 2 * Math.PI * (t / 4 + 0.055 * t * t)
    const ramp = Math.min(1, 0.35 + t / 6)
    const s = n * ramp * Math.sin(phase)
    a = [lift[0] * s, lift[1] * s, lift[2] * s]
  } else if (m === 'break') {
    a = t > 2.5 ? [lift[0] * n, lift[1] * n, lift[2] * n] : [0, 0, 0]
  } else if (m === 'scissors') {
    const sign = Math.floor(t / 1.6) % 2 === 0 ? 1 : -1
    const frac = t % 1.6
    const front = Math.min(1, frac / 0.35, (1.6 - frac) / 0.35)
    a = [lift[0] * n * sign * front, lift[1] * n * sign * front, lift[2] * n * sign * front]
  } else if (m === 'dive') {
    const s = Math.sin((2 * Math.PI * t) / 6)
    a = [vert[0] * s, vert[1] * s, vert[2] * s]
  } else if (m === 'combo') {
    const sH = n * (0.6 + 0.4 * Math.sin((2 * Math.PI * t) / 5))
    const sV = n * 0.45 * Math.sin((2 * Math.PI * t) / 3 + 1)
    a = [
      lift[0] * sH + vert[0] * 0.45 * Math.sin((2 * Math.PI * t) / 3 + 1),
      lift[1] * sH + vert[1] * 0.45 * Math.sin((2 * Math.PI * t) / 3 + 1),
      lift[2] * sH + vert[2] * 0.45 * Math.sin((2 * Math.PI * t) / 3 + 1),
    ]
  } else {
    a = [0, 0, 0]
  }
  // полный вектор ускорения цели ограничен n_target·g (зеркало sim.target_accel)
  const mag = norm(a)
  if (mag > n && n > 0) a = a.map((x) => (x * n) / mag)
  return perp(a, vT)
}

/** Продольное ускорение цели, м/с² (меняет МОДУЛЬ скорости; n_target — только вираж).
 *  constant → 0; accelerate → +A; decelerate → −A; pulse — импульс 0.15·T на фазе;
 *  sine — A·sin(2π(t−фаза)/T). Зеркало sim.target_long_accel. */
export function targetLongAccel(sc: Scenario, t: number): number {
  if (sc.target_speed_mode === 'constant') return 0
  const a = Math.abs(sc.target_longitudinal_g) * G
  const period = Math.max(sc.target_speed_period_s, 0.2)
  if (sc.target_speed_mode === 'accelerate') return a
  if (sc.target_speed_mode === 'decelerate') return -a
  if (sc.target_speed_mode === 'pulse') {
    const tau = t - sc.target_speed_phase
    return tau >= 0 && tau < 0.15 * period ? a : 0
  }
  if (sc.target_speed_mode === 'sine') return a * Math.sin((2 * Math.PI * (t - sc.target_speed_phase)) / period)
  return 0
}

/** Шаг цели: вираж (|v| сохраняется) + продольное изменение модуля в границах.
 *  Позиция — трапеция. constant ведёт себя в точности как integrate(). Зеркало integrate_target. */
export function integrateTarget(b: Body, sc: Scenario, t: number, dt: number) {
  const aLong = targetLongAccel(sc, t)
  const v0 = [...b.v]
  const speed = norm(v0)
  const pBefore = [...b.p]
  integrate(b, targetAccel(b.v, sc, t), dt)
  if (speed < 1e-9 || Math.abs(aLong) < 1e-12) return
  const speed1 = Math.max(sc.target_speed_min, Math.min(sc.target_speed_max, speed + aLong * dt))
  const newSpeed = norm(b.v)
  if (newSpeed > 1e-9 && Math.abs(speed1 - newSpeed) > 1e-12) b.v = b.v.map((x) => (x * speed1) / newSpeed)
  b.p = [
    pBefore[0] + ((v0[0] + b.v[0]) * dt) / 2,
    pBefore[1] + ((v0[1] + b.v[1]) * dt) / 2,
    pBefore[2] + ((v0[2] + b.v[2]) * dt) / 2,
  ]
}

/** Диагностика N_экв: какому коэффициенту ПН эквивалентна проекция команды.
 *  Зеркало pn.n_eff_from: q = perp(ω×V, V); N_экв = <a,q>/<q,q>. */
export function neffFrom(
  aCmd: number[],
  r: number[],
  vM: number[],
  vT: number[],
  nMax: number,
): { nEff: number | null; reason: string | null } {
  const vRel = sub(vT, vM)
  const r2 = dot(r, r) + 1e-12
  const omega = cross(r, vRel).map((x) => x / r2)
  const wv = cross(omega, vM)
  const k = dot(wv, vM) / (dot(vM, vM) + 1e-12)
  const q = sub(wv, vM.map((x) => x * k))
  const qn = norm(q)
  const aMag = norm(aCmd)
  if (!Number.isFinite(qn) || !Number.isFinite(aMag)) return { nEff: null, reason: 'nonfinite' }
  if (qn < 0.5) return { nEff: null, reason: 'no_geom' }
  if (aMag >= 0.98 * nMax * G) return { nEff: null, reason: 'saturation' }
  return { nEff: dot(aCmd, q) / dot(q, q), reason: null }
}

/** «ПН через ГСН»: команда только из измеренных угловых скоростей.
 *  Зеркало navedenie/pn.py::pn_seeker_accel. Оси: x по скорости, y вправо, z вверх
 *  (скоростная система). a = |v|·N·(ω_az·ŷ + ω_el·ẑ) — знак по месту ПОЛОЖИТЕЛЬНЫЙ
 *  (цель уходит вверх → команда вверх), как в Python. */
export function sensorPnAccel(azDot: number, elDot: number, vM: number[], nConst: number, nMax: number): number[] {
  const x = unit(vM)
  let right = cross([0, 0, 1], x)
  if (norm(right) < 0.05) right = cross([0, 1, 0], x)
  right = unit(right)
  const up = unit(cross(x, right))
  const speed = norm(vM) + 1e-9
  const a = [
    right[0] * azDot + up[0] * elDot,
    right[1] * azDot + up[1] * elDot,
    right[2] * azDot + up[2] * elDot,
  ].map((c) => c * speed * nConst)
  return clip(perp(a, vM), nMax)
}

export function* localRun(sc: Scenario, opts?: { learn?: boolean; lr?: number }): Generator<Frame> {
  // Авторитетная реализация трёхстепенной физической модели — Python (navedenie/physics.py).
  // Локальный браузерный симулятор её НЕ воспроизводит; чтобы не выдавать расходящиеся
  // числа, физический режим обслуживает только сервер (§14).
  if (sc.model === 'point_mass_3dof') {
    throw new Error('режим «Трёхстепенная модель движения центра масс» недоступен в локальном окне — используйте серверный расчёт')
  }
  const dt = 0.02
  let { pM, vM, pT, vT } = spawn(sc)
  const mB: Body = { p: pM, v: vM }
  const tB: Body = { p: pT, v: vT }
  // призрак: эталонный ПН летит ту же цель — метрика «похожести»
  const gB: Body = { p: [...pM], v: [...vM] }
  let t = 0
  let cpa = 1e9
  // инерция рулевого привода (1-е звено) — зеркало navedenie/engine.py; призрак не затронут
  const tauAct = sc.tau_act_s || 0
  let aExec: number[] = [0, 0, 0]
  let prevAz = 0
  let prevEl = 0
  let prevSize = 0
  let prevTheta = 0
  let thetaWin: number[] = []
  let azDotS = 0
  let elDotS = 0
  let first = true
  const gauss = () => {
    // Box–Muller для шума измерителей в окне
    const u = Math.max(Math.random(), 1e-9)
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random())
  }
  // эталонный ПН смотрит в узкий прибор; БИО — фовеальная полусфера
  const halfFovBio = ((Math.max(sc.bio_fov_deg, 2) * Math.PI) / 180) / 2
  const halfFovPn = ((sc.fov_deg * Math.PI) / 180) / 2
  const halfFov = sc.mode === 'pn' ? halfFovPn : halfFovBio

  while (t < sc.t_max) {
    const r = sub(tB.p, mB.p)
    // текущие скорости тел (внешние vM/vT из spawn — только начальная установка;
    // команды и диагностика считаются по ним, зеркало engine.run)
    const vM = mB.v
    const vT = tB.v
    const rng = norm(r)
    const vRel = sub(tB.v, mB.v)
    const vc = -dot(unit(r), vRel)
    const omega = cross(r, vRel).map((x) => x / (rng * rng + 1e-12)) as number[]

    const fwd = unit(mB.v)
    const worldUp = [0, 0, 1]
    let right = cross(worldUp, fwd)
    if (norm(right) < 0.05) right = cross([0, 1, 0], fwd)
    right = unit(right)
    const up = unit(cross(fwd, right))
    const u = unit(r)
    const ahead = dot(u, fwd)
    const truthAz = Math.atan2(dot(u, right), ahead)
    const truthEl = Math.atan2(dot(u, up), Math.hypot(ahead, dot(u, right)))

    // изображение → (шум) → декодирование: единственный источник углов агента
    const size = Math.min(0.35, 12 / rng)
    let img: number[][] = rasterFoveal(truthAz, truthEl, size, halfFov)
    const azNoise = (sc.noise_az_deg * Math.PI) / 180
    if (azNoise > 0) {
      const s = azNoise / Math.max(halfFov, 1e-9) / 2
      img = img.map((row) => row.map((v) => Math.max(0, v + gauss() * s)))
    }
    const dec = decodeImage(img, halfFov)
    const losAng = Math.acos(Math.min(1, Math.max(-1, ahead)))
    let lock = dec.peak > 0.25 && Math.abs(truthAz) <= halfFov && Math.abs(truthEl) <= halfFov
    if (lock && sc.lock_drop_p > 0 && Math.random() < sc.lock_drop_p) lock = false
    const az = lock ? dec.az : 0
    const el = lock ? dec.el : 0
    const azDot = first || !lock ? 0 : (az - prevAz) / dt
    const elDot = first || !lock ? 0 : (el - prevEl) / dt
    // фаза сближения: сглаживание декодированного размера + окно θ̇ + rho (зеркало seeker.observe)
    const thetaTau = 0.06
    const rawTheta = lock ? dec.theta : 0
    const theta = rawTheta + (prevTheta - rawTheta) * Math.exp(-dt / thetaTau)
    thetaWin = [...thetaWin.slice(-5), theta]
    const winDt = Math.max((thetaWin.length - 1) * dt, dt)
    const thetaDot = Math.max(-5, Math.min(5, (thetaWin[thetaWin.length - 1] - thetaWin[0]) / winDt))
    const rho = Math.max(-10, Math.min(30, thetaDot / Math.max(theta, 1e-4)))
    const tauContact = thetaDot > 1e-3 ? Math.max(0, Math.min(30, theta / thetaDot)) : 30
    // сглаженные угловые скорости — отдельные каналы для углоскоростных законов
    const kRate = 1 - Math.exp(-dt / 0.1)
    azDotS = lock ? azDotS + (azDot - azDotS) * kRate : azDotS * (1 - kRate)
    elDotS = lock ? elDotS + (elDot - elDotS) * kRate : elDotS * (1 - kRate)
    prevAz = az
    prevEl = el
    prevTheta = theta

    const aPn = clip(perp(cross(omega, vM).map((x) => x * sc.pn_n) as number[], vM), sc.n_max)
    // команда призрака считается по его собственной геометрии до интеграции шага
    const rG = sub(tB.p, gB.p)
    const vRelG = sub(tB.v, gB.v)
    const rngG = norm(rG)
    const omG = cross(rG, vRelG).map((x) => x / (rngG * rngG + 1e-12)) as number[]
    const aGhost = clip(perp(cross(omG, gB.v).map((x) => x * sc.pn_n) as number[], gB.v), sc.n_max)
    const gx = lock ? Math.tanh(azDot) : 0
    const gy = lock ? Math.tanh(elDot) : 0
    const loom = Math.max(0, first ? 0 : (dec.size - prevSize) / dt) * 40
    const feat = features(az, el, azDot, elDot, theta, loom, rho, gx, gy, lock)
    prevSize = dec.size
    first = false
    if (opts?.learn && lock) {
      const scale = sc.n_max * G + 1e-9
      sgd(feat, [
        Math.max(-1, Math.min(1, dot(aPn, up) / scale)),
        Math.max(-1, Math.min(1, dot(aPn, right) / scale)),
      ], opts.lr ?? 0.04)
    }
    const { pitch, yaw } = readout(feat)
    const aBio = clip(
      perp(
        add(
          up.map((x) => x * pitch * sc.n_max * G * sc.circuit_gain) as number[],
          right.map((x) => x * yaw * sc.n_max * G * sc.circuit_gain) as number[],
        ),
        vM,
      ),
      sc.n_max,
    )
    let aCmd: number[]
    if (sc.mode === 'pn') {
      if (sc.law === 'tpn') {
        // истинная ПН: a = N·V_c·(ω×r̂) по нормали к ЛВ — зеркало navedenie/pn.py
        const rHat = unit(r)
        const vcPos = Math.max(0, vc)
        const tpnVec = cross(omega, rHat).map((x) => x * sc.pn_n * vcPos)
        aCmd = clip(perp(tpnVec, vM), sc.n_max)
      } else if (sc.law === 'apn') {
        // a = N·(ω×v) + (N/2)·a_t,⊥ — размерностно корректно, зеркало navedenie/pn.py
        const targetLift = targetAccel(vT, sc, t)
        const pnVec = cross(omega, vM).map((x) => x * sc.pn_n)
        const comp = perp(targetLift, vM).map((x) => (x * sc.pn_n) / 2)
        aCmd = clip(perp(add(pnVec, comp), vM), sc.n_max)
      } else if (sc.law === 'pn_gsn') {
        // сенсорная ПН: декодированные угловые скорости, сглаженные фильтром ГСН
        aCmd = lock ? sensorPnAccel(azDotS, elDotS, vM, sc.pn_n, sc.n_max) : [0, 0, 0]
      } else if (sc.law === 'pn_sched_oracle') {
        // ПН с переменным N (эталон): истинное Vc/R — подсказка геометрией
        const rhoTrue = vc / Math.max(rng, 1)
        const nSched = Math.max(sc.pn_sched_n_min, Math.min(sc.pn_sched_n_max, sc.pn_sched_n0 + sc.pn_sched_k_rho * rhoTrue))
        aCmd = clip(perp(cross(omega, vM).map((x) => x * nSched) as number[], vM), sc.n_max)
      } else if (sc.law === 'pn_sched_sensor') {
        // ПН с переменным N (сенсорная): N из ИЗМЕРЕННОГО ρ, команда из сглаженных ω
        const nSched = Math.max(sc.pn_sched_n_min, Math.min(sc.pn_sched_n_max, sc.pn_sched_n0 + sc.pn_sched_k_rho * rho))
        aCmd = lock ? sensorPnAccel(azDotS, elDotS, vM, nSched, sc.n_max) : [0, 0, 0]
      } else if (sc.law === 'pure') {
        const desired = unit(r)
        const cur = unit(vM)
        aCmd = clip(
          perp([(desired[0] - cur[0]) * sc.n_max * G, (desired[1] - cur[1]) * sc.n_max * G, (desired[2] - cur[2]) * sc.n_max * G], vM),
          sc.n_max,
        )
      } else if (sc.law === 'clos') {
        const launchP = [0, 0, sc.alt_m]
        const dist = norm(sub(tB.p, launchP)) + 1e-9
        const frac = Math.max(0, Math.min(1, 1 - norm(sub(tB.p, mB.p)) / dist))
        const lineP = add(launchP, sub(tB.p, launchP), frac)
        const err = sub(lineP, mB.p)
        const k = (1.2 * norm(vM) / Math.max(rng, 300)) * 2
        aCmd = clip(perp(err.map((x) => x * k), vM), sc.n_max)
      } else {
        aCmd = aPn
      }
    } else if (sc.mode === 'both') aCmd = lock ? aBio : aPn
    else aCmd = lock ? aBio : [0, 0, 0]
    const nReq = norm(aCmd) / G
    // диагностика N_экв: истинная геометрия — постфактум, НЕ для управления
    const neff = neffFrom(aCmd, r, vM, vT, sc.n_max)
    const sat = nReq >= 0.98 * sc.n_max

    // шаг интеграции (|v| ракеты сохраняется; цель — с профилем скорости) и встреча
    const pM0 = [...mB.p]
    const pT0 = [...tB.p]
    const vM0 = [...mB.v]
    const vT0 = [...tB.v]
    const g0 = [...gB.p]
    integrateTarget(tB, sc, t, dt)
    let aFly = aCmd
    if (tauAct > 1e-12) {
      const kAct = dt / (tauAct + dt)
      aExec = perp(add(aExec, sub(aCmd, aExec).map((x) => x * kAct)), mB.v)
      aFly = aExec
    }
    integrate(mB, aFly, dt)
    integrate(gB, aGhost, dt)
    const enc = stepEncounter(pM0, mB.p, pT0, tB.p, sc.kill_radius_m)
    cpa = Math.min(cpa, enc.cpa)

    const layers = {
      Зрение: lock ? 0.4 + dec.size : 0.05,
      Поток: Math.min(1, Math.abs(azDot) + Math.abs(elDot)),
      Приближение: Math.min(1, dec.size * 3),
      Решение: Math.min(1, Math.abs(az) + Math.abs(el)),
      Мотор: Math.min(1, nReq / sc.n_max),
    }
    const activity = packActivity([
      ['глаз', img.flat()],
      ['DN', [pitch, yaw]],
    ])

    let event: string | null = t < dt ? 'пуск' : null
    if (enc.hit) {
      event = 'перехват'
    } else if (t > 0.6 && vc < 0 && rng > cpa + 50) event = 'промах'

    yield {
      t,
      missile: pM0,
      target: pT0,
      missile_v: vM0,
      target_v: vT0,
      a_cmd: aCmd,
      a_pn: aPn,
      n_req: nReq,
      n_lim: sc.n_max,
      range_m: rng,
      v_c: vc,
      omega_los: norm(omega),
      az: truthAz,
      el: truthEl,
      lock,
      miss: cpa,
      seeker: { image: img, size: dec.size, az_dot: azDot, el_dot: elDot },
      circuit: {
        ...emptyCircuit(),
        photo: img,
        dn: { pitch, yaw },
        lplc2: dec.size,
        layers,
        ...activity,
      },
      layers,
      event,
      ghost: g0,
      theta,
      theta_dot: thetaDot,
      rho,
      tau_contact: tauContact,
      n_eff: neff.nEff,
      n_eff_valid: neff.reason === null,
      n_eff_reason: neff.reason,
      n_eff_yaw: null,
      n_eff_pitch: null,
      sat,
      // tgo — устаревший alias = t_radial (R/Vc); t_cpa — время до ближайшего сближения
      tgo: vc > 1e-6 ? rng / vc : null,
      t_radial: vc > 1e-6 ? rng / vc : null,
      t_cpa: -dot(r, vRel) / Math.max(dot(vRel, vRel), 1e-12),
      speed_mode: sc.target_speed_mode,
      target_speed: norm(tB.v),
    }

    if (event === 'перехват' || event === 'промах') return
    t += dt
  }
}
