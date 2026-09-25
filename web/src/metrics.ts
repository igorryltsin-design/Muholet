/** Единые метрики прогона (версия 4): одна точка расчёта для серверных и локальных кадров.
 *
 *  hit — «перехват»: сфера срабатывания пересечена; cpaM — R_min, непрерывный минимум расстояния;
 *  tGuide — интерполированное время входа в сферу; nInt — J_n, интеграл модуля заданной
 *  нормальной перегрузки ∫|a_cmd|/g dt; refNrms — НСКО рассогласования с призраком-эталоном,
 *  нормированное НАЧАЛЬНОЙ дальностью; N_экв — диагностика адаптивности (только валидные отсчёты). */

import type { Frame, NeffPoint, RunMetrics } from './types'

const dist = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])

/** Навигационные метрики кадра: радиальная оценка времени t_radial = R/Vc (поле tgo),
 *  прогноз промаха при неизменных скоростях h_cv (поле zem; в кинематической модели h₀ ≡ h_cv),
 *  и угол упреждения. */
export function navMetrics(fr: Frame): { tgo: number; zem: number; gamma: number } {
  const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
  const nrm = (a: number[]) => Math.hypot(a[0], a[1], a[2]) || 1e-9
  const r = [fr.target[0] - fr.missile[0], fr.target[1] - fr.missile[1], fr.target[2] - fr.missile[2]]
  const vr = [fr.target_v[0] - fr.missile_v[0], fr.target_v[1] - fr.missile_v[1], fr.target_v[2] - fr.missile_v[2]]
  const vv2 = dot(vr, vr)
  const tgo = vv2 > 1e-9 ? Math.max(0, -dot(r, vr) / vv2) : 0
  const pip = [fr.target[0] + fr.target_v[0] * tgo, fr.target[1] + fr.target_v[1] * tgo, fr.target[2] + fr.target_v[2] * tgo]
  const mze = [fr.missile[0] + fr.missile_v[0] * tgo, fr.missile[1] + fr.missile_v[1] * tgo, fr.missile[2] + fr.missile_v[2] * tgo]
  const zem = nrm([pip[0] - mze[0], pip[1] - mze[1], pip[2] - mze[2]])
  const course = [pip[0] - fr.missile[0], pip[1] - fr.missile[1], pip[2] - fr.missile[2]]
  let gamma = 0
  if (nrm(fr.missile_v) > 1e-9 && nrm(course) > 1e-9) {
    const cs = dot(fr.missile_v, course) / (nrm(fr.missile_v) * nrm(course))
    gamma = (Math.acos(Math.max(-1, Math.min(1, cs))) * 180) / Math.PI
  }
  return { tgo, zem, gamma }
}

/** Проекции команды a_cmd на оси корпуса + перегрузка + сигнал LPLC2 — для «выхода на рули». */
export function controlOutputs(frame: Frame | null, nMax: number): { pitchAcc: number; yawAcc: number; gLoad: number; loomSig: number } {
  const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
  const cr = (a: number[], b: number[]): number[] => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
  const nrm = (a: number[]) => Math.hypot(a[0], a[1], a[2]) || 1e-9
  let pitchAcc = 0
  let yawAcc = 0
  const aC = frame?.a_cmd
  const vM = frame?.missile_v
  if (aC && vM && nrm(vM) > 1e-6) {
    const fwd = [vM[0] / nrm(vM), vM[1] / nrm(vM), vM[2] / nrm(vM)]
    let right = cr([0, 0, 1], fwd)
    if (nrm(right) < 0.05) right = cr([0, 1, 0], fwd)
    const rn = nrm(right)
    right = [right[0] / rn, right[1] / rn, right[2] / rn]
    const up = cr(fwd, right)
    const scale = (nMax || 1) * 9.81
    pitchAcc = Math.max(-1, Math.min(1, dot(aC, up) / scale))
    yawAcc = Math.max(-1, Math.min(1, dot(aC, right) / scale))
  }
  const gLoad = Math.min(1, (frame?.n_req ?? 0) / Math.max(nMax, 1))
  const loomSig = Math.min(1, Math.abs(frame?.circuit.lplc2 ?? 0) * 1.5)
  return { pitchAcc, yawAcc, gLoad, loomSig }
}

function corr(xs: number[], ys: number[]): number | null {
  if (xs.length < 12) return null
  const n = xs.length
  const mx = xs.reduce((s, v) => s + v, 0) / n
  const my = ys.reduce((s, v) => s + v, 0) / n
  let sxy = 0
  let sxx = 0
  let syy = 0
  for (let i = 0; i < n; i += 1) {
    const dx = xs[i] - mx
    const dy = ys[i] - my
    sxy += dx * dy
    sxx += dx * dx
    syy += dy * dy
  }
  if (sxx < 1e-12 || syy < 1e-12) return null
  return sxy / Math.sqrt(sxx * syy)
}

function pct(sorted: number[], p: number): number | null {
  if (!sorted.length) return null
  const idx = (sorted.length - 1) * p
  const lo = Math.floor(idx)
  const hi = Math.ceil(idx)
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo)
}

/** Точки N_экв последнего прогона — для лабораторных графиков адаптивности. */
export function neffPoints(frames: Frame[]): NeffPoint[] {
  return frames.map((fr) => ({
    t: fr.t,
    tgo: fr.tgo ?? null,
    rho: fr.rho ?? 0,
    vc: fr.v_c,
    nEff: fr.n_eff ?? null,
    valid: Boolean(fr.n_eff_valid),
    sat: Boolean(fr.sat),
    speedMode: fr.speed_mode ?? 'constant',
  }))
}

export function computeRunMetrics(frames: Frame[], killRadius: number): RunMetrics {
  let miss = Number.POSITIVE_INFINITY
  let nPeak = 0
  let nInt = 0
  let locks = 0
  let guideFrames = 0
  let devSum = 0
  let devSq = 0
  let devN = 0
  let range0: number | null = null
  let tGuide: number | null = null
  let last: Frame | null = null
  let prevT: number | null = null
  const nEffVals: number[] = []
  const nEffRhos: number[] = []
  const nEffTgos: number[] = []
  let satFrames = 0
  // argmin накопленного CPA — момент наибольшего сближения для η (угол встречи),
  // зеркало navedenie/engine.collect
  let bestStepMiss = Number.POSITIVE_INFINITY
  let etaVm: number[] | null = null
  let etaVt: number[] | null = null
  for (const fr of frames) {
    last = fr
    miss = Math.min(miss, fr.miss)
    if (fr.miss < bestStepMiss - 1e-12) {
      bestStepMiss = fr.miss
      etaVm = fr.missile_v
      etaVt = fr.target_v
    }
    // дубли-кадры терминальных событий не попадают в интегралы дважды
    const terminal = fr.event === 'hit' || fr.event === 'перехват' || fr.event === 'miss_pass' || fr.event === 'промах'
    if (fr.n_eff_valid && fr.n_eff !== null && Number.isFinite(fr.n_eff)) {
      nEffVals.push(fr.n_eff)
      nEffRhos.push(fr.rho ?? 0)
      if (fr.tgo !== null && fr.tgo !== undefined && Number.isFinite(fr.tgo)) nEffTgos.push(fr.tgo)
    }
    if (fr.sat) satFrames += 1
    if (!terminal) {
      nPeak = Math.max(nPeak, fr.n_req)
      if (prevT !== null) nInt += fr.n_req * Math.max(0, fr.t - prevT)
      guideFrames += 1
      if (fr.lock) locks += 1
      if (fr.ghost) {
        const d = [fr.missile[0] - fr.ghost[0], fr.missile[1] - fr.ghost[1], fr.missile[2] - fr.ghost[2]]
        devSum += Math.hypot(d[0], d[1], d[2])
        devSq += d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        devN += 1
        if (range0 === null) range0 = fr.range_m
      }
    }
    prevT = fr.t
    if ((fr.event === 'перехват' || fr.event === 'hit') && tGuide === null) tGuide = fr.t
  }
  const hit = tGuide !== null || (last !== null && last.miss <= killRadius)
  const tEnd = last?.t ?? 0
  const hCv = last ? navMetrics(last).zem : null
  // η — угол между векторами скоростей в момент наибольшего сближения (аргмин накопленного CPA)
  let eta: number | null = null
  if (etaVm && etaVt) {
    const nm = Math.hypot(...etaVm)
    const nt = Math.hypot(...etaVt)
    if (nm > 1e-9 && nt > 1e-9) {
      const cos = (etaVm[0] * etaVt[0] + etaVm[1] * etaVt[1] + etaVm[2] * etaVt[2]) / (nm * nt)
      eta = (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI
    }
  }
  const endRange = last?.range_m ?? null
  const sortedVals = [...nEffVals].sort((a, b) => a - b)
  const tgoPairs = nEffVals.map((v, i) => [v, nEffTgos[i]] as const).filter((p) => Number.isFinite(p[1]))
  return {
    metricsVersion: 4,
    miss: Number.isFinite(miss) ? miss : (last?.miss ?? Number.POSITIVE_INFINITY),
    cpaM: Number.isFinite(miss) ? miss : (last?.miss ?? Number.POSITIVE_INFINITY),
    triggerRangeM: killRadius,
    hCvM: hCv,
    h0M: hCv,
    endRangeM: endRange,
    impactAngleM: eta,
    terminalZem: hCv,
    hit,
    tGuide,
    tEnd,
    nPeak,
    nMean: tEnd > 0 ? nInt / tEnd : 0,
    nInt,
    lockFrac: guideFrames ? Math.min(1, locks / guideFrames) : 0,
    refNrms: devN && range0 && range0 > 0 ? Math.sqrt(devSq / devN) / range0 : null,
    refDev: devN ? devSum / devN : null,
    nEffMedian: pct(sortedVals, 0.5),
    nEffQ25: pct(sortedVals, 0.25),
    nEffQ75: pct(sortedVals, 0.75),
    nEffMin: sortedVals.length ? sortedVals[0] : null,
    nEffMax: sortedVals.length ? sortedVals[sortedVals.length - 1] : null,
    nEffValidFrac: frames.length ? nEffVals.length / frames.length : 0,
    satFrac: frames.length ? satFrames / frames.length : 0,
    corrNEffRho: corr(nEffVals, nEffRhos),
    corrNEffTgo: tgoPairs.length >= 12 ? corr(tgoPairs.map((p) => p[0]), tgoPairs.map((p) => p[1])) : null,
    nEffCount: nEffVals.length,
  }
}
