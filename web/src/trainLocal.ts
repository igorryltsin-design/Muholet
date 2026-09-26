/** Обучаемый выход контура в окне, если сервер молчит: те же метрики эпизодов, что и на сервере. */

import { getW, markTrained, setBrainKind } from './brain'
import { computeRunMetrics } from './metrics'
import { localRun } from './localSim'
import { DEFAULT_SCENARIO, type BrainKind, type Frame, type TrainPoint } from './types'

export function trainLocal(
  kind: BrainKind,
  episodes = 30,
  onProgress?: (ep: number, total: number, point: TrainPoint) => void,
  onFrame?: (fr: Frame) => void,
  lr = 0.04,
): TrainPoint[] {
  setBrainKind(kind)
  const aspects = ['head-on', 'beam', 'tail-chase'] as const
  const maneuvers = ['straight', 'weave', 'turn', 'scissors', 'dive'] as const
  const points: TrainPoint[] = []
  for (let ep = 0; ep < episodes; ep += 1) {
    const sc = {
      ...DEFAULT_SCENARIO_LAB,
      aspect: aspects[ep % 3],
      mode: 'bio' as const,
      t_max: 12,
      off_axis_m: 120 + (ep % 7) * 80,
      range_m: 4500 + (ep % 6) * 900,
      v_t: 180 + (ep % 5) * 30,
      // разнообразие эпизодов: манёвры, перегрузка цели, шумы и срывы захвата —
      // муха учится наводить в разных условиях, а не на одной траектории
      maneuver: ep % 3 === 0 ? 'straight' : maneuvers[ep % maneuvers.length],
      n_target: ep % 3 === 0 ? 0 : 2 + (ep % 3),
      noise_az_deg: [0, 0, 1.5, 2.5][ep % 4],
      lock_drop_p: ep % 5 === 4 ? 0.1 : 0,
    }
    const frames: Frame[] = []
    let i = 0
    for (const fr of localRun(sc, { learn: true, lr })) {
      frames.push(fr)
      // анимация обучения: каждый четвёртый кадр эпизода — глаз/мозг/рули живут
      if (onFrame && i % 4 === 0) onFrame(fr)
      i += 1
    }
    const m = computeRunMetrics(frames, sc.kill_radius_m)
    const W = getW()
    const point: TrainPoint = {
      ep: ep + 1,
      miss: m.miss,
      hit: m.hit,
      tGuide: m.tGuide,
      refDev: m.refDev ?? 0,
      nAvg: m.nInt,
      nPeak: m.nPeak,
      lockFrac: m.lockFrac,
      nrms: m.refNrms ?? 0,
      w: [...W[0], ...W[1]],
    }
    points.push(point)
    onProgress?.(ep + 1, episodes, point)
  }
  markTrained()
  return points
}

// сценарий-подложка для эпизодов: стартовые условия из дефолта, геометрию рулит цикл
const DEFAULT_SCENARIO_LAB = {
  ...DEFAULT_SCENARIO,
  range_m: 6000,
}
