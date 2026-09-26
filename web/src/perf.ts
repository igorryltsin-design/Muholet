/**
 * Тир производительности устройства — грубая, но дёшево вычисляемая оценка,
 * по которой сцена решает, что можно себе позволить (bloom, длина трасс,
 * число частиц), а что срезать на слабом железе. Никаких сетевых запросов,
 * никаких тяжёлых бенчмарков на старте — только доступные синхронные сигналы
 * плюс самообучение по факту (средний интервал кадра, измеренный самой
 * сценой, оседает в localStorage и учитывается со следующей сессии).
 */

export type PerfTier = 'high' | 'mid' | 'low'

const FPS_KEY = 'muholet-perf-fps'

function readStoredFrameMs(): number | null {
  try {
    const raw = localStorage.getItem(FPS_KEY)
    if (!raw) return null
    const v = Number(raw)
    return Number.isFinite(v) && v > 0 ? v : null
  } catch {
    return null // приватный режим — просто без памяти между сессиями
  }
}

/** Сцена вызывает это по факту нескольких секунд рендера — самообучение тира. */
export function recordFrameMs(ms: number) {
  try {
    localStorage.setItem(FPS_KEY, String(Math.round(ms)))
  } catch {
    /* приватный режим */
  }
}

let cached: PerfTier | null = null

/** Тир считается один раз за сессию (устройство не меняется на лету). */
export function getPerfTier(): PerfTier {
  if (cached) return cached
  const cores = typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 4 : 4
  const mem = (navigator as Navigator & { deviceMemory?: number }).deviceMemory
  const coarse = typeof matchMedia === 'function' && matchMedia('(pointer: coarse)').matches
  const storedMs = readStoredFrameMs() // прошлый живой замер сильнее эвристик по устройству

  let score = 0
  if (cores >= 8) score += 2
  else if (cores >= 4) score += 1
  if (mem === undefined || mem >= 6) score += 1
  if (!coarse) score += 1 // тач-устройства чаще слабее по GPU при том же CPU

  let tier: PerfTier = score >= 3 ? 'high' : score >= 2 ? 'mid' : 'low'
  if (storedMs !== null) {
    // >28мс/кадр (<36 FPS) устойчиво — понижаем тир независимо от эвристики выше
    if (storedMs > 28) tier = 'low'
    else if (storedMs > 18 && tier === 'high') tier = 'mid'
  }
  cached = tier
  return tier
}

/** Параметры эффектов по тиру — единая точка, чтобы сцены не дублировали числа. */
export function perfBudget(tier: PerfTier = getPerfTier()) {
  return {
    trailPoints: tier === 'low' ? 90 : tier === 'mid' ? 180 : 260,
    sparkCount: tier === 'low' ? 0 : tier === 'mid' ? 10 : 18,
    bloomStrength: tier === 'low' ? 0.5 : tier === 'mid' ? 0.75 : 1,
    brainParticles: tier === 'low' ? 0.5 : tier === 'mid' ? 0.75 : 1,
    allowCinematicReplay: tier !== 'low',
  }
}
