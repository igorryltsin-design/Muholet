/**
 * Тактильный отклик через navigator.vibrate — короткие паттерны на ключевые
 * события (перехват/промах, захват цели, вход в кинорежим, шаг тура).
 * API нет в Safari/iOS вовсе (не полифиллим — честная ограниченность, не
 * притворяемся вибро-моторчиком, которого нет); на прочих платформах без
 * поддержки — тихий no-op. Уважает prefers-reduced-motion: вибро — незапрошенный
 * стимул, приравниваем к «лишней анимации».
 */

let userEnabled = true

export function setHapticsEnabled(v: boolean) {
  userEnabled = v
}

function reducedMotion(): boolean {
  try {
    return matchMedia('(prefers-reduced-motion: reduce)').matches
  } catch {
    return false
  }
}

export function buzz(pattern: number | readonly number[]) {
  if (!userEnabled || reducedMotion()) return
  try {
    if (typeof navigator !== 'undefined' && 'vibrate' in navigator) {
      navigator.vibrate(typeof pattern === 'number' ? pattern : [...pattern])
    }
  } catch {
    /* устройство отклонило запрос — не критично */
  }
}

export const HAPTIC = {
  hit: [20, 40, 80],
  miss: [15],
  grab: 8,
  cinemaToggle: 12,
  tourStep: 10,
} as const
