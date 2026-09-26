import { useEffect, useRef, useState } from 'react'

/**
 * Плавно доводит показанное число до реального значения за `ms` — цифры
 * телеметрии не «щёлкают» на кадровом шаге плейбека (см. App.tsx::playFrames,
 * dt=40/80мс), а плывут. Чистый rAF-твин, без библиотек.
 *
 * prefers-reduced-motion: значение снапается мгновенно, rAF не запускается —
 * глобальное CSS-правило (styles.css) гасит переходы/анимации, но не эту
 * JS-логику, поэтому проверка здесь обязательна.
 */
export function useTweenedNumber(value: number, ms = 180): number {
  const [shown, setShown] = useState(value)
  const fromRef = useRef(value)
  const rafRef = useRef(0)

  useEffect(() => {
    if (!Number.isFinite(value)) return
    const reduced = (() => {
      try {
        return matchMedia('(prefers-reduced-motion: reduce)').matches
      } catch {
        return false
      }
    })()
    if (reduced) {
      fromRef.current = value
      setShown(value)
      return
    }
    const from = fromRef.current
    if (from === value) return
    const t0 = performance.now()
    const tick = () => {
      const p = Math.min(1, (performance.now() - t0) / ms)
      const eased = 1 - (1 - p) * (1 - p) // ease-out — быстрый старт, мягкий подход к цели
      const cur = from + (value - from) * eased
      setShown(cur)
      if (p < 1) {
        rafRef.current = requestAnimationFrame(tick)
      } else {
        fromRef.current = value
      }
    }
    cancelAnimationFrame(rafRef.current)
    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, ms])

  return shown
}
