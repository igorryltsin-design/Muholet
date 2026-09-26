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
  // текущее нарисованное значение, обновляется НА КАЖДОМ тике (не только на
  // завершении) — старт следующего твина берётся отсюда. Кадры телеметрии
  // приходят чаще (40/80мс), чем длится твин (180мс): без этого «from»
  // застревал на значении до самого первого прерванного твина, и число не
  // доезжало до цели, а на каждом новом кадре срывалось назад к этой точке
  const shownRef = useRef(value)
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
      shownRef.current = value
      setShown(value)
      return
    }
    const from = shownRef.current
    if (from === value) return
    const t0 = performance.now()
    const tick = () => {
      const p = Math.min(1, (performance.now() - t0) / ms)
      const eased = 1 - (1 - p) * (1 - p) // ease-out — быстрый старт, мягкий подход к цели
      const cur = from + (value - from) * eased
      shownRef.current = cur
      setShown(cur)
      if (p < 1) rafRef.current = requestAnimationFrame(tick)
    }
    cancelAnimationFrame(rafRef.current)
    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, ms])

  return shown
}
