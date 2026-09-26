import { useEffect, useRef } from 'react'
import { foveaAngles } from './localSim'
import type { Frame } from './types'

/** Кадр ГСН: фовеальная сетчатка 16×16 — плотный центр, редкая периферия.
 *  Светлое пятно — цель; рамка — захват. Ячейки рисуются на своих УГЛОВЫХ
 *  позициях (нелинейный центр), днём — тёмными чернилами на светлом. */

export function SeekerView({ frame, day, bioFovDeg }: { frame: Frame | null; day?: boolean; bioFovDeg?: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const frameRef = useRef(frame)
  const dayRef = useRef(day ?? false)
  const fovRef = useRef(bioFovDeg ?? 165)
  frameRef.current = frame
  dayRef.current = day ?? false
  fovRef.current = bioFovDeg ?? 165

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const fit = () => {
      const dpr = Math.min(2, window.devicePixelRatio || 1)
      const w = Math.max(canvas.clientWidth, 80)
      const h = Math.max(canvas.clientHeight, 80)
      canvas.width = Math.floor(w * dpr)
      canvas.height = Math.floor(h * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    fit()
    const ro = new ResizeObserver(fit)
    ro.observe(canvas)
    let raf = 0
    const draw = () => {
      const w = canvas.clientWidth || 120
      const h = canvas.clientHeight || 120
      const t = performance.now() / 1000
      const fr = frameRef.current
      const dayNow = dayRef.current
      ctx.fillStyle = dayNow ? '#e8eee9' : '#05080a'
      ctx.fillRect(0, 0, w, h)

      const img = fr?.seeker.image
      const pad = 6
      ctx.fillStyle = dayNow ? '#e8eee9' : '#05080a'
      ctx.fillRect(0, 0, w, h)

      // размер сетчатки: вложенный массив 16×16 → длина СТРОКИ; плоский → корень
      const n = img
        ? Math.max(1, Math.round(Array.isArray(img[0]) ? (img[0] as number[]).length : Math.sqrt(img.length)))
        : 16
      // фовеальная геометрия: линии сетки на угловых позициях (плотный центр, редкий край)
      const halfFov = ((Math.max(fovRef.current, 2) * Math.PI) / 180) / 2
      const ax = foveaAngles(halfFov, n)
      const axMin = ax[0]
      const axMax = ax[n - 1]
      const cx = (i: number) => {
        const a = i === 0 ? axMin : i === n ? axMax : (ax[i - 1] + ax[i]) / 2
        return pad + ((a - axMin) / (axMax - axMin)) * (w - pad * 2)
      }
      const cy = (j: number) => {
        const a = j === 0 ? axMin : j === n ? axMax : (ax[j - 1] + ax[j]) / 2
        return pad + ((a - axMin) / (axMax - axMin)) * (h - pad * 2)
      }
      let hot = { i: -1, j: -1, v: 0 }
      for (let j = 0; j < n; j += 1) {
        for (let i = 0; i < n; i += 1) {
          const v = Number(img?.[j]?.[i] ?? img?.[j * n + i] ?? 0)
          if (v > hot.v) hot = { i, j, v }
          if (dayNow) {
            const g = Math.round(235 - v * 210)
            ctx.fillStyle = `rgb(${Math.round(g * 0.45)}, ${g}, ${Math.round(g * 0.72)})`
          } else {
            const g = Math.round(16 + v * 230)
            ctx.fillStyle = `rgb(${Math.round(g * 0.32)}, ${g}, ${Math.round(g * 0.66)})`
          }
          const x0 = cx(i)
          const x1 = cx(i + 1)
          const y0 = cy(j)
          const y1 = cy(j + 1)
          ctx.fillRect(x0 + 0.5, y0 + 0.5, Math.max(1, x1 - x0 - 1), Math.max(1, y1 - y0 - 1))
        }
      }

      // рамка-омматидии и центральная марка
      ctx.strokeStyle = dayNow ? 'rgba(10,122,90,0.4)' : 'rgba(125,255,200,0.25)'
      ctx.lineWidth = 1
      ctx.strokeRect(pad + 0.5, pad + 0.5, w - pad * 2 - 1, h - pad * 2 - 1)
      ctx.strokeStyle = dayNow ? 'rgba(10,122,90,0.45)' : 'rgba(125,255,200,0.3)'
      ctx.beginPath()
      ctx.moveTo(w / 2 - 6, h / 2)
      ctx.lineTo(w / 2 + 6, h / 2)
      ctx.moveTo(w / 2, h / 2 - 6)
      ctx.lineTo(w / 2, h / 2 + 6)
      ctx.stroke()

      if (fr?.lock && hot.v > 0.25) {
        const bx = cx(hot.i)
        const by = cy(hot.j)
        const cw = cx(hot.i + 1) - bx
        const ch = cy(hot.j + 1) - by
        ctx.strokeStyle = dayNow ? '#8a6a1a' : '#e7c15a'
        ctx.lineWidth = 1.4
        const c = 4
        ;[
          [bx, by, 1, 1],
          [bx + cw, by, -1, 1],
          [bx, by + ch, 1, -1],
          [bx + cw, by + ch, -1, -1],
        ].forEach(([x, y, sx, sy]) => {
          ctx.beginPath()
          ctx.moveTo(x + sx * c, y)
          ctx.lineTo(x, y)
          ctx.lineTo(x, y + sy * c)
          ctx.stroke()
        })
      } else {
        // поиск: бегущая полоса развёртки
        const sy = pad + ((t * 40) % (h - pad * 2))
        ctx.fillStyle = dayNow ? 'rgba(10,122,90,0.12)' : 'rgba(125,255,200,0.07)'
        ctx.fillRect(pad, sy, w - pad * 2, 5)
      }

      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
    }
  }, [])

  return <canvas className="viewport viewport--seeker" ref={ref} />
}
