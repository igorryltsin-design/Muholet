import { useEffect, useRef, useState } from 'react'
import type { Frame } from './types'

/**
 * Схема зрительного контура дрозофилы: глаз → ламина → медулла (T4/T5) →
 * лобула (LC11/LC10) → LPLC2 и AOTU/грибовидное тело → DN. По связям бегут
 * импульсы, яркость узлов — реальная активность нейронов из кадра.
 */

type RegionKey = 'eye' | 'lamina' | 'medulla' | 'lobula' | 'lplc2' | 'aotu' | 'dn'

type RegionDef = {
  label: string
  sub: string
  tip: string
  x: number
  y: number
  rx: number
  ry: number
}

const REGIONS: Record<RegionKey, RegionDef> = {
  eye: {
    label: 'ГЛАЗ',
    sub: 'кадр 8×8',
    tip: 'Глаз — фасеточный, у дрозофилы ~800 омматидий. Головка ГСН строит кадр 8×8: светлое пятно — цель.',
    x: 0.085,
    y: 0.44,
    rx: 0.05,
    ry: 0.21,
  },
  lamina: {
    label: 'ЛАМИНА',
    sub: 'контраст',
    tip: 'Ламина — первая оптическая пластинка: усиливает контраст и чистит шум, прежде чем образ пойдёт дальше.',
    x: 0.235,
    y: 0.44,
    rx: 0.043,
    ry: 0.13,
  },
  medulla: {
    label: 'МЕДУЛЛА',
    sub: 'T4/T5 · поток',
    tip: 'Медулла: нейроны T4/T5 — детекторы движения. Видят, куда и как быстро ползёт образ цели — оптический поток.',
    x: 0.4,
    y: 0.44,
    rx: 0.05,
    ry: 0.15,
  },
  lobula: {
    label: 'ЛОБУЛА',
    sub: 'LC11/LC10',
    tip: 'Лобула: LC11 — в какой четверти кадра цель, LC10 — смещение по вертикали и горизонтали. «Где противник».',
    x: 0.555,
    y: 0.44,
    rx: 0.05,
    ry: 0.15,
  },
  lplc2: {
    label: 'LPLC2',
    sub: 'приближение',
    tip: 'LPLC2 — радар столкновения: пятно цели растёт — сигнал «цель прёт на муху», самый громкий сигнал тревоги.',
    x: 0.555,
    y: 0.82,
    rx: 0.042,
    ry: 0.095,
  },
  aotu: {
    label: 'AOTU·ГБ',
    sub: 'решение',
    tip: 'AOTU и грибовидное тело — ассоциативный узел: сшивает «где цель» и «насколько страшно» в решение о манёвре.',
    x: 0.715,
    y: 0.15,
    rx: 0.045,
    ry: 0.11,
  },
  dn: {
    label: 'DN',
    sub: 'тангаж·рыскание',
    tip: 'Дауннейроны — последний узел перед мышцами: DN выдают тангаж и рыскание. Два сигнала — две оси рулей ракеты.',
    x: 0.885,
    y: 0.44,
    rx: 0.038,
    ry: 0.14,
  },
}

const LINKS: Array<{ from: RegionKey; to: RegionKey; bend: number }> = [
  { from: 'eye', to: 'lamina', bend: 0 },
  { from: 'lamina', to: 'medulla', bend: 0 },
  { from: 'medulla', to: 'lobula', bend: 0 },
  { from: 'lobula', to: 'lplc2', bend: 0.5 },
  { from: 'lobula', to: 'aotu', bend: -0.5 },
  { from: 'aotu', to: 'dn', bend: 0.5 },
  { from: 'lplc2', to: 'dn', bend: 0.5 },
]

// Детерминированные «нейроны» внутри узла — декоративная популяция точек.
function dotOffsets(key: RegionKey, n: number): Array<{ dx: number; dy: number; s: number }> {
  const out: Array<{ dx: number; dy: number; s: number }> = []
  let seed = key.length * 977 + key.charCodeAt(0) * 31
  const rnd = () => {
    seed = (seed * 16807) % 2147483647
    return seed / 2147483647
  }
  for (let i = 0; i < n; i += 1) {
    const a = rnd() * Math.PI * 2
    const r = Math.sqrt(rnd())
    out.push({ dx: Math.cos(a) * r, dy: Math.sin(a) * r, s: 0.6 + rnd() * 0.8 })
  }
  return out
}

const REGION_DOTS = Object.fromEntries(
  (Object.keys(REGIONS) as RegionKey[]).map((k) => [k, dotOffsets(k, 14)]),
) as Record<RegionKey, Array<{ dx: number; dy: number; s: number }>>

function quadPos(i: number): [number, number] {
  // 0: лево-верх, 1: право-верх, 2: лево-низ, 3: право-низ
  return [[-0.5, -0.45], [0.5, -0.45], [-0.5, 0.45], [0.5, 0.45]][i] as [number, number]
}

export function BrainView({ frame }: { frame: Frame | null }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const frameRef = useRef(frame)
  frameRef.current = frame
  const hoverRef = useRef<RegionKey | null>(null)
  const [hover, setHover] = useState<RegionKey | null>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const fit = () => {
      const dpr = Math.min(2, window.devicePixelRatio || 1)
      const w = Math.max(canvas.clientWidth, 240)
      const h = Math.max(canvas.clientHeight, 140)
      canvas.width = Math.floor(w * dpr)
      canvas.height = Math.floor(h * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    fit()
    const ro = new ResizeObserver(fit)
    ro.observe(canvas)

    const energy = (k: RegionKey): number => {
      const fr = frameRef.current
      const L = (fr?.layers || {}) as Record<string, number>
      const val = (a: string, b: string) => Number(L[a] ?? L[b] ?? 0)
      switch (k) {
        case 'eye':
          return Math.min(1, val('Зрение', 'VISION') * 1.5)
        case 'lamina':
          return Math.min(1, val('Зрение', 'VISION') * 1.3)
        case 'medulla':
          return Math.min(1, val('Поток', 'FLOW') * 1.6)
        case 'lobula':
          return Math.min(1, 0.3 * val('Зрение', 'VISION') + 0.8 * val('Решение', 'DECISION'))
        case 'lplc2':
          return Math.min(1, Math.abs(fr?.circuit.lplc2 ?? 0) * 1.6)
        case 'aotu':
          return Math.min(1, val('Решение', 'DECISION') * 1.6)
        case 'dn':
          return Math.min(1, val('Мотор', 'MOTOR') * 1.6 + Math.abs(fr?.circuit.dn.yaw ?? 0) * 0.4)
      }
    }

    const nodeXY = (k: RegionKey, w: number, h: number): [number, number] => [REGIONS[k].x * w, REGIONS[k].y * h]

    const bezier = (a: [number, number], b: [number, number], bend: number): [number, number, number, number, number, number] => {
      const mx = (a[0] + b[0]) / 2
      const my = (a[1] + b[1]) / 2
      const dx = b[0] - a[0]
      const dy = b[1] - a[1]
      const nx = -dy
      const ny = dx
      const len = Math.hypot(nx, ny) || 1
      const cx = mx + (nx / len) * bend * Math.min(Math.abs(dx), 90) * 0.9
      const cy = my + (ny / len) * bend * Math.min(Math.abs(dx), 90) * 0.9
      return [a[0], a[1], cx, cy, b[0], b[1]]
    }

    const draw = () => {
      const w = canvas.clientWidth || 320
      const h = canvas.clientHeight || 190
      const t = performance.now() / 1000
      const fr = frameRef.current

      ctx.fillStyle = '#071014'
      ctx.fillRect(0, 0, w, h)

      // фоновая сетка
      ctx.strokeStyle = 'rgba(125,255,200,0.045)'
      ctx.lineWidth = 1
      for (let gx = 0; gx < w; gx += 26) {
        ctx.beginPath()
        ctx.moveTo(gx, 0)
        ctx.lineTo(gx, h)
        ctx.stroke()
      }
      for (let gy = 0; gy < h; gy += 26) {
        ctx.beginPath()
        ctx.moveTo(0, gy)
        ctx.lineTo(w, gy)
        ctx.stroke()
      }

      const photo = fr?.circuit.photo

      // ─── связи: линия + бегущие импульсы ───
      for (const link of LINKS) {
        const e = Math.max(energy(link.from), energy(link.to) * 0.7)
        const b = bezier(nodeXY(link.from, w, h), nodeXY(link.to, w, h), link.bend)
        ctx.strokeStyle = `rgba(125,255,200,${0.1 + e * 0.3})`
        ctx.lineWidth = 1 + e * 1.4
        ctx.beginPath()
        ctx.moveTo(b[0], b[1])
        ctx.quadraticCurveTo(b[2], b[3], b[4], b[5])
        ctx.stroke()

        const speed = 0.25 + e * 0.9
        const phase = (t * speed) % 1
        for (let p = 0; p < 3; p += 1) {
          const tt = (phase + p / 3) % 1
          const u = 1 - tt
          const px = u * u * b[0] + 2 * u * tt * b[2] + tt * tt * b[4]
          const py = u * u * b[1] + 2 * u * tt * b[3] + tt * tt * b[5]
          const fade = Math.sin((tt) * Math.PI)
          ctx.fillStyle = `rgba(170,255,222,${(0.15 + e * 0.75) * fade})`
          ctx.beginPath()
          ctx.arc(px, py, 1.4 + e * 1.6, 0, Math.PI * 2)
          ctx.fill()
        }
      }

      // ─── узлы ───
      const idle = frameRef.current ? 0 : 0.06
      for (const key of Object.keys(REGIONS) as RegionKey[]) {
        const def = REGIONS[key]
        const cx = def.x * w
        const cy = def.y * h
        const rx = def.rx * w
        const ry = def.ry * h
        const flicker = idle ? 0.04 * Math.abs(Math.sin(t * 2.2 + def.x * 20)) : 0
        const e = Math.min(1, energy(key) + flicker)
        const hovered = hoverRef.current === key

        // свечение узла
        const glow = ctx.createRadialGradient(cx, cy, 1, cx, cy, Math.max(rx, ry) * 1.5)
        glow.addColorStop(0, `rgba(125,255,200,${0.1 + e * 0.4})`)
        glow.addColorStop(1, 'rgba(125,255,200,0)')
        ctx.fillStyle = glow
        ctx.beginPath()
        ctx.arc(cx, cy, Math.max(rx, ry) * 1.5, 0, Math.PI * 2)
        ctx.fill()

        // корпус — «живой» овал
        ctx.strokeStyle = hovered ? 'rgba(231,193,90,0.9)' : `rgba(125,255,200,${0.3 + e * 0.5})`
        ctx.lineWidth = hovered ? 1.6 : 1
        ctx.beginPath()
        for (let a = 0; a <= 32; a += 1) {
          const th = (a / 32) * Math.PI * 2
          const wob = 1 + Math.sin(th * 3 + key.length) * 0.06
          const px = cx + Math.cos(th) * rx * wob
          const py = cy + Math.sin(th) * ry * wob
          if (a === 0) ctx.moveTo(px, py)
          else ctx.lineTo(px, py)
        }
        ctx.closePath()
        ctx.stroke()

        // нейроны внутри
        for (const d of REGION_DOTS[key]) {
          const tw = idle ? 0.5 + 0.5 * Math.sin(t * 3 + d.dx * 10) : 1
          const alpha = 0.1 + e * 0.75 * (0.5 + 0.5 * d.s) * tw
          ctx.fillStyle = `rgba(150,255,214,${alpha})`
          ctx.beginPath()
          ctx.arc(cx + d.dx * rx * 0.62, cy + d.dy * ry * 0.62, (1.1 + e * 1.5) * d.s, 0, Math.PI * 2)
          ctx.fill()
        }

        // особые визуализации
        if (key === 'eye' && photo?.length) {
          const cols = photo.length
          const rows = photo[0]?.length ?? 8
          const cellW = (rx * 1.5) / cols
          const cellH = (ry * 1.7) / rows
          for (let j = 0; j < rows; j += 1) {
            for (let i = 0; i < cols; i += 1) {
              const v = Number(photo[i]?.[j] ?? 0)
              if (v < 0.04) continue
              ctx.fillStyle = `rgba(231,193,90,${0.25 + v * 0.75})`
              ctx.beginPath()
              ctx.arc(cx - rx * 0.75 + i * cellW + cellW / 2, cy - ry * 0.85 + j * cellH + cellH / 2, Math.min(cellW, cellH) * 0.42, 0, Math.PI * 2)
              ctx.fill()
            }
          }
        }
        if (key === 'medulla') {
          // T4: куда ползёт образ — четыре направления
          const t4 = fr?.circuit.t4 ?? [0, 0, 0, 0]
          const dirs: Array<[number, number]> = [
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1],
          ]
          dirs.forEach(([dx, dy], i) => {
            const v = Math.min(1, Math.abs(Number(t4[i] ?? 0)) * 3)
            if (v < 0.05) return
            const ax = cx + dx * rx * 0.5
            const ay = cy + dy * ry * 0.5
            ctx.strokeStyle = `rgba(231,193,90,${0.3 + v * 0.7})`
            ctx.lineWidth = 1.4
            ctx.beginPath()
            ctx.moveTo(cx + dx * rx * 0.12, cy + dy * ry * 0.12)
            ctx.lineTo(ax, ay)
            ctx.stroke()
            const ang = Math.atan2(dy, dx)
            ctx.beginPath()
            ctx.moveTo(ax, ay)
            ctx.lineTo(ax - Math.cos(ang - 0.5) * 5, ay - Math.sin(ang - 0.5) * 5)
            ctx.lineTo(ax - Math.cos(ang + 0.5) * 5, ay - Math.sin(ang + 0.5) * 5)
            ctx.closePath()
            ctx.fillStyle = `rgba(231,193,90,${0.35 + v * 0.6})`
            ctx.fill()
          })
        }
        if (key === 'lobula') {
          // четверть кадра, где сидит цель — по пеленгу ГСН
          const az = fr?.az ?? 0
          const el = fr?.el ?? 0
          const lock = fr?.lock ?? false
          if (lock) {
            const qx = az >= 0 ? 0.28 : -0.28
            const qy = el >= 0 ? -0.24 : 0.24
            ctx.fillStyle = 'rgba(231,193,90,0.85)'
            ctx.beginPath()
            ctx.arc(cx + qx * rx, cy + qy * ry, 2.6, 0, Math.PI * 2)
            ctx.fill()
            ctx.strokeStyle = 'rgba(231,193,90,0.4)'
            ctx.beginPath()
            ctx.moveTo(cx - rx * 0.5, cy)
            ctx.lineTo(cx + rx * 0.5, cy)
            ctx.moveTo(cx, cy - ry * 0.5)
            ctx.lineTo(cx, cy + ry * 0.5)
            ctx.stroke()
          }
        }
        if (key === 'lplc2') {
          // расширяющееся кольцо — «пятно растёт»
          const loom = fr?.circuit.lplc2 ?? 0
          const pr = ((t * (0.4 + loom * 1.2)) % 1)
          ctx.strokeStyle = `rgba(255,138,106,${(0.15 + loom * 0.6) * (1 - pr)})`
          ctx.lineWidth = 1.6
          ctx.beginPath()
          ctx.arc(cx, cy, rx * (0.3 + pr * 1.4), 0, Math.PI * 2)
          ctx.stroke()
        }
        if (key === 'dn') {
          // выход: две оси рулей
          const pitch = fr?.circuit.dn.pitch ?? 0
          const yaw = fr?.circuit.dn.yaw ?? 0
          ctx.strokeStyle = 'rgba(255,138,106,0.85)'
          ctx.lineWidth = 2
          ctx.beginPath()
          ctx.moveTo(cx - 6, cy - pitch * ry * 0.55)
          ctx.lineTo(cx + 6, cy - pitch * ry * 0.55)
          ctx.moveTo(cx + yaw * rx * 0.55, cy - 6)
          ctx.lineTo(cx + yaw * rx * 0.55, cy + 6)
          ctx.stroke()
        }

        // подписи
        ctx.textAlign = 'center'
        ctx.font = '600 10px "Chakra Petch", sans-serif'
        ctx.fillStyle = hovered ? '#e7c15a' : e > 0.15 ? '#c8ffe8' : '#6d8490'
        ctx.fillText(def.label, cx, cy + ry + 13)
        ctx.font = '9px "Share Tech Mono", monospace'
        ctx.fillStyle = 'rgba(109,132,144,0.85)'
        ctx.fillText(def.sub, cx, cy + ry + 24)
      }

      raf = requestAnimationFrame(draw)
    }
    let raf = 0
    raf = requestAnimationFrame(draw)

    const onMove = (ev: MouseEvent) => {
      const rect = canvas.getBoundingClientRect()
      const mx = ev.clientX - rect.left
      const my = ev.clientY - rect.top
      const w = canvas.clientWidth
      const h = canvas.clientHeight
      let found: RegionKey | null = null
      for (const key of Object.keys(REGIONS) as RegionKey[]) {
        const def = REGIONS[key]
        const dx = (mx - def.x * w) / (def.rx * w * 1.35)
        const dy = (my - def.y * h) / (def.ry * h * 1.35)
        if (dx * dx + dy * dy <= 1) {
          found = key
          break
        }
      }
      if (found !== hoverRef.current) {
        hoverRef.current = found
        setHover(found)
      }
    }
    const onLeave = () => {
      hoverRef.current = null
      setHover(null)
    }
    canvas.addEventListener('mousemove', onMove)
    canvas.addEventListener('mouseleave', onLeave)

    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
      canvas.removeEventListener('mousemove', onMove)
      canvas.removeEventListener('mouseleave', onLeave)
    }
  }, [])

  return (
    <div className="brainwrap">
      <canvas className="viewport viewport--brain" ref={ref} />
      <p className="brainhint">
        {hover ? (
          <>
            <b>{REGIONS[hover].label}</b> — {REGIONS[hover].tip}
          </>
        ) : (
          'Наведите курсор на узел схемы. Пуск — и по цепочке побежит сигнал: глаз → поток → решение → рули.'
        )}
      </p>
    </div>
  )
}
