/**
 * Движок canvas-графиков лаборатории: интерактивные полотна (колесо — масштаб,
 * перетаскивание — сдвиг, двойной клик — сброс), кнопки PNG/CSV на полотне.
 * Палитра читается из CSS design-токенов — темы не дублируются в JS.
 */
import { useEffect } from 'react'

export type Series = { data: number[]; color: string; dash?: number[]; label: string; width?: number }

/** Рисовать при каждом рендере и на ресайз окна (масштаб графиков живёт в WeakMap по canvas). */
export function useAutoRedraw(draw: () => void) {
  useEffect(() => {
    draw()
    window.addEventListener('resize', draw)
    return () => window.removeEventListener('resize', draw)
  })
}

type Palette = {
  accent: string
  amber: string
  red: string
  dim: string
  text: string
  grid: string
  minmax: string
  cursor: string
}

let cachedPalette: { themeKey: string; value: Palette } | null = null

/** Палитра из design-токенов темы. Кэш привязан к CSS-классам темы, поэтому
 * переключение день/ночь не может на несколько секунд оставить старые цвета
 * на canvas — это было заметно как светлые/тёмные артефакты. */
export function palette(): Palette {
  const themeKey = `${document.documentElement.className}|${document.body.className}`
  if (cachedPalette?.themeKey === themeKey) return cachedPalette.value
  const cs = getComputedStyle(document.body)
  const v = (name: string, fallback: string) => cs.getPropertyValue(name).trim() || fallback
  const value: Palette = {
    accent: v('--accent-primary', '#45e0a4'),
    amber: v('--accent-secondary', '#e3bd58'),
    red: v('--status-danger', '#ff7a5e'),
    dim: v('--text-muted', '#85a094'),
    text: v('--text-primary', '#e6f2ec'),
    grid: v('--chart-grid', 'rgba(148,190,178,0.18)'),
    minmax: v('--chart-minmax', '#6b8a80'),
    cursor: v('--chart-cursor', 'rgba(230,242,236,0.45)'),
  }
  cachedPalette = { themeKey, value }
  return value
}

export const CHART = { accent: () => palette().accent, amber: () => palette().amber, red: () => palette().red, minmax: () => palette().minmax }

export function download(name: string, text: string, mime = 'text/csv;charset=utf-8') {
  const blob = new Blob([text], { type: mime })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = name
  a.click()
  URL.revokeObjectURL(a.href)
}

export function exportCsv(name: string, head: string, rows: (string | number)[][]) {
  download(name, [head, ...rows.map((r) => r.join(','))].join('\n'))
}

/** Скользящее среднее по окну — тренд сквозь шум эпизодов. */
export function movingAvg(data: (number | null)[], window = 7): (number | null)[] {
  const out: (number | null)[] = []
  for (let i = 0; i < data.length; i += 1) {
    const acc: number[] = []
    for (let j = Math.max(0, i - window + 1); j <= i; j += 1) {
      const v = data[j]
      if (v !== null && Number.isFinite(v)) acc.push(v)
    }
    out.push(acc.length ? acc.reduce((s, v) => s + v, 0) / acc.length : null)
  }
  return out
}

export function sessionBorders(train: { session?: number }[]): number[] {
  const out: number[] = []
  for (let i = 1; i < train.length; i += 1) {
    if (train[i].session !== undefined && train[i].session !== train[i - 1].session) out.push(i)
  }
  return out
}

/** Перцентиль отсортированного массива (для сводки N_экв). */
export function pctSorted(sorted: number[], p: number): number | null {
  if (!sorted.length) return null
  const idx = (sorted.length - 1) * p
  const lo = Math.floor(idx)
  const hi = Math.ceil(idx)
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo)
}

/** Пирсон-корреляция (null — выборка мала или вырождена). */
export function pearson(xs: number[], ys: number[]): number | null {
  if (xs.length < 12 || xs.length !== ys.length) return null
  const n = xs.length
  const mx = xs.reduce((s, v) => s + v, 0) / n
  const my = ys.reduce((s, v) => s + v, 0) / n
  let sxy = 0
  let sxx = 0
  let syy = 0
  for (let i = 0; i < n; i += 1) {
    sxy += (xs[i] - mx) * (ys[i] - my)
    sxx += (xs[i] - mx) ** 2
    syy += (ys[i] - my) ** 2
  }
  if (sxx < 1e-12 || syy < 1e-12) return null
  return sxy / Math.sqrt(sxx * syy)
}

type ChartState = {
  series: Series[]
  title: string
  vlines: number[]
  x0: number
  x1: number
  sig: string
  dragPx: number | null
  hoverPx: number | null
  btns: Array<{ x: number; w: number; label: string; kind: 'png' | 'csv' | 'reset' }>
}

const chartStates = new WeakMap<HTMLCanvasElement, ChartState>()
const chartAttached = new WeakSet<HTMLCanvasElement>()
// Подписи графиков — часть основного научного интерфейса. Поля рассчитаны под
// читаемые 12–13 px на Full HD, без необходимости увеличивать масштаб браузера.
const CPAD = { l: 56, r: 12, t: 30, b: 26 }

function fitCanvasText(g: CanvasRenderingContext2D, text: string, maxWidth: number) {
  if (g.measureText(text).width <= maxWidth) return text
  let lo = 0
  let hi = text.length
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2)
    if (g.measureText(`${text.slice(0, mid)}…`).width <= maxWidth) lo = mid
    else hi = mid - 1
  }
  return `${text.slice(0, lo).trimEnd()}…`
}

function renderChart(cv: HTMLCanvasElement, st: ChartState) {
  const P = palette()
  const dpr = Math.min(2, window.devicePixelRatio || 1)
  const w = Math.max(cv.clientWidth, 120)
  const h = Math.max(cv.clientHeight, 80)
  cv.width = Math.floor(w * dpr)
  cv.height = Math.floor(h * dpr)
  const g = cv.getContext('2d')!
  g.setTransform(dpr, 0, 0, dpr, 0, 0)
  g.clearRect(0, 0, w, h)

  g.font = '13px "Share Tech Mono", monospace'
  g.fillStyle = P.dim
  g.textAlign = 'left'
  // Справа зарезервировано место для PNG/CSV/reset. Длинные научные названия
  // завершаются многоточием, а не обрезаются границей соседней карточки.
  g.fillText(fitCanvasText(g, st.title, Math.max(80, w - CPAD.l - CPAD.r - 112)), CPAD.l, 17)

  const n = Math.max(...st.series.map((s) => s.data.length), 1)
  if (!Number.isFinite(st.x1)) st.x1 = n - 1
  st.x0 = Math.max(0, Math.min(st.x0, n - 2))
  st.x1 = Math.max(st.x0 + 3, Math.min(st.x1, n - 1))
  const i0 = Math.max(0, Math.floor(st.x0))
  const i1 = Math.min(n - 1, Math.ceil(st.x1))
  const plotW = w - CPAD.l - CPAD.r
  const plotH = h - CPAD.t - CPAD.b
  const px = (gi: number) => CPAD.l + ((gi - st.x0) / Math.max(st.x1 - st.x0, 1e-9)) * plotW

  const vis = (s: Series) => s.data.slice(i0, i1 + 1).filter((v) => Number.isFinite(v))
  const all = st.series.flatMap(vis).filter((v) => Number.isFinite(v))
  if (all.length < 1) {
    g.fillStyle = P.dim
    g.textAlign = 'center'
    g.font = '13px "Share Tech Mono", monospace'
    g.fillText('нет данных', w / 2, h / 2)
    return
  }
  let lo = Math.min(...all)
  let hi = Math.max(...all)
  if (hi - lo < 1e-9) {
    hi += 1
    lo -= 1
  }
  const spanY = hi - lo
  const py = (v: number) => CPAD.t + plotH * (1 - (v - lo) / spanY)

  // сетка + шкала Y
  g.strokeStyle = P.grid
  g.lineWidth = 1
  g.font = '12px "Share Tech Mono", monospace'
  for (let i = 0; i <= 4; i += 1) {
    const y = CPAD.t + (plotH * i) / 4
    g.beginPath()
    g.moveTo(CPAD.l, y)
    g.lineTo(w - CPAD.r, y)
    g.stroke()
    const val = hi - (spanY * i) / 4
    g.fillStyle = P.dim
    g.textAlign = 'right'
    g.fillText(Math.abs(val) >= 100 ? val.toFixed(0) : val.toFixed(1), CPAD.l - 4, y + 3)
  }

  // границы сессий
  for (const idx of st.vlines) {
    if (idx < i0 || idx > i1) continue
    g.strokeStyle = P.amber + '77'
    g.setLineDash([3, 3])
    g.beginPath()
    g.moveTo(px(idx), CPAD.t)
    g.lineTo(px(idx), h - CPAD.b)
    g.stroke()
    g.setLineDash([])
  }

  // серии
  for (const s of st.series) {
    if (s.data.length === 0) continue
    g.strokeStyle = s.color
    g.lineWidth = s.width ?? 1.4
    g.setLineDash(s.dash ?? [])
    g.beginPath()
    let started = false
    for (let i = i0; i <= i1 && i < s.data.length; i += 1) {
      const v = s.data[i]
      if (!Number.isFinite(v)) {
        started = false
        continue
      }
      const X = px(i)
      const Y = py(v)
      if (!started) {
        g.moveTo(X, Y)
        started = true
      } else g.lineTo(X, Y)
    }
    g.stroke()
    g.setLineDash([])
    if (s.data.length <= 64) {
      g.fillStyle = s.color
      s.data.forEach((v, i) => {
        if (i < i0 || i > i1 || !Number.isFinite(v)) return
        g.beginPath()
        g.arc(px(i), py(v), 2, 0, Math.PI * 2)
        g.fill()
      })
    }
  }

  // курсор: вертикаль + значения серий в ближайшей точке
  if (st.hoverPx !== null && st.hoverPx >= CPAD.l && st.hoverPx <= w - CPAD.r) {
    const gi = Math.round(st.x0 + ((st.hoverPx - CPAD.l) / plotW) * (st.x1 - st.x0))
    if (gi >= i0 && gi <= i1) {
      g.strokeStyle = P.cursor
      g.beginPath()
      g.moveTo(px(gi), CPAD.t)
      g.lineTo(px(gi), h - CPAD.b)
      g.stroke()
      const parts: string[] = [`i=${gi}`]
      for (const s of st.series) {
        const v = s.data[gi]
        if (Number.isFinite(v)) parts.push(`${s.label} ${Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1)}`)
      }
      g.fillStyle = P.text
      g.textAlign = 'left'
      g.font = '12px "Share Tech Mono", monospace'
      g.fillText(parts.join(' · '), Math.min(px(gi) + 6, w - 260), CPAD.t + 12)
    }
  }

  g.fillStyle = P.dim
  g.textAlign = 'left'
  g.font = '12px "Share Tech Mono", monospace'
  g.fillText(`n=${n} · колесо — масштаб · тянуть — сдвиг · 2×клик — сброс`, CPAD.l, h - 4)

  // кнопки: PNG / CSV / сброс
  st.btns = [
    { x: w - CPAD.r - 40, w: 40, label: 'PNG', kind: 'png' as const },
    { x: w - CPAD.r - 84, w: 40, label: 'CSV', kind: 'csv' as const },
    { x: w - CPAD.r - 104, w: 20, label: '⟲', kind: 'reset' as const },
  ]
  for (const b of st.btns) {
    g.strokeStyle = P.grid
    g.lineWidth = 1
    g.strokeRect(b.x, 6, b.w, 19)
    g.fillStyle = P.dim
    g.textAlign = 'center'
    g.fillText(b.label, b.x + b.w / 2, 20)
  }
  g.textAlign = 'left'
}

export function drawChart(cv: HTMLCanvasElement, series: Series[], title: string, vlines: number[] = []) {
  let st = chartStates.get(cv)
  const sig = title
  if (!st || st.sig !== sig) {
    st = { series, title, vlines, x0: 0, x1: NaN, sig, dragPx: null, hoverPx: null, btns: [] }
  } else {
    st.series = series
    st.title = title
    st.vlines = vlines
  }
  chartStates.set(cv, st)
  renderChart(cv, st)
  if (!chartAttached.has(cv)) {
    chartAttached.add(cv)
    const stGet = () => chartStates.get(cv)
    const redraw = () => {
      const s = stGet()
      if (s) renderChart(cv, s)
    }
    const dlBlob = (name: string, blob: Blob) => {
      const a = document.createElement('a')
      a.href = URL.createObjectURL(blob)
      a.download = name
      a.click()
      URL.revokeObjectURL(a.href)
    }
    cv.addEventListener('wheel', (e) => {
      const s = stGet()
      if (!s) return
      e.preventDefault()
      const rect = cv.getBoundingClientRect()
      const plotW = rect.width - CPAD.l - CPAD.r
      const u = Math.min(1, Math.max(0, (e.clientX - rect.left - CPAD.l) / plotW))
      const n = Math.max(...s.series.map((x) => x.data.length), 2)
      if (!Number.isFinite(s.x1)) s.x1 = n - 1
      const c = s.x0 + (s.x1 - s.x0) * u
      const factor = e.deltaY > 0 ? 1.2 : 1 / 1.2
      let a = c - (c - s.x0) * factor
      let b = c + (s.x1 - c) * factor
      a = Math.max(0, a)
      b = Math.min(n - 1, b)
      if (b - a < 4) {
        const mid = (a + b) / 2
        a = mid - 2
        b = mid + 2
      }
      s.x0 = a
      s.x1 = b
      redraw()
    }, { passive: false })
    cv.addEventListener('mousedown', (e) => {
      const s = stGet()
      if (!s) return
      const rect = cv.getBoundingClientRect()
      const mx = e.clientX - rect.left
      if (s.btns.some((b) => mx >= b.x && mx <= b.x + b.w && e.clientY - rect.top >= 6 && e.clientY - rect.top <= 25)) return
      s.dragPx = mx
    })
    window.addEventListener('mousemove', (e) => {
      const s = stGet()
      if (!s) return
      const rect = cv.getBoundingClientRect()
      const mx = e.clientX - rect.left
      if (s.dragPx !== null) {
        const plotW = rect.width - CPAD.l - CPAD.r
        const n = Math.max(...s.series.map((x) => x.data.length), 2)
        const dIdx = ((mx - s.dragPx) / plotW) * (s.x1 - s.x0)
        s.x0 += dIdx
        s.x1 += dIdx
        const width = s.x1 - s.x0
        if (s.x0 < 0) { s.x0 = 0; s.x1 = width }
        if (s.x1 > n - 1) { s.x1 = n - 1; s.x0 = n - 1 - width }
        s.dragPx = mx
        redraw()
      } else if (mx >= CPAD.l && mx <= rect.width - CPAD.r && e.clientY >= rect.top && e.clientY <= rect.bottom) {
        s.hoverPx = mx
        redraw()
      }
    })
    window.addEventListener('mouseup', () => {
      const s = stGet()
      if (s) s.dragPx = null
    })
    cv.addEventListener('mouseleave', () => {
      const s = stGet()
      if (s) { s.hoverPx = null; redraw() }
    })
    cv.addEventListener('dblclick', () => {
      const s = stGet()
      if (s) { s.x0 = 0; s.x1 = NaN; redraw() }
    })
    cv.addEventListener('click', (e) => {
      const s = stGet()
      if (!s) return
      const rect = cv.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top
      for (const b of s.btns) {
        if (mx >= b.x && mx <= b.x + b.w && my >= 6 && my <= 25) {
          const stamp = new Date().toISOString().slice(11, 19).replace(/:/g, '')
          if (b.kind === 'png') {
            cv.toBlob((blob) => blob && dlBlob(`muholet-${stamp}.png`, blob))
          } else if (b.kind === 'csv') {
            const head = ['i', ...s.series.map((x) => x.label)].join(',')
            const n = Math.max(...s.series.map((x) => x.data.length))
            const rows: string[] = []
            for (let i = 0; i < n; i += 1) {
              rows.push([String(i), ...s.series.map((x) => (Number.isFinite(x.data[i]) ? String(x.data[i]) : ''))].join(','))
            }
            dlBlob(`muholet-${stamp}.csv`, new Blob([[head, ...rows].join('\n')], { type: 'text/csv' }))
          } else {
            s.x0 = 0
            s.x1 = NaN
            redraw()
          }
          return
        }
      }
    })
  }
}

/** План-вид X-Y: вид сверху на траектории последнего прогона (ракета / эталон / цель). */
export function drawPlanView(cv: HTMLCanvasElement, traj: { missile: number[][]; ghost: number[][]; target: number[][] } | null) {
  const P = palette()
  const dpr = Math.min(2, window.devicePixelRatio || 1)
  const w = Math.max(cv.clientWidth, 120)
  const h = Math.max(cv.clientHeight, 80)
  cv.width = Math.floor(w * dpr)
  cv.height = Math.floor(h * dpr)
  const g = cv.getContext('2d')!
  g.setTransform(dpr, 0, 0, dpr, 0, 0)
  g.clearRect(0, 0, w, h)
  g.font = '12px "Share Tech Mono", monospace'
  g.fillStyle = P.dim
  g.textAlign = 'left'
  g.fillText('План-вид: вид сверху, X — дальность, Y — бок, м', 10, 13)
  if (!traj || traj.missile.length < 2) {
    g.fillStyle = P.dim
    g.textAlign = 'center'
    g.font = '13px "Share Tech Mono", monospace'
    g.fillText('нет траекторий — сделайте пуск', w / 2, h / 2)
    return
  }
  const all = [...traj.missile, ...traj.target]
  const xs = all.map((p) => p[0])
  const ys = all.map((p) => p[1])
  let x0 = Math.min(...xs)
  let x1 = Math.max(...xs)
  let y0 = Math.min(...ys)
  let y1 = Math.max(...ys)
  const pad = Math.max(x1 - x0, y1 - y0, 1) * 0.08
  x0 -= pad
  x1 += pad
  y0 -= pad
  y1 += pad
  const PX = (x: number) => 10 + ((x - x0) / (x1 - x0)) * (w - 20)
  const PY = (y: number) => h - 10 - ((y - y0) / (y1 - y0)) * (h - 22)
  const poly = (pts: number[][], color: string, dash: number[]) => {
    if (pts.length < 2) return
    g.strokeStyle = color
    g.lineWidth = 1.6
    g.setLineDash(dash)
    g.beginPath()
    pts.forEach((p, i) => (i ? g.lineTo(PX(p[0]), PY(p[1])) : g.moveTo(PX(p[0]), PY(p[1]))))
    g.stroke()
    g.setLineDash([])
  }
  poly(traj.target, P.red, [3, 3])
  poly(traj.ghost, P.amber, [6, 4])
  poly(traj.missile, P.accent, [])
  // старт-метки и легенда
  g.fillStyle = P.accent
  g.beginPath()
  g.arc(PX(traj.missile[0][0]), PY(traj.missile[0][1]), 3, 0, Math.PI * 2)
  g.fill()
  g.fillStyle = P.red
  g.beginPath()
  g.arc(PX(traj.target[traj.target.length - 1][0]), PY(traj.target[traj.target.length - 1][1]), 3, 0, Math.PI * 2)
  g.fill()
  g.textAlign = 'right'
  g.fillStyle = P.accent
  g.fillText('— ракета', w - 10, h - 22)
  g.fillStyle = P.amber
  g.fillText('– – эталон МПС', w - 10, h - 10)
}

/** Гистограмма: ряд bin-ов {lo, hi, n} на числовой оси X (для Monte-Carlo рассеивания). */
export function drawHist(
  cv: HTMLCanvasElement,
  bins: { lo: number; hi: number; n: number }[],
  title: string,
  marker?: { x: number; label: string; color?: string },
) {
  const P = palette()
  const dpr = Math.min(2, window.devicePixelRatio || 1)
  const w = Math.max(cv.clientWidth, 120)
  const h = Math.max(cv.clientHeight, 80)
  cv.width = Math.floor(w * dpr)
  cv.height = Math.floor(h * dpr)
  const g = cv.getContext('2d')!
  g.setTransform(dpr, 0, 0, dpr, 0, 0)
  g.clearRect(0, 0, w, h)
  g.font = '13px "Share Tech Mono", monospace'
  g.fillStyle = P.dim
  g.textAlign = 'left'
  g.fillText(fitCanvasText(g, title, Math.max(80, w - CPAD.l - CPAD.r)), CPAD.l, 17)
  if (!bins.length) {
    g.textAlign = 'center'
    g.fillText('нет данных', w / 2, h / 2)
    return
  }
  const x0 = bins[0].lo
  const x1 = bins[bins.length - 1].hi
  const nMax = Math.max(...bins.map((b) => b.n), 1)
  const plotW = w - CPAD.l - CPAD.r
  const plotH = h - CPAD.t - CPAD.b
  const px = (v: number) => CPAD.l + ((v - x0) / Math.max(x1 - x0, 1e-9)) * plotW
  const py = (v: number) => CPAD.t + plotH - (v / nMax) * plotH
  g.strokeStyle = P.grid
  g.lineWidth = 1
  for (const t of [0, 0.5, 1]) {
    g.beginPath()
    g.moveTo(CPAD.l, py(t * nMax))
    g.lineTo(w - CPAD.r, py(t * nMax))
    g.stroke()
    g.fillStyle = P.dim
    g.textAlign = 'right'
    g.fillText(String(Math.round(t * nMax)), CPAD.l - 6, py(t * nMax) + 4)
  }
  g.fillStyle = P.accent
  for (const b of bins) {
    const bx = px(b.lo)
    const bw = Math.max(1, px(b.hi) - bx - 1)
    const by = py(b.n)
    g.globalAlpha = 0.85
    g.fillRect(bx, by, bw, CPAD.t + plotH - by)
    g.globalAlpha = 1
  }
  if (marker && Number.isFinite(marker.x) && marker.x >= x0 && marker.x <= x1) {
    g.strokeStyle = marker.color ?? P.red
    g.setLineDash([4, 3])
    g.beginPath()
    g.moveTo(px(marker.x), CPAD.t)
    g.lineTo(px(marker.x), CPAD.t + plotH)
    g.stroke()
    g.setLineDash([])
    g.fillStyle = marker.color ?? P.red
    g.textAlign = 'left'
    g.fillText(marker.label, Math.min(px(marker.x) + 4, w - CPAD.r - 90), CPAD.t + 12)
  }
  g.fillStyle = P.dim
  g.textAlign = 'center'
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    const v = x0 + t * (x1 - x0)
    g.fillText(Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(1)} км` : `${v.toFixed(0)} м`, px(v), h - 10)
  }
}
