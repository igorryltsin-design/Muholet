import { useEffect, useRef, useState } from 'react'
import type { Scenario } from '../types'

/**
 * «Свободная расстановка» (aspect='free'): ручной сценарий перехвата.
 * Два вида в одном масштабе по X: план X–Y (вид сверху) и профиль X–H (сбоку).
 * Цель перетаскивается в обоих видах; курс (азимут) на плане и подъём в
 * профиле — тягой за кончик стрелки. Формулировка зеркалит sim.py::spawn:
 * азимут от +X против часовой к +Y, подъём над горизонтом, град.
 */

type SetFn = <K extends keyof Scenario>(key: K, value: Scenario[K]) => void

const DEG = Math.PI / 180
/** Квант масштаба из ряда 1/2/5·10ⁿ (м), min 2000 — картинка не «дышит» при drag. */
function quant(v: number): number {
  const need = Math.max(2000, v)
  for (let p = 1000; p <= 1e6; p *= 10) for (const m of [1, 2, 5]) { const q = p * m; if (q >= need) return q }
  return 2e6
}
const snap = (v: number) => Math.round(v / 50) * 50
const normHdg = (d: number) => { let x = d % 360; if (x < 0) x += 360; return Math.round(x * 2) / 2 }
const clipClimb = (d: number) => Math.min(Math.max(Math.round(d * 2) / 2, -90), 90)
const num = (s: string) => { const n = Number(s); return Number.isFinite(n) ? n : 0 }
const km = (m: number) => +((m / 1000).toFixed(1))
/** Потолки ручной расстановки: дальше этих чисел стенд уже не «ракета и цель»,
 *  а недостижимая география — ввод и перетаскивание жёстко ограничены. */
const LIMIT_XY = 300_000
const LIMIT_ALT = 30_000
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

/** Начальный азимут вектора скорости цели для типового аспекта (град.) —
 *  чтобы переключение в ручное положение сохраняло видимую геометрию. */
function presetAspect(sc: Scenario): { tx: number; ty: number; talt: number; thdg: number } {
  const r = Math.max(sc.range_m, 1)
  const rng = sc.range_m
  if (sc.aspect === 'beam') {
    const raw = [0.55, -0.65, 0]
    const n = Math.hypot(...raw)
    return { tx: (rng * raw[0]) / n, ty: (rng * raw[1]) / n, talt: 4000, thdg: 90 }
  }
  if (sc.aspect === 'tail-chase') {
    const raw = [0.45, (0.4 * sc.off_axis_m) / r, 40 / r]
    const n = Math.hypot(...raw)
    return { tx: (rng * raw[0]) / n, ty: (rng * raw[1]) / n, talt: 4000 + (rng * raw[2]) / n, thdg: 0 }
  }
  const raw = [1, sc.off_axis_m / r, 80 / r]
  const n = Math.hypot(...raw)
  return { tx: (rng * raw[0]) / n, ty: (rng * raw[1]) / n, talt: 4000 + (rng * raw[2]) / n, thdg: 180 }
}

/** Патч сценария при переключении в ручной режим: взять геометрию текущего
 *  аспекта и её же числами (ракета — по курсу 0°, как во всех пресетах). */
export function freePatchFromPreset(sc: Scenario): Partial<Scenario> {
  const g = sc.aspect === 'free' ? { tx: sc.free_tx, ty: sc.free_ty, talt: sc.free_talt, thdg: sc.free_thdg } : presetAspect(sc)
  return {
    free_tx: Math.round(g.tx),
    free_ty: Math.round(g.ty),
    free_talt: Math.round(g.talt),
    free_mhdg: 0,
    free_mclimb: 0,
    free_thdg: Math.round(g.thdg),
    free_tclimb: 0,
  }
}

// ── пресеты расстановки (localStorage, живут независимо от сценария) ──
const PRESET_LS = 'muholet-free-presets'
const PRESET_KEYS = ['free_tx', 'free_ty', 'free_talt', 'free_mhdg', 'free_mclimb', 'free_thdg', 'free_tclimb', 'alt_m', 'v_m', 'v_t'] as const
type Preset = { name: string; vals: Partial<Record<(typeof PRESET_KEYS)[number], number>> }
function loadPresets(): Preset[] {
  try {
    const a = JSON.parse(localStorage.getItem(PRESET_LS) || '[]')
    return Array.isArray(a) ? a.filter((p: Preset) => p && typeof p.name === 'string' && p.vals) : []
  } catch {
    return []
  }
}

// геометрия видов: общий масштаб X; план 300×200 (ось X на y=100), профиль 300×120 (земля на y=104)
const OX = 26, OY = 100, RX = 288, SPAN = 262, GR = 104, HTOP = 88, TIP_HIT = 12, TIP_LEN = 30

export function FreeGeometryEditor({ sc, set }: { sc: Scenario; set: SetFn }) {
  const fitR = () => quant(1.3 * Math.hypot(sc.free_tx, sc.free_ty))
  const fitH = () => quant(1.25 * Math.max(sc.free_talt, sc.alt_m, 500))
  const [rmax, setRmax] = useState(fitR)
  const [hmax, setHmax] = useState(fitH)
  const [presets, setPresets] = useState<Preset[]>(loadPresets)
  const [pname, setPname] = useState('')
  // серверная библиотека сценариев (data/scenarios, переживает перезапуск стенда)
  const [remote, setRemote] = useState<{ name: string }[]>([])
  const [remoteNote, setRemoteNote] = useState('')
  const syncRemote = () => {
    fetch('/api/scenarios')
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((d) => setRemote((d.scenarios || []).map((s: { name: string }) => ({ name: s.name }))))
      .catch(() => setRemote([]))
  }
  useEffect(syncRemote, [])
  const saveRemote = async () => {
    const name = (pname.trim() || `расстановка ${new Date().toISOString().slice(0, 16).replace('T', ' ')}`).slice(0, 64)
    try {
      const r = await fetch('/api/scenarios', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, scenario: sc }),
      })
      if (!r.ok) throw new Error(String(r.status))
      setRemoteNote('')
      setPname('')
      syncRemote()
    } catch {
      setRemoteNote('архив доступен только на стенде')
    }
  }
  const applyRemote = async (name: string) => {
    try {
      const r = await fetch(`/api/scenarios/${encodeURIComponent(name)}`)
      if (!r.ok) throw new Error(String(r.status))
      const d = await r.json()
      for (const [k, v] of Object.entries(d.scenario || {})) if (k in sc && k !== 'aspect') set(k as keyof Scenario, v as never)
      if (d.scenario?.aspect) set('aspect', d.scenario.aspect)
      setRemoteNote('')
    } catch {
      setRemoteNote('не удалось загрузить сценарий со стенда')
    }
  }
  const delRemote = async (name: string) => {
    try {
      await fetch(`/api/scenarios/${encodeURIComponent(name)}`, { method: 'DELETE' })
      syncRemote()
    } catch {
      /* стенд офлайн — список и так почистим при следующем открытии */
    }
  }
  const planRef = useRef<SVGSVGElement>(null)
  const profRef = useRef<SVGSVGElement>(null)
  const pdrag = useRef<null | 'target' | 'mhdg' | 'thdg'>(null)
  const vdrag = useRef<null | 'target' | 'mclimb' | 'tclimb'>(null)

  // масштаб растёт сам, когда цель выходит за поле; уменьшается только кнопкой «вписать»
  if (fitR() > rmax) setRmax(fitR())
  if (fitH() > hmax) setHmax(fitH())

  const S = SPAN / rmax
  const SH = HTOP / hmax
  const px = (x: number) => OX + x * S
  const py = (y: number) => OY - y * S
  const hx = (x: number) => OX + x * S
  const hy = (h: number) => GR - h * SH

  // кончики стрелок плана: курс в плане — истинный угол на картинке
  const planTip = (x: number, y: number, hdg: number): [number, number] => [px(x) + TIP_LEN * Math.cos(hdg * DEG), py(y) - TIP_LEN * Math.sin(hdg * DEG)]
  const mTipP = planTip(0, 0, sc.free_mhdg)
  const tTipP = planTip(sc.free_tx, sc.free_ty, sc.free_thdg)
  // стрелка профиля — подъём γ: фиксированные 30 px, направление из мировых
  // смещений [340·cosγ, 340·sinγ] (масштабы осей дают честный визуальный угол);
  // азимут в вид сбоку не проецируем — иначе ψ=90° вырождал бы стрелку в точку на теле цели
  const profTip = (x: number, h: number, climb: number): [number, number] => {
    const dpx = 340 * Math.cos(climb * DEG) * S
    const dpz = 340 * Math.sin(climb * DEG) * SH
    const n = Math.hypot(dpx, dpz) || 1
    return [hx(x) + (30 * dpx) / n, hy(h) - (30 * dpz) / n]
  }
  const mTipV = profTip(0, sc.alt_m, sc.free_mclimb)
  const tTipV = profTip(sc.free_tx, sc.free_talt, sc.free_tclimb)

  const toLocal = (svg: SVGSVGElement, H: number, cx: number, cy: number): [number, number] => {
    const r = svg.getBoundingClientRect()
    return [((cx - r.left) / r.width) * 300, ((cy - r.top) / r.height) * H]
  }
  const planApply = (cx: number, cy: number) => {
    if (!pdrag.current || !planRef.current) return
    const [sx, sy] = toLocal(planRef.current, 200, cx, cy)
    const wx = (sx - OX) / S
    const wy = (OY - sy) / S
    if (pdrag.current === 'target') { set('free_tx', clamp(snap(wx), -LIMIT_XY, LIMIT_XY)); set('free_ty', clamp(snap(wy), -LIMIT_XY, LIMIT_XY)) }
    else if (pdrag.current === 'thdg') set('free_thdg', normHdg(Math.atan2(wy - sc.free_ty, wx - sc.free_tx) / DEG))
    else set('free_mhdg', normHdg(Math.atan2(wy, wx) / DEG))
  }
  const profApply = (cx: number, cy: number) => {
    if (!vdrag.current || !profRef.current) return
    const [sx, sy] = toLocal(profRef.current, 120, cx, cy)
    const wx = (sx - OX) / S
    const wh = (GR - sy) / SH
    if (vdrag.current === 'target') { set('free_tx', clamp(snap(wx), 0, LIMIT_XY)); set('free_talt', clamp(snap(wh), 0, LIMIT_ALT)) }
    else if (vdrag.current === 'tclimb') set('free_tclimb', clipClimb(Math.atan2(wh - sc.free_talt, Math.max(Math.abs(wx - sc.free_tx), 50)) / DEG))
    else set('free_mclimb', clipClimb(Math.atan2(wh - sc.alt_m, Math.max(Math.abs(wx), 50)) / DEG))
  }
  const down = (drag: { current: string | null }, mode: (sx: number, sy: number) => string, apply: (cx: number, cy: number) => void) =>
    (e: React.PointerEvent<SVGSVGElement>) => {
      const [sx, sy] = toLocal(e.currentTarget, e.currentTarget === planRef.current ? 200 : 120, e.clientX, e.clientY)
      drag.current = mode(sx, sy)
      e.currentTarget.setPointerCapture(e.pointerId)
      apply(e.clientX, e.clientY)
    }
  const move = (apply: (cx: number, cy: number) => void) => (e: React.PointerEvent<SVGSVGElement>) => apply(e.clientX, e.clientY)
  const up = (drag: { current: string | null }) => () => { drag.current = null }

  const savePreset = () => {
    const name = pname.trim() || `расстановка ${presets.length + 1}`
    const vals = Object.fromEntries(PRESET_KEYS.map((k) => [k, sc[k]])) as Preset['vals']
    const next = [{ name, vals }, ...presets.filter((p) => p.name !== name)].slice(0, 12)
    setPresets(next)
    setPname('')
    try { localStorage.setItem(PRESET_LS, JSON.stringify(next)) } catch { /* приватный режим — пресеты живут до перезагрузки */ }
  }
  const delPreset = (name: string) => {
    const next = presets.filter((p) => p.name !== name)
    setPresets(next)
    try { localStorage.setItem(PRESET_LS, JSON.stringify(next)) } catch { /* см. savePreset */ }
  }
  const applyPreset = (p: Preset) => {
    for (const [k, v] of Object.entries(p.vals)) if (v !== undefined) set(k as keyof Scenario, v as never)
  }

  const range = Math.hypot(sc.free_tx, sc.free_ty, sc.free_talt - sc.alt_m)
  // Честная геометрия: окно счёта и поле зрения ГСН. Скорость сближения не бывает
  // больше суммы скоростей — если и при ней не успеем, перехват вне расчёта.
  const tMin = range / Math.max(sc.v_m + sc.v_t, 1)
  const mg = Math.cos(sc.free_mclimb * DEG)
  const mV: [number, number, number] = [mg * Math.cos(sc.free_mhdg * DEG), mg * Math.sin(sc.free_mhdg * DEG), Math.sin(sc.free_mclimb * DEG)]
  const rVec: [number, number, number] = [sc.free_tx, sc.free_ty, sc.free_talt - sc.alt_m]
  const rN = Math.hypot(...rVec) || 1
  const alpha = Math.acos(clamp((rVec[0] * mV[0] + rVec[1] * mV[1] + rVec[2] * mV[2]) / rN, -1, 1)) / DEG
  const halfFov = (sc.mode === 'bio' || sc.mode === 'both' ? sc.bio_fov_deg : sc.fov_deg) / 2
  return (
    <div className="free-ed">
      <div className="free-views">
        <svg
          ref={planRef}
          viewBox="0 0 300 200"
          data-tip="План (вид сверху): X — вперёд по трассе пуска, Y — бок (плюс — влево). Тяните жёлтую точку цели; кончик жёлтой стрелки — курс цели ψ, кончик красной — курс ракеты. Дуги — кольца дальности."
          onPointerDown={down(pdrag, (sx, sy) =>
            Math.hypot(sx - tTipP[0], sy - tTipP[1]) < TIP_HIT ? 'thdg' :
            Math.hypot(sx - mTipP[0], sy - mTipP[1]) < TIP_HIT ? 'mhdg' : 'target', planApply)}
          onPointerMove={move(planApply)}
          onPointerUp={up(pdrag)}
        >
          <line x1={OX} y1={OY} x2={RX} y2={OY} stroke="var(--border-default)" strokeWidth={1} />
          {[1, 2, 3, 4].map((i) => (
            <g key={i}>
              <circle cx={OX} cy={OY} r={(SPAN * i) / 4} fill="none" stroke="var(--border-default)" strokeWidth={0.6} strokeDasharray="2 3" />
              <text x={OX + (SPAN * i) / 4} y={OY - 4} fill="var(--text-muted)" fontSize={8} textAnchor={i === 4 ? 'end' : 'middle'}>{km((rmax * i) / 4)}</text>
            </g>
          ))}
          <text x={RX} y={OY + 12} fill="var(--text-muted)" fontSize={9} textAnchor="end">вперёд X →</text>
          <line x1={px(0)} y1={py(0)} x2={mTipP[0]} y2={mTipP[1]} stroke="#e0584f" strokeWidth={2} />
          <circle cx={mTipP[0]} cy={mTipP[1]} r={3.5} fill="#e0584f" style={{ cursor: 'grab' }} />
          <text x={px(0) + 7} y={py(0) + 3} fill="var(--text-secondary)" fontSize={9}>пуск</text>
          <text x={mTipP[0] + 5} y={mTipP[1] - 5} fill="var(--text-muted)" fontSize={8.5}>ψ={sc.free_mhdg}°</text>
          <line x1={px(sc.free_tx)} y1={py(sc.free_ty)} x2={tTipP[0]} y2={tTipP[1]} stroke="var(--accent-secondary)" strokeWidth={2} />
          <circle cx={tTipP[0]} cy={tTipP[1]} r={3.5} fill="var(--accent-secondary)" style={{ cursor: 'grab' }} />
          <circle cx={px(sc.free_tx)} cy={py(sc.free_ty)} r={5} fill="var(--accent-secondary)" style={{ cursor: 'grab' }} />
          <text x={px(sc.free_tx) + 8} y={py(sc.free_ty) + 3} fill="var(--text-secondary)" fontSize={9}>цель</text>
          <text x={tTipP[0] + 5} y={tTipP[1] - 5} fill="var(--text-muted)" fontSize={8.5}>ψ={sc.free_thdg}°</text>
          <text x={10} y={194} fill="var(--text-muted)" fontSize={9}>наклонная дальность {(range / 1000).toFixed(2)} км</text>
        </svg>
        <svg
          ref={profRef}
          viewBox="0 0 300 120"
          data-tip="Профиль (вид сбоку): та же горизонталь X, вертикаль — высота. Тяните жёлтую точку цели — вперёд/вверх; кончик стрелки задаёт подъём γ (+90° — вертикально вверх, −90° — отвесно вниз). Азимут в вид сбоку не показан — только курс на плане."
          onPointerDown={down(vdrag, (sx, sy) =>
            Math.hypot(sx - tTipV[0], sy - tTipV[1]) < TIP_HIT ? 'tclimb' :
            Math.hypot(sx - mTipV[0], sy - mTipV[1]) < TIP_HIT ? 'mclimb' : 'target', profApply)}
          onPointerMove={move(profApply)}
          onPointerUp={up(vdrag)}
        >
          <line x1={OX} y1={GR} x2={RX} y2={GR} stroke="var(--border-default)" strokeWidth={1.2} />
          {[1, 2, 3].map((i) => (
            <g key={i}>
              <line x1={OX} y1={hy((hmax * i) / 4)} x2={RX} y2={hy((hmax * i) / 4)} stroke="var(--border-default)" strokeWidth={0.6} strokeDasharray="2 3" />
              <text x={RX} y={hy((hmax * i) / 4) - 2} fill="var(--text-muted)" fontSize={8} textAnchor="end">{km((hmax * i) / 4)} км</text>
            </g>
          ))}
          <text x={OX + 2} y={115} fill="var(--text-muted)" fontSize={8.5}>Земля</text>
          <text x={RX} y={GR + 12} fill="var(--text-muted)" fontSize={9} textAnchor="end">высота ↑</text>
          <line x1={hx(0)} y1={hy(sc.alt_m)} x2={mTipV[0]} y2={mTipV[1]} stroke="#e0584f" strokeWidth={2} />
          <circle cx={mTipV[0]} cy={mTipV[1]} r={3.5} fill="#e0584f" style={{ cursor: 'grab' }} />
          <circle cx={hx(0)} cy={hy(sc.alt_m)} r={4} fill="#e0584f" />
          <text x={hx(0) + 7} y={hy(sc.alt_m) + 3} fill="var(--text-secondary)" fontSize={9}>пуск {km(sc.alt_m)} км</text>
          <text x={mTipV[0] + 5} y={mTipV[1] - 5} fill="var(--text-muted)" fontSize={8.5}>γ={sc.free_mclimb}°</text>
          <line x1={hx(sc.free_tx)} y1={hy(sc.free_talt)} x2={tTipV[0]} y2={tTipV[1]} stroke="var(--accent-secondary)" strokeWidth={2} />
          <circle cx={tTipV[0]} cy={tTipV[1]} r={3.5} fill="var(--accent-secondary)" style={{ cursor: 'grab' }} />
          <circle cx={hx(sc.free_tx)} cy={hy(sc.free_talt)} r={5} fill="var(--accent-secondary)" style={{ cursor: 'grab' }} />
          <text x={hx(sc.free_tx) + 8} y={hy(sc.free_talt) - 6} fill="var(--text-secondary)" fontSize={9}>цель {km(sc.free_talt)} км</text>
          <text x={tTipV[0] + 5} y={tTipV[1] - 5} fill="var(--text-muted)" fontSize={8.5}>γ={sc.free_tclimb}°</text>
        </svg>
      </div>

      {(tMin > sc.t_max || alpha > halfFov) && (
        <p className="lab-hint" style={{ color: 'var(--accent-secondary)', margin: '2px 0 0' }}>
          {tMin > sc.t_max && (
            <>Даже на встречных курсах сближение займёт ≥ {(tMin / 60).toFixed(1)} мин, а окно счёта — {sc.t_max} с: перехват не успеется. Уменьшите дистанцию или увеличьте «Время счёта» в «Дополнительно». </>
          )}
          {alpha > halfFov && <>На старте линия визирования вне поля зрения ГСН ({alpha.toFixed(1)}° &gt; {halfFov.toFixed(1)}°) — захвата не будет; цель дальше/выше — наведите трассу пуска на неё.</>}
        </p>
      )}

      <div className="free-bar">
        <button data-tip="Вписать текущую расстановку в масштаб обоих видов (масштаб сам только растёт)." onClick={() => { setRmax(fitR()); setHmax(fitH()) }}>⟲ вписать</button>
        <input value={pname} placeholder="имя расстановки" onChange={(e) => setPname(e.target.value)} data-tip="Имя для сохраняемой расстановки (пусто — автоматическое)." />
        <button data-tip="Сохранить текущие позиции, курсы, скорости и высоту пуска как пресет (до 12, в этом браузере)." onClick={savePreset}>сохранить</button>
        <button data-tip="Положить ТЕКУЩИЙ СЦЕНАРИЙ ЦЕЛИКОМ (все поля, не только расстановку) в архив на стенде — переживёт перезапуск и виден с любого браузера. Имя — из поля слева или автоматом." onClick={saveRemote}>в архив</button>
      </div>
      {remoteNote && <p className="lab-hint" style={{ margin: '2px 0 0' }}>{remoteNote}</p>}
      {presets.length > 0 && (
        <div className="free-bar" data-tip="Клик по имени — применить расстановку, × — удалить.">
          {presets.map((p) => (
            <span className="chip" key={p.name}>
              <button onClick={() => applyPreset(p)}>{p.name}</button>
              <button aria-label={`удалить ${p.name}`} onClick={() => delPreset(p.name)}>×</button>
            </span>
          ))}
        </div>
      )}
      {remote.length > 0 && (
        <div className="free-bar" data-tip="Архив на стенде (сохраняется в data/scenarios). Клик — загрузить весь сценарий, × — удалить со стенда.">
          {remote.map((s) => (
            <span className="chip" key={s.name}>
              <button onClick={() => applyRemote(s.name)}>☁ {s.name}</button>
              <button aria-label={`удалить ${s.name} из архива`} onClick={() => delRemote(s.name)}>×</button>
            </span>
          ))}
        </div>
      )}

      <div className="fields">
        <label data-tip="Продольная координата цели от точки пуска, вдоль +X. Ставится перетаскиванием в обоих видах.">
          Цель вперёд, км
          <input type="number" step={0.5} value={sc.free_tx / 1000} onChange={(e) => set('free_tx', clamp(num(e.target.value) * 1000, -LIMIT_XY, LIMIT_XY))} />
        </label>
        <label data-tip="Боковое смещение цели: плюс — влево от трассы пуска.">
          Цель вбок, км
          <input type="number" step={0.5} value={sc.free_ty / 1000} onChange={(e) => set('free_ty', clamp(num(e.target.value) * 1000, -LIMIT_XY, LIMIT_XY))} />
        </label>
        <label data-tip="Абсолютная высота цели, м (высота пуска — в «Ракета и цель»); видна в профиле.">
          Высота цели, м
          <input type="number" step={100} min={0} max={LIMIT_ALT} value={sc.free_talt} onChange={(e) => set('free_talt', clamp(num(e.target.value), 0, LIMIT_ALT))} />
        </label>
        <label data-tip="Азимут начальной скорости ракеты: 0° — прямо на ось X, плюс — против часовой. Тяга за кончик красной стрелки на плане.">
          Курс ракеты, °
          <input type="number" step={5} value={sc.free_mhdg} onChange={(e) => set('free_mhdg', normHdg(num(e.target.value)))} />
        </label>
        <label data-tip="Начальный подъём скорости ракеты над горизонтом (минус — пикирование). Тяга за кончик стрелки в профиле.">
          Подъём ракеты, °
          <input type="number" step={5} min={-90} max={90} value={sc.free_mclimb} onChange={(e) => set('free_mclimb', clipClimb(num(e.target.value)))} />
        </label>
        <label data-tip="Азимут скорости цели: 180° — лобовой курс к пуску, 0° — убегание. Тяга за кончик жёлтой стрелки на плане.">
          Курс цели, °
          <input type="number" step={5} value={sc.free_thdg} onChange={(e) => set('free_thdg', normHdg(num(e.target.value)))} />
        </label>
        <label data-tip="Начальный подъём цели над горизонтом, град. Тяга за кончик стрелки в профиле.">
          Подъём цели, °
          <input type="number" step={5} min={-90} max={90} value={sc.free_tclimb} onChange={(e) => set('free_tclimb', clipClimb(num(e.target.value)))} />
        </label>
      </div>
    </div>
  )
}
