import { useEffect, useState } from 'react'
import { EngagementView, type Playback } from '../lazyViews'
import { ScenarioInspector } from './ScenarioInspector'
import { FreeGeometryEditor } from './FreeGeometryEditor'
import { TelemetryBar, TelemetryDrawer } from './TelemetryBar'
import { ASPECT_LABEL, LAW_RU, MODE_LABEL, fmt } from './labels'
import type { CamMode, Frame, Scenario } from '../types'

/** Рабочее пространство «Полёт»: сцена + контекстный инспектор + полоса телеметрии. */
export function FlightWorkspace({
  frame,
  sc,
  set,
  onManeuver,
  playback,
  day,
  camMode,
  onCamMode,
  geometryOn,
  onToggleGeometry,
  sceneBadge,
  swarmCurve,
  egg,
  cofly,
  shtrumCaption,
  silent,
  sceneIdle,
  busy,
  done,
  log,
  onClearLog,
}: {
  frame: Frame | null
  sc: Scenario
  set: <K extends keyof Scenario>(key: K, value: Scenario[K]) => void
  onManeuver: (m: Scenario['maneuver']) => void
  playback: Playback | null
  day: boolean
  camMode: CamMode
  onCamMode: (m: CamMode) => void
  geometryOn: boolean
  onToggleGeometry: () => void
  sceneBadge: string | null
  swarmCurve: number[] | null
  egg: boolean
  /** юмор-режим: пол напарника ('m'/'f') или null — режим выключен */
  cofly: 'm' | 'f' | null
  shtrumCaption: { text: string; he: boolean } | null
  silent: boolean
  sceneIdle: boolean
  busy: boolean
  done: string | null
  log: string[]
  onClearLog: () => void
}) {
  // на узких экранах инспектор — выезжающая панель; на широких виден всегда
  const [inspOpen, setInspOpen] = useState(false)
  const [telOpen, setTelOpen] = useState(false)
  // крупный редактор расстановки — плавающей панелью поверх сцены (в колонке инспектора он мелок)
  const [freeOpen, setFreeOpen] = useState(true)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === 'Escape' && inspOpen) setInspOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [inspOpen])

  const statusText = done ?? (busy ? 'Считаю…' : '')
  const statusKind: '' | 'is-ok' | 'is-bad' = done ? (done.startsWith('Перехват') || done.startsWith('Ракета') ? 'is-ok' : 'is-bad') : ''

  return (
    <div className="flight">
      <div className="flight__main">
        <div className="scene-card">
          <EngagementView
            frame={frame}
            scenario={sc}
            playback={playback}
            day={day}
            camMode={camMode}
            geometryOn={geometryOn}
            sceneIdle={sceneIdle}
            sceneBadge={sceneBadge}
            swarmCurve={swarmCurve}
            cockpit={egg}
            cofly={cofly}
            silent={silent}
            onFreeGeom={(p) => {
              set('free_tx', p.free_tx)
              set('free_ty', p.free_ty)
              set('free_talt', p.free_talt)
            }}
          />

          {sc.aspect === 'free' && freeOpen && (
            <div className="free-pop">
              <div className="free-pop__head">
                <b>Расстановка</b>
                <span className="lab-hint">цель тянется и прямо в сцене: X/Y, с Shift — высота</span>
                <button type="button" aria-label="свернуть панель расстановки" onClick={() => setFreeOpen(false)}>—</button>
              </div>
              <FreeGeometryEditor sc={sc} set={set} />
            </div>
          )}

          <div className="scene-overlay scene-overlay--topleft">
            <span className="chip" data-tip="Как летит цель относительно ракеты: навстречу, поперёк курса или в догон.">
              {ASPECT_LABEL[sc.aspect]}
            </span>
            <span className={`chip ${sc.mode === 'bio' ? 'is-warn' : ''}`} data-tip="Закон управления: МПС — эталонная математика, био — мозг дрозофилы, смешанный — био с резервным МПС.">
              {MODE_LABEL[sc.mode]}
            </span>
            <span className="chip" data-tip="Закон наведения режима МПС.">
              {LAW_RU[sc.law] ?? sc.law}
            </span>
            <span className={`chip ${frame?.lock ? 'is-active' : ''}`} data-tip="Цель внутри кадра головки — наведение идёт по ней.">
              {frame?.lock ? 'захват' : 'поиск'}
            </span>
            <span className="chip num" data-tip="Время от пуска.">
              t = {fmt(frame?.t ?? 0, 2)} с
            </span>
          </div>

          <div className="scene-overlay scene-overlay--topright">
            {statusText && <span className={`chip ${statusKind ? (statusKind === 'is-ok' ? 'is-ok' : 'is-bad') : ''}`}>{statusText}</span>}
            <select value={camMode} data-tip="Режим камеры: авто следит за серединой «ракета—цель»; свободная — под вашим управлением; вдогон — вид из-за ракеты. Двойной клик по сцене — вернуть взгляд." onChange={(e) => onCamMode(e.target.value as CamMode)}>
              <option value="auto">камера: авто</option>
              <option value="free">камера: свободная</option>
              <option value="chase">камера: вдогон</option>
            </select>
            <button type="button" className={geometryOn ? 'on' : ''} data-tip="Треугольник перехвата, круг БЧ, вектор команды и ожидаемый промах прямо на сцене." onClick={onToggleGeometry}>
              геометрия
            </button>
            {sc.aspect === 'free' && (
              <button type="button" className={freeOpen ? 'on' : ''} data-tip="Крупный редактор ручной расстановки: план, профиль, курсы и пресеты. В свободной схеме открывается сам, сворачивается и возвращается сюда." onClick={() => setFreeOpen((v) => !v)}>
                расстановка
              </button>
            )}
          </div>

          <button type="button" className="scene-fab scene-fab--params" onClick={() => setInspOpen((o) => !o)}>
            {inspOpen ? 'Скрыть параметры' : 'Параметры пуска'}
          </button>
          <button type="button" className="scene-fab scene-fab--tel" onClick={() => setTelOpen(true)}>
            Телеметрия
          </button>

          {sceneBadge && <div className="scene-badge">{sceneBadge}</div>}
          {shtrumCaption && (
            <div className="shtrum-caption">
              <b>{shtrumCaption.he ? '♂ Штруман' : '♀ Штрумана'}</b> {shtrumCaption.text}
            </div>
          )}
        </div>

        <TelemetryBar frame={frame} scenario={sc} statusText={statusText} statusKind={statusKind} onOpenFull={() => setTelOpen(true)} />
      </div>

      <div className={inspOpen ? 'inspector is-open' : 'inspector'}>
        <ScenarioInspector sc={sc} set={set} onManeuver={onManeuver} busy={busy} frame={frame} day={day} />
      </div>

      {telOpen && <TelemetryDrawer frame={frame} scenario={sc} day={day} log={log} onClearLog={onClearLog} onClose={() => setTelOpen(false)} />}
    </div>
  )
}
