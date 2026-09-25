import { useRef } from 'react'
import { EngagementView, type Playback } from '../lazyViews'
import { Menu } from '../ui'
import { drawChart, palette, useAutoRedraw } from '../lab/charts'
import { fmt } from './labels'
import type { CamMode, Frame, GenPoint, Scenario } from '../types'

/** Рабочее пространство «Рой»: сцена эволюции + параметры популяции + прогресс + чемпион. */
export function SwarmWorkspace({
  frame,
  sc,
  playback,
  day,
  camMode,
  onCamMode,
  geometryOn,
  onToggleGeometry,
  sceneBadge,
  swarmCurve,
  swarmValid,
  egg,
  royCaption,
  silent,
  busy,
  swarmRunning,
  swarmInfo,
  swarmCfg,
  onSwarmCfg,
  onStartSwarm,
  onStopSwarm,
  onRookieVsVeteran,
  onFlyVsPN,
  genHistory,
  championToBrain,
  swarmExport,
  onExportSwarmCsv,
  expList,
  onRefreshExperiments,
  onSaveExperimentServer,
  onLoadExperimentServer,
}: {
  frame: Frame | null
  sc: Scenario
  playback: Playback | null
  day: boolean
  camMode: CamMode
  onCamMode: (m: CamMode) => void
  geometryOn: boolean
  onToggleGeometry: () => void
  sceneBadge: string | null
  swarmCurve: number[] | null
  /** индексы валидационных поколений в кривой */
  swarmValid: number[]
  egg: boolean
  royCaption: { text: string } | null
  silent: boolean
  busy: boolean
  swarmRunning: boolean
  swarmInfo: { gen: number; bestMiss: number; avgFit: number; bio: number; pn: number; fits: number[]; geo: string; champion?: number } | null
  swarmCfg: { size: number; eliteK: number; mutation: number }
  onSwarmCfg: (patch: Partial<{ size: number; eliteK: number; mutation: number }>) => void
  onStartSwarm: () => void
  onStopSwarm: () => void
  onRookieVsVeteran: () => void
  onFlyVsPN: () => void
  genHistory: GenPoint[]
  championToBrain: () => void
  swarmExport: () => void
  onExportSwarmCsv: () => void
  expList: { name: string; saved_at: string; size: number }[]
  onRefreshExperiments: () => void
  onSaveExperimentServer: () => void
  onLoadExperimentServer: (name: string) => void
}) {
  const chartRef = useRef<HTMLCanvasElement>(null)
  useAutoRedraw(() => {
    if (!chartRef.current) return
    const P = palette()
    drawChart(
      chartRef.current,
      [
        { data: genHistory.map((p) => p.best), color: P.accent, label: 'лучший' },
        { data: genHistory.map((p) => p.avg), color: P.amber, dash: [5, 3], label: 'средний' },
      ],
      'Прогресс роя: промах по поколениям, м',
    )
  })

  return (
    <div className="twocol-ws">
      <div className="ws-main">
        <EngagementView
          frame={frame}
          scenario={sc}
          playback={playback}
          day={day}
          camMode={camMode}
          geometryOn={geometryOn}
          sceneIdle={false}
          sceneBadge={sceneBadge}
          swarmCurve={swarmCurve}
          swarmValid={swarmValid}
          cockpit={egg}
          silent={silent}
        />
        {royCaption && (
          <div className="shtrum-caption shtrum-caption--wendy">
            <b>♀ Командир роя</b> {royCaption.text}
          </div>
        )}
        <div className="scene-overlay scene-overlay--topleft">
          <span className="chip is-warn">рой · {swarmInfo ? `поколение ${swarmInfo.gen}` : 'эволюция: отбор лучших'}</span>
          {swarmInfo && (
            <span className="chip" data-tip="Геометрия текущего поколения: каждое поколение летит новую цель.">
              {swarmInfo.geo}
            </span>
          )}
        </div>
      </div>

      <div className="ws-side">
        <h2 className="ws-side__title">Рой мух</h2>

        <section className="ws-card">
          <h3>Действия</h3>
          {swarmRunning ? (
            <button type="button" className="primary" data-tip="Остановить эволюцию (текущий кадр останется в статистике)." onClick={onStopSwarm}>
              Остановить рой
            </button>
          ) : (
            <button type="button" className="primary" data-tip="Выпустить рой и начать эволюцию наведения." disabled={busy} onClick={onStartSwarm}>
              Пустить рой
            </button>
          )}
          <button type="button" data-tip="Необученная муха и лучшая из роя летят одну цель одновременно — видно, чему рой научился." disabled={busy || swarmRunning} onClick={onRookieVsVeteran}>
            стажёр против ветерана
          </button>
          <button type="button" data-tip="Обученная муха против эталонного ПН на одной цели — кто точнее." disabled={busy || swarmRunning} onClick={onFlyVsPN}>
            муха против ПН
          </button>
        </section>

        <section className="ws-card">
          <h3>Параметры популяции</h3>
          <div className="fields fields--2" style={{ padding: 0 }}>
            <label data-tip="Сколько мух летит в одном поколении (8–48).">
              Размер роя
              <input
                type="number"
                min={8}
                max={48}
                value={swarmCfg.size}
                disabled={swarmRunning}
                onChange={(e) => onSwarmCfg({ size: Math.max(8, Math.min(48, Number(e.target.value) || 8)) })}
              />
            </label>
            <label data-tip="Лучшие мухи, переходящие в следующее поколение без изменений.">
              Элита
              <input
                type="number"
                min={1}
                max={10}
                value={swarmCfg.eliteK}
                disabled={swarmRunning}
                onChange={(e) => onSwarmCfg({ eliteK: Math.max(1, Math.min(10, Number(e.target.value) || 1)) })}
              />
            </label>
            <label data-tip="Сила мутаций при размножении. Мало — эволюция вялая, много — потомки теряют навыки.">
              Мутация
              <input
                type="number"
                step={0.05}
                value={swarmCfg.mutation}
                disabled={swarmRunning}
                onChange={(e) => onSwarmCfg({ mutation: Math.max(0.02, Number(e.target.value) || 0.25) })}
              />
            </label>
          </div>
          <p className="lab-hint" style={{ margin: 0 }}>каждое поколение — новая цель: рой учится наводить, а не запоминать одну цель</p>
        </section>

        <section className="ws-card">
          <h3>Поколение</h3>
          <div className="stat-grid">
            <span data-tip="Промах лучшей мухи последнего поколения.">
              наименьшее сближение <b>{swarmInfo ? `${fmt(swarmInfo.bestMiss, 0)} м` : '—'}</b>
            </span>
            <span data-tip="Чемпион на фиксированном эталонном трио (обновляется на валидационных поколениях).">
              чемпион (эталон) <b>{swarmInfo?.champion !== undefined ? `${fmt(swarmInfo.champion, 0)} м` : '—'}</b>
            </span>
            <span>
              мух био / ПН <b>{swarmInfo ? `${swarmInfo.bio} / ${swarmInfo.pn}` : '—'}</b>
            </span>
          </div>
        </section>

        <section className="ws-card">
          <h3>Прогресс</h3>
          <canvas ref={chartRef} className="chart-box" />
        </section>

        <section className="ws-card">
          <h3>Чемпион</h3>
          <button type="button" className="go" data-tip="Веса лучшей биомухи — в обучаемую схему (279 клеток)." disabled={!genHistory.length} onClick={championToBrain}>
            Чемпион → в схему
          </button>
          <div className="lab-archive">
            <Menu
              label="Архив и экспорт"
              items={[
                { label: 'Снимок роя в файл (JSON)', disabled: !genHistory.length, onClick: swarmExport },
                { label: 'Снимок в архив стенда', disabled: !genHistory.length, onClick: onSaveExperimentServer },
                { label: 'История поколений (CSV)', disabled: !genHistory.length, onClick: onExportSwarmCsv },
                { label: 'Обновить список архива', onClick: onRefreshExperiments },
              ]}
            />
            <select
              value=""
              aria-label="Загрузить эксперимент из архива"
              onChange={(e) => {
                if (e.target.value) onLoadExperimentServer(e.target.value)
                e.target.value = ''
              }}
            >
              <option value="">— из архива ({expList.length}) —</option>
              {expList.map((e) => (
                <option key={e.name} value={e.name}>
                  {e.name} · {e.saved_at.slice(5, 16)}
                </option>
              ))}
            </select>
          </div>
        </section>
      </div>
    </div>
  )
}
