import { useRef } from 'react'
import { CHART, drawChart, exportCsv, movingAvg, palette, sessionBorders, useAutoRedraw } from '../charts'
import { LabScreen } from '../LabScreen'
import type { LabData, TrainPoint } from '../types'

const fmt = (v: number | null | undefined, d = 1) => (v === null || v === undefined || !Number.isFinite(v) ? '—' : v.toFixed(d))
const rr2 = (v: number | null | undefined) => (v === null || v === undefined || !Number.isFinite(v) ? '—' : String(Math.round(v)))

/** Экран «Обучение»: кривые имитационного обучения. */
export function TrainScreen({
  data,
  training,
  trainProgress,
  onTrain,
  onTrainStop,
  smooth,
  showFact,
}: {
  data: LabData
  training: boolean
  trainProgress: { ep: number; total: number } | null
  onTrain: () => void
  onTrainStop: () => void
  smooth: number
  showFact: boolean
}) {
  const c1 = useRef<HTMLCanvasElement>(null)
  const c2 = useRef<HTMLCanvasElement>(null)
  const c3 = useRef<HTMLCanvasElement>(null)
  const c4 = useRef<HTMLCanvasElement>(null)
  const c5 = useRef<HTMLCanvasElement>(null)
  const c6 = useRef<HTMLCanvasElement>(null)

  useAutoRedraw(() => {
    const borders = sessionBorders(data.train)
    const raw = (pick: (p: TrainPoint) => number | null) => data.train.map(pick)
    const trend = (pick: (p: TrainPoint) => number | null, color: string) => [
      ...(showFact
        ? [{ data: raw(pick).map((v) => (v === null ? NaN : v)).filter((v) => Number.isFinite(v)) as number[], color, width: 1, label: 'факт' }]
        : []),
      { data: movingAvg(raw(pick), smooth).map((v) => (v === null ? NaN : v)).filter((v) => Number.isFinite(v)) as number[], color, width: 2.4, label: `среднее ${smooth}` },
    ]
    if (c1.current) drawChart(c1.current, trend((p) => p.miss, CHART.accent()), 'Промах по эпизодам, м — тонкая линия факт, яркая — среднее за окно', borders)
    if (c2.current) drawChart(c2.current, trend((p) => (p.refDev > 0 ? p.refDev : null), CHART.amber()), 'Отклонение от траектории эталонного ПН, м — мера похожести, ПН не «оптимален» в общем случае', borders)
    if (c3.current) {
      const tg = data.train.map((p) => p.tGuide)
      const maxT = Math.max(...tg.filter((v): v is number => v !== null), 0)
      const hits: number[] = data.train.map((p) => (p.hit ? 1 : 0))
      const rate = hits.map((_, i) => {
        const a = hits.slice(Math.max(0, i - 6), i + 1)
        return a.reduce((s, v) => s + v, 0) / a.length
      })
      drawChart(
        c3.current,
        [...trend((p) => p.tGuide, CHART.accent()), { data: rate.map((r) => r * (maxT || 1)), color: CHART.amber(), dash: [5, 3], width: 1.6, label: 'доля перехватов' }],
        'Время наведения, с (фосфор) и доля перехватов в окне 7, приведённая к шкале (янтарь)',
        borders,
      )
    }
    if (c4.current) {
      const abs = data.train.map((p) => p.w.map((x) => Math.abs(x)))
      const avgOf = (from: number) => abs.map((row) => row.slice(from, from + 8).reduce((s, v) => s + v, 0) / 8)
      const allFlat = abs.map((row) => row.flat())
      drawChart(
        c4.current,
        [
          { data: allFlat.map((row) => Math.min(...row)), color: CHART.minmax(), dash: [2, 2], label: 'min' },
          { data: allFlat.map((row) => Math.max(...row)), color: CHART.minmax(), dash: [2, 2], label: 'max' },
          { data: avgOf(0), color: CHART.accent(), width: 2.2, label: 'тангаж' },
          { data: avgOf(8), color: CHART.amber(), width: 2.2, label: 'рыскание' },
        ],
        'Сила выхода: средняя |вес| канала (фосфор — тангаж, янтарь — рыскание), серый пунктир — min/max',
        borders,
      )
    }
    if (c5.current)
      drawChart(
        c5.current,
        [
          { data: data.train.map((p) => p.nAvg ?? NaN).filter((v) => Number.isFinite(v)), color: CHART.accent(), width: 2.2, label: 'средняя g' },
          { data: data.train.map((p) => p.nPeak ?? NaN).filter((v) => Number.isFinite(v)), color: CHART.amber(), dash: [5, 3], label: 'пик g' },
        ],
        'Перегрузка по эпизодам, g — средняя (фосфор) и пиковая (янтарь): «цена» манёвра',
        borders,
      )
    if (c6.current)
      drawChart(
        c6.current,
        [
          { data: data.train.map((p) => (p.lockFrac ?? NaN) * 100).filter((v) => Number.isFinite(v)), color: CHART.accent(), width: 2.2, label: 'захват, %' },
          { data: data.train.map((p) => (p.nrms ?? NaN) * 100).filter((v) => Number.isFinite(v)), color: CHART.amber(), dash: [5, 3], label: 'норм. СКО ×100' },
        ],
        'Захват цели, % (фосфор) и нормированное СКО ×100 (янтарь) по эпизодам',
        borders,
      )
  })

  // сводка последней сессии
  const sessOf = (q: { session?: number }) => q.session ?? 0
  const maxSess = Math.max(...data.train.map(sessOf), 0)
  const rows = data.train.filter((p) => sessOf(p) === maxSess)
  const n = rows.length
  const misses = rows.map((r) => r.miss)
  const avgTail = (arr: number[]) => (arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0)
  const tail = avgTail(misses.slice(-5))
  const prev = avgTail(misses.slice(-10, -5))
  const trendDir = n < 10 ? 'мало данных' : tail < prev - 2 ? '↓ улучшается' : tail > prev + 2 ? '↑ ухудшается' : '→ стабильно'
  const best = n ? Math.min(...misses) : null
  const hits = rows.filter((r) => r.hit).length

  return (
    <LabScreen
      title="Обучение"
      about="Муха учится подражать учителю-ПН, летя сама — со своими шумами и срывами захвата. Каждая сессия отделена пунктиром; тонкая линия — факт, яркая — скользящее среднее."
      primary={
        training ? (
          <button type="button" className="primary" onClick={onTrainStop}>
            Остановить обучение
          </button>
        ) : (
          <button type="button" className="primary" onClick={onTrain}>
            Запустить обучение
          </button>
        )
      }
      exports={[
        {
          label: 'Обучение: эпизоды и веса (CSV)',
          disabled: !data.train.length,
          onClick: () =>
            exportCsv(
              'muholet-train.csv',
              'ep,miss_m,hit,t_guide_s,ref_dev_m,' + Array.from({ length: 16 }, (_, i) => `w${i}`).join(','),
              data.train.map((p) => [p.ep, p.miss.toFixed(1), p.hit ? 1 : 0, p.tGuide === null ? '' : p.tGuide.toFixed(2), p.refDev.toFixed(1), ...p.w.map((x) => x.toFixed(4))]),
            ),
        },
        {
          label: 'MATLAB: только кривые (CSV)',
          disabled: !data.train.length,
          onClick: () =>
            exportCsv(
              'muholet-train-matlab.csv',
              'ep,miss_m,hit,t_guide_s,ref_dev_m',
              data.train.map((p) => [p.ep, p.miss.toFixed(1), p.hit ? 1 : 0, p.tGuide === null ? '' : p.tGuide.toFixed(2), p.refDev.toFixed(1)]),
            ),
        },
      ]}
      status={
        <>
          <span>
            сессия <b>{Math.max(maxSess, 1)}</b>
          </span>
          <span>
            эпизодов <b>{n}</b>
          </span>
          <span>
            промах <b>{rr2(rows[0]?.miss)} → {rr2(rows[n - 1]?.miss)} м</b>
          </span>
          <span>
            лучший <b>{rr2(best)} м</b>
          </span>
          <span>
            перехваты <b>{n ? Math.round((hits / n) * 100) : 0}%</b>
          </span>
          <span>
            тренд <b>{trendDir}</b>
          </span>
        </>
      }
      note={
        data.summary
          ? `Сводка канонического трио: промах ${fmt(data.summary.missBefore, 0)} → ${fmt(data.summary.missAfter, 0)} м · отклонение от ПН ${fmt(data.summary.refDevBefore, 0)} → ${fmt(data.summary.refDevAfter, 0)} м · время наведения ${fmt(data.summary.tGuideAfter, 2)} с · перехваты ${Math.round((data.summary.hitRateAfter ?? 0) * 100)}%.`
          : 'Настройки обучения (режим, шаг, число эпизодов) — в пространстве «Мозг», кнопка «Параметры обучения».'
      }
    >
      {trainProgress && (
        <div className="lab-progress">
          <div className="lab-progress__bar" style={{ width: `${Math.round((trainProgress.ep / Math.max(trainProgress.total, 1)) * 100)}%` }} />
          <span>
            обучение · эпизод {trainProgress.ep}/{trainProgress.total}
          </span>
        </div>
      )}
      <div className="lab-grid">
        <div className="chart-card"><canvas ref={c1} /></div>
        <div className="chart-card"><canvas ref={c2} /></div>
        <div className="chart-card"><canvas ref={c3} /></div>
        <div className="chart-card"><canvas ref={c4} /></div>
        <div className="chart-card"><canvas ref={c5} /></div>
        <div className="chart-card"><canvas ref={c6} /></div>
      </div>
    </LabScreen>
  )
}

/** Экран «Рой» лаборатории: история поколений, чемпион, архив. */
export function SwarmLabScreen({
  data,
  expList,
  onRefreshExperiments,
  onLoadExperimentServer,
  onDeleteExperimentServer,
  onSaveExperimentServer,
  onSwarmExport,
  onExportSwarmCsv,
  championToBrain,
}: {
  data: LabData
  expList: { name: string; saved_at: string; size: number }[]
  onRefreshExperiments: () => void
  onLoadExperimentServer: (name: string) => void
  onDeleteExperimentServer: (name: string) => void
  onSaveExperimentServer: () => void
  onSwarmExport: () => void
  onExportSwarmCsv: () => void
  championToBrain: () => void
}) {
  const c1 = useRef<HTMLCanvasElement>(null)
  const c2 = useRef<HTMLCanvasElement>(null)
  const c3 = useRef<HTMLCanvasElement>(null)
  const c4 = useRef<HTMLCanvasElement>(null)

  useAutoRedraw(() => {
    if (c1.current)
      drawChart(
        c1.current,
        [
          { data: data.gen.map((p) => p.best), color: CHART.accent(), label: 'лучший' },
          { data: data.gen.map((p) => p.avg), color: CHART.amber(), dash: [5, 3], label: 'средний' },
          { data: data.gen.map((p) => p.worst ?? NaN).filter((v) => Number.isFinite(v)), color: CHART.red(), dash: [2, 3], label: 'худший' },
        ],
        'Промах по поколениям, м (фосфор — лучший, янтарь — средний, красный — худший)',
      )
    if (c2.current)
      drawChart(c2.current, [{ data: data.gen.map((p) => (p.hitRate ?? NaN) * 100).filter((v) => Number.isFinite(v)), color: CHART.accent(), label: '% перехватов' }], 'Доля перехватов в поколении, %')
    if (c3.current)
      drawChart(c3.current, [{ data: data.gen.map((p) => (p.diversity ?? NaN) * 100).filter((v) => Number.isFinite(v)), color: CHART.amber(), label: 'разнообразие' }], 'Разнообразие весов популяции (средний разброс), усл. ед.')
    if (c4.current)
      drawChart(c4.current, [{ data: data.gen.map((p) => p.champ ?? NaN).filter((v) => Number.isFinite(v)), color: CHART.amber(), label: 'чемпион' }], 'Чемпион на эталонном трио, м (обновляется на валидациях)')
  })

  const gens = data.gen
  const bestAll = gens.length ? Math.min(...gens.map((g) => g.best)) : null
  const bestGen = bestAll !== null ? gens.findIndex((g) => g.best === bestAll) + 1 : 0
  const champs = gens.map((g) => g.champ).filter((v) => v !== undefined && Number.isFinite(v as number))
  const champ = champs.length ? champs[champs.length - 1] : null
  const lastHit = gens.length ? (gens[gens.length - 1].hitRate ?? 0) * 100 : 0

  return (
    <LabScreen
      title="Рой"
      about="История эволюции: каждый запуск роя в пространстве «Рой» добавляет поколения сюда. Чемпион — лучшая муха на фиксированном эталонном трио."
      primary={
        <button type="button" className="primary" disabled={!data.gen.length} onClick={championToBrain}>
          Чемпион → в схему
        </button>
      }
      actions={
        <button type="button" disabled={!data.gen.length} onClick={onSaveExperimentServer}>
          В архив стенда
        </button>
      }
      exports={[
        { label: 'Поколения (CSV)', disabled: !data.gen.length, onClick: onExportSwarmCsv },
        { label: 'Полный снимок эксперимента (JSON)', disabled: !data.gen.length, onClick: onSwarmExport },
      ]}
      status={
        <>
          <span>
            поколений <b>{gens.length}</b>
          </span>
          <span>
            лучший за всё время <b>{rr2(bestAll)} м</b> {bestGen ? `(поколение ${bestGen})` : ''}
          </span>
          <span>
            чемпион (эталон) <b>{rr2(champ as number | null)} м</b>
          </span>
          <span>
            перехваты последнего <b>{Math.round(lastHit)}%</b>
          </span>
        </>
      }
    >
      <div className="lab-grid">
        <div className="chart-card"><canvas ref={c1} /></div>
        <div className="chart-card"><canvas ref={c2} /></div>
        <div className="chart-card"><canvas ref={c3} /></div>
        <div className="chart-card"><canvas ref={c4} /></div>
      </div>
      <div className="lab-archive">
        <span>Архив стенда ({expList.length}):</span>
        <select
          value=""
          aria-label="Загрузить эксперимент из архива"
          onChange={(e) => {
            if (e.target.value) onLoadExperimentServer(e.target.value)
            e.target.value = ''
          }}
        >
          <option value="">— загрузить эксперимент —</option>
          {expList.map((e) => (
            <option key={e.name} value={e.name}>
              {e.name} · {e.saved_at.slice(5, 16)}
            </option>
          ))}
        </select>
        <select
          value=""
          disabled={!expList.length}
          aria-label="Удалить эксперимент из архива"
          title={expList.length ? 'Удалить выбранный снимок из data/experiments' : 'Архив пуст'}
          onChange={(e) => {
            if (e.target.value) onDeleteExperimentServer(e.target.value)
            e.target.value = ''
          }}
        >
          <option value="">— удалить из архива —</option>
          {expList.map((e) => (
            <option key={e.name} value={e.name}>
              ✕ {e.name}
            </option>
          ))}
        </select>
        <button type="button" onClick={onRefreshExperiments}>
          Обновить список
        </button>
      </div>
    </LabScreen>
  )
}
