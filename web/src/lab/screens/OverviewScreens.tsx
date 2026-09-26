import { useRef } from 'react'
import { drawChart, drawPlanView, exportCsv, palette, useAutoRedraw } from '../charts'
import { LabScreen } from '../LabScreen'
import type { LabData } from '../types'

const fmt = (v: number | null | undefined, d = 1) => (v === null || v === undefined || !Number.isFinite(v) ? '—' : v.toFixed(d))
const fmtT = (v: number | null | undefined) => (v === null || v === undefined ? '—' : `${v.toFixed(2)} с`)
const rr2 = (v: number | null | undefined) => (v === null || v === undefined || !Number.isFinite(v) ? '—' : String(Math.round(v)))

/** Экран «Прогон»: кривые последнего пуска. */
export function RunScreen({ data, onMatlabExport, onTrajExport }: { data: LabData; onMatlabExport: () => void; onTrajExport: () => void }) {
  const c1 = useRef<HTMLCanvasElement>(null)
  const c2 = useRef<HTMLCanvasElement>(null)
  const c3 = useRef<HTMLCanvasElement>(null)
  const c4 = useRef<HTMLCanvasElement>(null)
  const c7 = useRef<HTMLCanvasElement>(null)
  const lastRun = data.runs[0]

  useAutoRedraw(() => {
    const P = palette()
    if (c1.current)
      drawChart(c1.current, [{ data: data.zem.map((p) => p.zem), color: P.amber, label: 'h_cv' }], 'Прогноз промаха h_cv по кадрам, м (стремится к нулю — наведение работает)')
    if (c2.current)
      drawChart(c2.current, [{ data: data.zem.map((p) => p.dev ?? NaN).filter((v) => Number.isFinite(v)), color: P.accent, label: 'откл.' }], 'Отклонение от эталонной траектории ПН, м')
    if (c3.current) {
      const traj = data.traj
      const series = traj
        ? [
            { data: traj.missile.map((p) => p[1]), color: P.accent, label: 'ракета' },
            { data: traj.ghost.map((p) => p[1]), color: P.amber, dash: [5, 3], label: 'эталон ПН' },
            { data: traj.target.map((p) => p[1]), color: P.red, dash: [2, 2], label: 'цель' },
          ]
        : []
      drawChart(c3.current, series, 'Боковая координата по времени, м (ракета против эталона ПН)')
    }
    if (c4.current)
      drawChart(c4.current, [{ data: data.zem.map((p) => p.rng / 1000), color: P.accent, label: 'дальность' }], 'Дальность «ракета—цель» по времени, км')
    if (c7.current) drawPlanView(c7.current, data.traj ?? null)
  })

  return (
    <LabScreen
      title="Прогон"
      about="Кривые последнего пуска: прогноз промаха h_cv (без дальнейшего управления), отклонение от эталонного ПН, боковой канал, дальность и план-вид траекторий. Данные попадают сюда после каждого пуска на сцене."
      exports={[
        {
          label: 'Прогон: время, дальность, h_cv (CSV)',
          disabled: !data.zem.length,
          onClick: () =>
            exportCsv('muholet-run.csv', 't_s,rng_km,zem_m,ref_dev_m', data.zem.map((p) => [p.t.toFixed(2), (p.rng / 1000).toFixed(3), p.zem.toFixed(1), p.dev === null ? '' : p.dev.toFixed(1)])),
        },
        { label: 'MATLAB: координаты и метрики (CSV)', disabled: !data.zem.length, onClick: onMatlabExport },
        { label: 'Траектории + эталон, полный CSV', disabled: !data.zem.length, onClick: onTrajExport },
      ]}
      status={
        lastRun ? (
          <>
            <span>
              последний прогон: <b>{lastRun.label}</b>
            </span>
            <span>
              промах <b>{fmt(lastRun.miss, 0)} м</b>
            </span>
            <span>
              итог <b>{lastRun.hit ? 'перехват' : 'мимо'}</b>
            </span>
            <span>
              время наведения <b>{fmtT(lastRun.tGuide)}</b>
            </span>
            <span>
              отклонение от эталона <b>{fmt(lastRun.refDev, 0)} м</b>
            </span>
            <span>
              захват <b>{Math.round(lastRun.lockFrac * 100)}%</b>
            </span>
          </>
        ) : (
          <span>Сделайте пуск в пространстве «Полёт» — метрики прогона лягут сюда.</span>
        )
      }
      note="Промах без манёвра — куда придёт ракета, если больше не маневрирует. Работа наведения — стянуть эту кривую в ноль к моменту встречи."
    >
      <div className="lab-grid">
        <div className="chart-card"><canvas ref={c1} /></div>
        <div className="chart-card"><canvas ref={c2} /></div>
        <div className="chart-card"><canvas ref={c3} /></div>
        <div className="chart-card"><canvas ref={c4} /></div>
        <div className="chart-card chart-card--tall" style={{ gridColumn: '1 / -1' }}><canvas ref={c7} /></div>
      </div>
    </LabScreen>
  )
}

/** Сводка по сессиям обучения (группировка по session). */
function TrainSessionRows({ data }: { data: LabData }) {
  const sessions = new Map<number, typeof data.train>()
  data.train.forEach((p) => {
    const s = p.session ?? 0
    if (!sessions.has(s)) sessions.set(s, [])
    sessions.get(s)!.push(p)
  })
  const rows = [...sessions.entries()]
    .sort((a, b) => b[0] - a[0])
    .map(([s, pts]) => {
      const misses = pts.map((p) => p.miss)
      return {
        session: s,
        n: pts.length,
        first: misses[0],
        last: misses[misses.length - 1],
        best: Math.min(...misses),
        hits: pts.filter((p) => p.hit).length,
      }
    })
  if (!rows.length) return <p className="empty-state">Обучение ещё не запускалось — запустите его в пространстве «Мозг».</p>
  return (
    <table>
      <thead>
        <tr>
          <th>Сессия</th>
          <th>Эпизодов</th>
          <th>Промах: начало → конец</th>
          <th>Лучший</th>
          <th>Перехваты</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.session}>
            <td>№ {Math.max(r.session, 1)}</td>
            <td>{r.n}</td>
            <td>
              {rr2(r.first)} → {rr2(r.last)} м
            </td>
            <td className="best">{rr2(r.best)} м</td>
            <td>{Math.round((r.hits / r.n) * 100)}%</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/** Экран «Сводка»: история прогонов и итоги обучения/роя за всё время. */
export function SummaryScreen({ data }: { data: LabData }) {
  const gens = data.gen
  const bestAll = gens.length ? Math.min(...gens.map((g) => g.best)) : null
  const bestGen = bestAll !== null ? gens.findIndex((g) => g.best === bestAll) + 1 : 0
  const champs = gens.map((g) => g.champ).filter((v) => v !== undefined && Number.isFinite(v as number))
  const champ = champs.length ? champs[champs.length - 1] : null

  return (
    <LabScreen
      title="Сводка"
      about="Что произошло за всё время: последние прогоны, итоги сессий обучения и роя. Данные хранятся в этом окне и переживают перезагрузку страницы."
      exports={
        data.runs.length
          ? [
              {
                label: 'История прогонов (CSV)',
                onClick: () =>
                  exportCsv(
                    'muholet-summary.csv',
                    'when,label,miss_m,hit,t_guide_s,ref_dev_m,n_peak,lock_frac,eta_deg',
                    data.runs.map((r) => [new Date(r.at).toISOString(), `"${r.label}"`, r.miss.toFixed(1), r.hit ? 1 : 0, r.tGuide === null ? '' : r.tGuide.toFixed(2), r.refDev === null ? '' : r.refDev.toFixed(1), r.nPeak.toFixed(1), r.lockFrac.toFixed(3), r.eta === null || r.eta === undefined ? '' : r.eta.toFixed(1)]),
                  ),
              },
            ]
          : undefined
      }
      status={
        <>
          <span>
            прогонов <b>{data.runs.length}</b>
          </span>
          <span>
            эпизодов обучения <b>{data.train.length}</b>
          </span>
          <span>
            поколений роя <b>{gens.length}</b>
          </span>
          {champ !== null && (
            <span>
              чемпион роя <b>{rr2(champ as number)} м</b>
            </span>
          )}
        </>
      }
    >
      <div className="lab-grid lab-grid--single" style={{ gap: 16 }}>
        <section className="table-card">
          <table>
            <thead>
              <tr>
                <th>Когда</th>
                <th>Прогон</th>
                <th>Промах, м</th>
                <th>Итог</th>
                <th>Время</th>
                <th>Откл. от эталона</th>
                <th>Пик g</th>
                <th>Захват</th>
                <th data-tip="η — угол между векторами скоростей ракеты и цели в момент наибольшего сближения: 180° — лобовой курс, 0° — догон. Терминальная метрика качества наведения.">η, °</th>
              </tr>
            </thead>
            <tbody>
              {data.runs.length === 0 && (
                <tr>
                  <td colSpan={9}>
                    <p className="empty-state">Прогонов пока не было — нажмите «Пуск» в пространстве «Полёт».</p>
                  </td>
                </tr>
              )}
              {data.runs.map((r, i) => (
                <tr key={r.at + '-' + i}>
                  <td>
                    <small>{new Date(r.at).toLocaleTimeString('ru-RU')}</small>
                  </td>
                  <td>{r.label}</td>
                  <td className="best">{fmt(r.miss, 0)}</td>
                  <td>{r.hit ? 'перехват' : 'мимо'}</td>
                  <td>{fmtT(r.tGuide)}</td>
                  <td>{r.refDev === null ? '—' : fmt(r.refDev, 0)}</td>
                  <td>{fmt(r.nPeak, 1)}</td>
                  <td>{Math.round(r.lockFrac * 100)}%</td>
                  <td>{r.eta === null || r.eta === undefined ? '—' : fmt(r.eta, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <div>
          <h3 className="lab-sub">Сессии обучения</h3>
          <section className="table-card" style={{ marginTop: 6 }}>
            <TrainSessionRows data={data} />
          </section>
        </div>

        <div>
          <h3 className="lab-sub">Рой</h3>
          <div className="lab-screen__status" style={{ marginTop: 6 }}>
            <span>
              лучший за всё время <b>{bestAll !== null ? `${rr2(bestAll)} м` : '—'}</b> {bestGen ? `(поколение ${bestGen})` : ''}
            </span>
            <span>
              чемпион на эталоне <b>{champ !== null ? `${rr2(champ as number)} м` : '—'}</b>
            </span>
            <span>
              перехваты последнего поколения <b>{gens.length ? `${Math.round((gens[gens.length - 1].hitRate ?? 0) * 100)}%` : '—'}</b>
            </span>
          </div>
        </div>
      </div>
    </LabScreen>
  )
}
