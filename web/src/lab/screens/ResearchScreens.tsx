import { useRef } from 'react'
import { CHART, drawChart, exportCsv, palette, useAutoRedraw } from '../charts'
import { LabScreen } from '../LabScreen'
import { ServerOnlyBadge } from '../ServerOnlyBadge'
import type { CoevData, LadderData, MapData, ScalingData, TransferData } from '../types'
import type { BrainKind } from '../types'

const fmt = (v: number | null | undefined, d = 1) => (v === null || v === undefined || !Number.isFinite(v) ? '—' : v.toFixed(d))

const MANEUVER_RU: Record<string, string> = {
  straight: 'прямолинейно',
  turn: 'вираж',
  weave: 'змейка',
  weave_var: 'перем. змейка',
  break: 'резкий вираж',
  scissors: 'ножницы',
  dive: 'горка/пике',
  combo: 'всё сразу',
}

/** Экран «Карта»: карта преимуществ «муха против ПН» + коэволюция. */
export function MapScreen({
  mapData,
  mapBusy,
  onRunMap,
  mapRepeats,
  onMapRepeats,
  mapRetina,
  onMapRetina,
  mapColor,
  onMapColor,
  coev,
  coevBusy,
  onCoevStart,
  onCoevStep,
  onCoevReset,
  onCoevTrain,
  coevTraining,
  coevLadder,
  serverOnline,
}: {
  mapData: MapData | null
  mapBusy: boolean
  onRunMap: () => void
  mapRepeats: number
  onMapRepeats: (v: number) => void
  mapRetina: number
  onMapRetina: (v: number) => void
  mapColor: 'miss' | 'energy'
  onMapColor: (v: 'miss' | 'energy') => void
  coev: CoevData | null
  coevBusy: boolean
  onCoevStart: () => void
  onCoevStep: () => void
  onCoevReset: () => void
  onCoevTrain: () => void
  coevTraining: boolean
  coevLadder: LadderData | null
  serverOnline: boolean | null
}) {
  const offline = serverOnline === false
  return (
    <LabScreen
      title="Карта"
      about="Сетка «манёвр × закон наведения» (5 законов, включая истинная ПН): промах сохранённого мозга против эталона с тем же законом на встречных 8 км. Зелёная ячейка — муха точнее эталона, красная — хуже."
      primary={
        <div className="row" style={{ gap: 8 }}>
          <button type="button" className="primary" disabled={mapBusy || offline} onClick={onRunMap}>
            {mapBusy ? 'Облёт сетки… (2-5 мин)' : 'Построить карту'}
          </button>
          {offline && <ServerOnlyBadge />}
        </div>
      }
      actions={
        <button type="button" className={mapColor === 'energy' ? 'on' : ''} data-tip="Цвет ячеек: по точности (промах) или по «цене» траектории — интегралу перегрузки ∫n²dt." disabled={!mapData} onClick={() => onMapColor(mapColor === 'miss' ? 'energy' : 'miss')}>
          цвет: {mapColor === 'miss' ? 'точность' : 'энергия'}
        </button>
      }
      exports={[
        {
          label: 'Карта преимуществ (CSV)',
          disabled: !mapData,
          onClick: () =>
            mapData &&
            exportCsv(
              'muholet-map.csv',
              'maneuver,law,miss_pn_m,hit_pn,n_int_pn,miss_fly_m,miss_fly_min,miss_fly_max,hit_fly_share,n_int_fly,advantage_m,energy_advantage_m,repeats',
              mapData.rows.map((r) => [r.maneuver, r.law, r.miss_pn.toFixed(1), r.hit_pn ? 1 : 0, (r.n_int_pn ?? 0).toFixed(1), r.miss_fly.toFixed(1), (r.miss_fly_min ?? 0).toFixed(1), (r.miss_fly_max ?? 0).toFixed(1), (r.hit_fly_share ?? 0).toFixed(2), (r.n_int_fly ?? 0).toFixed(1), r.advantage.toFixed(1), ((r.n_int_pn ?? 0) - (r.n_int_fly ?? 0)).toFixed(1), r.repeats ?? 1]),
            ),
        },
      ]}
    >
      <section className="lab-subcard">
        <div className="row">
          <label data-tip="Сколько раз повторить био-прогон в каждой ячейке (шумы меняются): ячейка показывает медиану и разброс.">
            Повторы
            <input type="number" min={1} max={5} value={mapRepeats} onChange={(e) => onMapRepeats(Math.max(1, Math.min(5, Number(e.target.value) || 1)))} />
          </label>
          <label data-tip="Доля «умерших» омматидиев сетчатки при построении карты — карта отказов входа.">
            Отказ сетчатки, %
            <input type="number" min={0} max={80} step={10} value={Math.round(mapRetina * 100)} onChange={(e) => onMapRetina(Math.max(0, Math.min(80, Number(e.target.value) || 0)) / 100)} />
          </label>
        </div>
        {mapBusy && <p className="lab-hint">Облёт сетки: каждая ячейка — прогон мухи и эталонного закона, это занимает несколько минут.</p>}
        {mapData && (
          <>
            <div className="table-card" style={{ border: 'none' }}>
              <table>
                <thead>
                  <tr>
                    <th>Манёвр</th>
                    <th>ПН</th>
                    <th>истинная ПН</th>
                    <th>ПН+комп.</th>
                    <th>Метод погони</th>
                    <th>Метод трёх точек</th>
                  </tr>
                </thead>
                <tbody>
                  {['прямолинейно', 'вираж', 'змейка', 'ножницы', 'горка/пике'].map((m) => (
                    <tr key={m}>
                      <td className="map-maneuver">{m}</td>
                      {[['pn', 'ПН'], ['tpn', 'истинная ПН'], ['apn', 'ПН+комп.'], ['pure', 'метод погони'], ['clos', 'метод трёх точек']].map(([id, l]) => {
                        const cell = mapData.rows.find((r) => r.maneuver === m && r.law_id === id)
                        if (!cell) return <td key={l}>—</td>
                        const adv = mapColor === 'energy' ? (cell.n_int_pn ?? 0) - (cell.n_int_fly ?? 0) : cell.advantage
                        const tie = mapColor === 'energy' ? (cell.n_int_pn ?? 1) * 0.02 : 5
                        const cls = adv > tie ? 'map-win' : adv < -tie ? 'map-lose' : 'map-tie'
                        const tip = mapColor === 'energy'
                          ? `энергия эталона ${cell.n_int_pn?.toFixed(0) ?? '—'} g·с · мухи ${cell.n_int_fly?.toFixed(0) ?? '—'} g·с · промахи ${cell.miss_pn.toFixed(0)}/${cell.miss_fly.toFixed(0)} м`
                          : `эталон ${cell.miss_pn.toFixed(1)} м (${cell.hit_pn ? 'попала' : 'мимо'}) · муха ${cell.miss_fly.toFixed(1)} м (${cell.hit_fly ? 'попала' : 'мимо'})${(cell.repeats ?? 1) > 1 ? ` · ${cell.repeats} повторов` : ''}`
                        return (
                          <td key={l} className={cls} data-tip={tip}>
                            {cell.miss_fly.toFixed(0)} <small>vs {cell.miss_pn.toFixed(0)}</small>
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {mapData.rows[0]?.n_int_fly !== undefined && (
              <>
                <h3 className="lab-sub" style={{ margin: 0 }}>Парето: промах против цены манёвра</h3>
                <svg viewBox="0 0 300 130" className="coev-ladder" role="img" aria-label="Парето: промах против энергии">
                  {[0, 1, 2, 3].map((i) => (
                    <line key={i} x1="34" x2="295" y1={10 + i * 30} y2={10 + i * 30} stroke={palette().grid} />
                  ))}
                  <text x="2" y="16" fill={palette().accent} fontSize="8">∫g·с</text>
                  <text x="240" y="126" fill={palette().accent} fontSize="8">промах, м</text>
                  {mapData.rows.map((r) => {
                    const maxMiss = Math.max(...mapData.rows.map((x) => Math.max(x.miss_fly, x.miss_pn)), 10)
                    const maxY = Math.max(...mapData.rows.map((x) => Math.max(x.n_int_fly ?? 0, x.n_int_pn ?? 0)), 1)
                    const px = (v: number) => 36 + (v / maxMiss) * 255
                    const py = (v: number) => 115 - (v / maxY) * 95
                    return (
                      <g key={`${r.maneuver}-${r.law_id}`}>
                        <circle cx={px(r.miss_pn)} cy={py(r.n_int_pn ?? 0)} r="2.5" fill={palette().amber} opacity="0.85" />
                        <rect x={px(r.miss_fly) - 2.5} y={py(r.n_int_fly ?? 0) - 2.5} width="5" height="5" fill={palette().accent} opacity="0.9" />
                      </g>
                    )
                  })}
                </svg>
                <p className="lab-hint">Квадрат — муха, круг — ПН с тем же законом. Точка левее-ниже = точнее и экономичнее.</p>
              </>
            )}
            <p className="lab-hint">
              В ячейке — промах мухи и для сравнения промах эталона с этим законом. <b>Зелёная</b> — муха точнее (больше чем на 5 м),{' '}
              <b>красная</b> — хуже, серая — паритет. Истинная ПН (TPN) берёт команду по нормали к линии визирования и
              пропорционально закрывающей скорости — на боковых геометриях расходится с обычным ПН; на лобовых ячейки почти совпадают.
              Мозг: {mapData.brain}, {mapData.n_cells.toLocaleString('ru')} нейронов.
            </p>
          </>
        )}
        {!mapData && !mapBusy && <p className="lab-hint">Постройте карту: около минуты облёта сетки. Повторы добавляют устойчивость к шумам.</p>}
      </section>

      <section className="lab-subcard">
        <h3 className="lab-sub" style={{ margin: 0 }}>Коэволюция: цель против мухи</h3>
        <div className="row">
          <button type="button" data-tip="Сбросить популяцию целей и начать гонку вооружений заново." disabled={coevBusy || offline} onClick={onCoevStart}>
            {coevBusy ? 'Запускаю…' : 'Старт'}
          </button>
          <button type="button" data-tip="Одно поколение: цели мутируют, выживает та, от которой муха промахнулась сильнее всего." disabled={coevBusy || offline} onClick={onCoevStep}>
            {coevBusy ? 'Облёт…' : 'Поколение'}
          </button>
          <button type="button" data-tip="Сбросить гонку." disabled={coevBusy} onClick={onCoevReset}>
            Сброс
          </button>
          {offline && <ServerOnlyBadge />}
        </div>
        {coev && (
          <>
            <svg viewBox="0 0 300 60" className="coev-ladder" role="img" aria-label="Лестница коэволюции: чемпион и медиана популяции">
              {coev.history.length > 0 && (
                <>
                  <polyline
                    fill="none"
                    stroke={palette().amber}
                    strokeWidth="1.2"
                    strokeDasharray="3 2"
                    points={coev.history
                      .map((h, i) => {
                        const maxM = Math.max(...coev.history.map((x) => Math.max(x.miss_fly, x.median_miss ?? 0)), 10)
                        const x = coev.history.length > 1 ? (i / (coev.history.length - 1)) * 290 + 5 : 150
                        const y = 55 - (((h.median_miss ?? h.miss_fly)) / maxM) * 50
                        return `${x.toFixed(1)},${y.toFixed(1)}`
                      })
                      .join(' ')}
                  />
                  <polyline
                    fill="none"
                    stroke={palette().accent}
                    strokeWidth="1.5"
                    points={coev.history
                      .map((h, i) => {
                        const maxM = Math.max(...coev.history.map((x) => Math.max(x.miss_fly, x.median_miss ?? 0)), 10)
                        const x = coev.history.length > 1 ? (i / (coev.history.length - 1)) * 290 + 5 : 150
                        const y = 55 - (h.miss_fly / maxM) * 50
                        return `${x.toFixed(1)},${y.toFixed(1)}`
                      })
                      .join(' ')}
                  />
                  {coev.history.map((h, i) => {
                    if (!h.retrained) return null
                    const maxM = Math.max(...coev.history.map((x) => Math.max(x.miss_fly, x.median_miss ?? 0)), 10)
                    const x = coev.history.length > 1 ? (i / (coev.history.length - 1)) * 290 + 5 : 150
                    const y = 55 - (h.miss_fly / maxM) * 50
                    return <circle key={h.gen} cx={x} cy={y} r="3" fill="none" stroke={palette().red} strokeWidth="1.2" />
                  })}
                </>
              )}
            </svg>
            <p className="lab-hint">
              Поколение {coev.gen}. Опаснейшая цель: <b>{coev.best.maneuver}</b>, перегрузка {coev.best.n_target.toFixed(0)}, дальность{' '}
              {(coev.best.range_m / 1000).toFixed(1)} км — муха мажет на {coev.best.miss_fly.toFixed(0)} м. Сплошная линия — чемпион
              (приспособленность самой опасной цели), пунктир — медиана популяции (насколько опасны «обычные» цели); красное кольцо — поколение,
              после которого муху дообучали. Лестница вверх = цель учится уклоняться.
            </p>
            {coev.history.length > 1 && (
              <button
                type="button"
                onClick={() =>
                  exportCsv(
                    'muholet-coevolution.csv',
                    'gen,miss_fly_champion_m,median_miss_pop_m,retrained,fly_before_m,fly_after_m,maneuver,n_target,range_m,off_axis_m',
                    coev.history.map((h) => [h.gen, h.miss_fly, h.median_miss ?? '', h.retrained ? 1 : 0, h.fly_before ?? '', h.fly_after ?? '', h.maneuver, h.n_target ?? '', h.range_m ?? '', h.off_axis_m ?? '']),
                  )
                }
              >
                История (CSV)
              </button>
            )}
          </>
        )}
        <div className="row" style={{ gap: 8 }}>
          <button type="button" className="go" data-tip="Дообучить муху по результату против опаснейших целей коэволюции и замерить лестницу." disabled={coevTraining || !coev || offline} onClick={onCoevTrain}>
            {coevTraining ? 'Дообучаю муху…' : 'Замкнуть петлю: дообучить муху'}
          </button>
          {offline && <ServerOnlyBadge />}
        </div>
        {coevLadder && (
          <p className="lab-hint">
            Лестница: против чемпиона муха промахивалась на <b>{coevLadder.fly_miss_before} м</b>, после дообучения ({coevLadder.generations} поколений) —{' '}
            <b>{coevLadder.fly_miss_after} м</b>
            {coevLadder.fly_miss_after < coevLadder.fly_miss_before ? ' — муха выиграла раунд!' : ' — в этом раунде цель сильнее.'}
          </p>
        )}
      </section>
    </LabScreen>
  )
}

/** Экран «Переносимость»: матрица «обучался на одном — испытан на всех». */
export function TransferScreen({
  transfer,
  transferBusy,
  transferKind,
  onTransferKind,
  onRunTransfer,
  serverOnline,
}: {
  transfer: TransferData | null
  transferBusy: boolean
  transferKind: BrainKind
  onTransferKind: (k: BrainKind) => void
  onRunTransfer: () => void
  serverOnline: boolean | null
}) {
  const offline = serverOnline === false
  return (
    <LabScreen
      title="Переносимость"
      about="Каждая строка — отдельный свежий мозг, обученный только на одном манёвре; столбцы — испытания на всех манёврах. Диагональ — «обучался здесь», вне диагонали — перенос навыка."
      primary={
        <div className="row" style={{ gap: 8 }}>
          <button type="button" className="primary" disabled={transferBusy || offline} onClick={onRunTransfer}>
            {transferBusy ? 'Обучаю и испытываю…' : 'Построить матрицу'}
          </button>
          {offline && <ServerOnlyBadge />}
        </div>
      }
      actions={
        <label data-tip="Какой мозг обучать в матрице. Коннектом считается несколько минут, схема/полный — секунды." className="chip">
          Мозг
          <select value={transferKind} onChange={(e) => onTransferKind(e.target.value as BrainKind)} style={{ width: 'auto' }}>
            <option value="connectome">коннектом</option>
            <option value="full">полный</option>
            <option value="stub">схема</option>
          </select>
        </label>
      }
      exports={[
        {
          label: 'Матрица переносимости (CSV)',
          disabled: !transfer,
          onClick: () =>
            transfer &&
            exportCsv(
              'muholet-transfer.csv',
              'train_maneuver,train_miss_med,' + transfer.test_maneuvers.join(','),
              transfer.rows.map((r) => [r.train, r.train_miss_med.toFixed(1), ...transfer.test_maneuvers.map((t) => (r.tests[t] ?? NaN).toFixed(1))]),
            ),
        },
      ]}
      status={
        transfer ? (
          <>
            <span>
              мозг <b>{({ stub: 'схема', full: 'полный', connectome: 'коннектом' } as Record<string, string>)[transfer.kind] ?? transfer.kind}</b>
            </span>
            <span>
              эпизодов на манёвр <b>{transfer.episodes}</b>
            </span>
            <span>
              считалась <b>{Math.round(transfer.seconds)} с</b>
            </span>
            <span>
              строк <b>{transfer.rows.length}</b>
            </span>
          </>
        ) : transferBusy ? (
          <span>Обучаю свежий мозг на каждом манёвре и облетаю испытания… коннектом займёт несколько минут</span>
        ) : undefined
      }
      note="◉ — диагональ (обучался на этом манёвре). Узкая диагональ и широкая остальная таблица = мозг заучивает, а не обобщает; ровная таблица = навык переносится. Зелёная ячейка — промах малый, красная — большой (шкала по столбцу, логарифм)."
    >
      <section className="table-card">
        {transfer ? (
          <table>
            <thead>
              <tr>
                <th>обучался ↓ / испытан →</th>
                {transfer.test_maneuvers.map((t) => (
                  <th key={t}>{MANEUVER_RU[t] ?? t}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {transfer.rows.map((r) => {
                const colMax: Record<string, number> = {}
                transfer.test_maneuvers.forEach((t) => {
                  colMax[t] = Math.max(...transfer.rows.map((x) => x.tests[t] ?? 0), 1)
                })
                return (
                  <tr key={r.train}>
                    <td className="map-maneuver">
                      {MANEUVER_RU[r.train] ?? r.train}
                      <small> · учёба {r.train_miss_med.toFixed(0)} м</small>
                    </td>
                    {transfer.test_maneuvers.map((t) => {
                      const v = r.tests[t] ?? NaN
                      const tlog = Math.log1p(v) / Math.log1p(colMax[t])
                      const diag = r.train === t
                      const cls = v <= 60 ? 'map-win' : tlog > 0.6 ? 'map-lose' : 'map-tie'
                      return (
                        <td key={t} className={cls} data-tip={`${diag ? 'диагональ: обучался на этом манёвре' : 'перенос: обучался на «' + r.train + '»'} · испытание «${t}» → промах ${v.toFixed(1)} м`}>
                          {v.toFixed(0)}
                          {diag ? ' ◉' : ''}
                        </td>
                      )
                    })}
                  </tr>
                )
              })}
            </tbody>
          </table>
        ) : (
          <p className="empty-state">{transferBusy ? 'Считаю…' : 'Матрица ещё не построена. Выберите мозг и нажмите «Построить матрицу».'}</p>
        )}
      </section>
    </LabScreen>
  )
}

/** Экран «Масштаб мозга»: кривая «размер ↔ точность». */
export function ScalingScreen({
  scaling,
  scalingBusy,
  scalingKind,
  onScalingKind,
  onRunScaling,
  serverOnline,
}: {
  scaling: ScalingData | null
  scalingBusy: boolean
  scalingKind: 'full' | 'connectome'
  onScalingKind: (k: 'full' | 'connectome') => void
  onRunScaling: () => void
  serverOnline: boolean | null
}) {
  const offline = serverOnline === false
  const c1 = useRef<HTMLCanvasElement>(null)
  const c2 = useRef<HTMLCanvasElement>(null)

  useAutoRedraw(() => {
    const rows = scaling?.rows ?? []
    if (c1.current)
      drawChart(
        c1.current,
        [
          { data: rows.map((r) => r.miss_before), color: CHART.minmax(), dash: [3, 2], label: 'до' },
          { data: rows.map((r) => r.miss_after), color: CHART.accent(), width: 2.2, label: 'после' },
          { data: rows.map((r) => r.train_miss_med), color: CHART.amber(), dash: [5, 3], label: 'учёба' },
        ],
        'Промах по каноническому трио, м — размеры мозга по оси X (до обучения / после / медиана учёбы)',
      )
    if (c2.current)
      drawChart(c2.current, [{ data: rows.map((r) => r.params), color: CHART.amber(), width: 2.2, label: 'параметров' }], 'Число обучаемых параметров у размеров мозга (линейный рост — честная шкала исследования)')
  })

  return (
    <LabScreen
      title="Масштаб мозга"
      about="Мозги разного размера обучаются одинаково и сравниваются по каноническому трио: прямая проверка того, окупается ли расширение сети."
      primary={
        <div className="row" style={{ gap: 8 }}>
          <button type="button" className="primary" disabled={scalingBusy || offline} onClick={onRunScaling}>
            {scalingBusy ? 'Растю мозги…' : 'Прогнать серию'}
          </button>
          {offline && <ServerOnlyBadge />}
        </div>
      }
      actions={
        <label data-tip="Чей размер меряем: каналы коннектома 32/64/128 или пул полного мозга 8/16/32/64." className="chip">
          Ось размера
          <select value={scalingKind} onChange={(e) => onScalingKind(e.target.value as 'full' | 'connectome')} style={{ width: 'auto' }}>
            <option value="connectome">каналы коннектома</option>
            <option value="full">пул полного мозга</option>
          </select>
        </label>
      }
      exports={[
        {
          label: 'Кривая масштабируемости (CSV)',
          disabled: !scaling,
          onClick: () =>
            scaling &&
            exportCsv(
              'muholet-scaling.csv',
              'size,params,n_cells,miss_before,miss_after,hit_rate,ref_dev,train_miss_med',
              scaling.rows.map((r) => [r.size, r.params, r.n_cells, r.miss_before.toFixed(1), r.miss_after.toFixed(1), r.hit_rate_after.toFixed(2), r.ref_dev_after.toFixed(1), r.train_miss_med.toFixed(1)]),
            ),
        },
      ]}
      note={
        scaling
          ? `Каждый размер — свежий мозг, одни и те же ${scaling.episodes} эпизодов и один шаг обучения, сравнение на эталонном трио. Если кривая вышла на полку, большие мозги себя не окупают.`
          : scalingBusy
            ? 'Создаю мозги разного размера и обучаю их одинаково… коннектом займёт несколько минут.'
            : 'Нажмите «Прогнать серию»: мозги 32/64/128 каналов (или пул 8/16/32/64) обучатся одинаково и лягут на кривую «параметры ↔ промах».'
      }
    >
      <div className="lab-grid">
        <div className="chart-card"><canvas ref={c1} /></div>
        <div className="chart-card"><canvas ref={c2} /></div>
      </div>
      {scaling && (
        <section className="table-card">
          <table>
            <thead>
              <tr>
                <th>Размер</th>
                <th>Обучаемых параметров</th>
                <th>Нейронов</th>
                <th>Промах до, м</th>
                <th>Промах после, м</th>
                <th>Перехваты</th>
                <th>Откл. от ПН, м</th>
              </tr>
            </thead>
            <tbody>
              {scaling.rows.map((r) => (
                <tr key={r.size}>
                  <td>{scaling.kind === 'connectome' ? `${r.size} каналов` : `пул ${r.size}`}</td>
                  <td>{r.params.toLocaleString('ru')}</td>
                  <td>{r.n_cells.toLocaleString('ru')}</td>
                  <td>{r.miss_before.toFixed(1)}</td>
                  <td className={r.miss_after === Math.min(...scaling.rows.map((x) => x.miss_after)) ? 'best' : ''}>
                    <b>{r.miss_after.toFixed(1)}</b>
                  </td>
                  <td>{Math.round(r.hit_rate_after * 100)}%</td>
                  <td>{fmt(r.ref_dev_after, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </LabScreen>
  )
}
