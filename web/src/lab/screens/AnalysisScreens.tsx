import { useEffect, useRef, useState } from 'react'
import { CHART, download, drawChart, drawHist, exportCsv, palette, pearson, pctSorted, useAutoRedraw } from '../charts'
import { LabScreen } from '../LabScreen'
import type { AblationData, CaptureZoneData, CoevData, CmpRow, DistillData, FaultsData, LadderData, MonteCarloData } from '../types'
import type { BrainKind, LabData } from '../types'

const fmt = (v: number | null | undefined, d = 1) => (v === null || v === undefined || !Number.isFinite(v) ? '—' : v.toFixed(d))
const fmtT = (v: number | null | undefined) => (v === null || v === undefined ? '—' : `${v.toFixed(2)} с`)
const fmtNum = (v: number | null) => (v === null || !Number.isFinite(v) ? '—' : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2))

const SPEED_MODE_RU: Record<string, string> = {
  constant: 'постоянная',
  accelerate: 'разгон',
  decelerate: 'торможение',
  pulse: 'импульс',
  sine: 'синусоида',
}

/** Экран «Сравнение»: сценарий на всех мозгах и законах + испытания мозгов (абляция, отказы, дистилляция). */
export function CompareScreen({
  compare,
  compareLoading,
  onRunCompare,
  onOverlay,
  overlayBusy,
  ablation,
  ablationBusy,
  onRunAblation,
  faults,
  faultsBusy,
  onRunFaults,
  distill,
  distillBusy,
  onRunDistill,
  onApplyDistill,
  distillTeacher,
  onDistillTeacher,
  onDistillSave,
  onDistillLoad,
  distillSaved,
}: {
  compare: { rows?: CmpRow[]; error?: string } | null
  compareLoading: boolean
  onRunCompare: () => void
  onOverlay: () => void
  overlayBusy: boolean
  ablation: AblationData | null
  ablationBusy: boolean
  onRunAblation: () => void
  faults: FaultsData | null
  faultsBusy: boolean
  onRunFaults: () => void
  distill: DistillData | null
  distillBusy: boolean
  onRunDistill: () => void
  onApplyDistill: () => void
  distillTeacher: 'connectome' | 'full' | 'stub'
  onDistillTeacher: (t: 'connectome' | 'full' | 'stub') => void
  onDistillSave: () => void
  onDistillLoad: () => void
  distillSaved: boolean
}) {
  return (
    <LabScreen
      title="Сравнение"
      about="Текущий сценарий прогоняется на эталонном МПС, законах наведения и всех мозгах. Эталонный МПС видит точную геометрию, поэтому это сравнение «с подсказкой»."
      primary={
        <button type="button" className="primary" disabled={compareLoading} onClick={onRunCompare}>
          {compareLoading ? 'Сравниваю…' : 'Прогнать сравнение'}
        </button>
      }
      actions={
        <button type="button" disabled={overlayBusy} data-tip="Прогнать сравнение и наложить все траектории на 3D-сцену." onClick={onOverlay}>
          {overlayBusy ? 'Снимаю траектории…' : 'Наложить на сцену'}
        </button>
      }
      exports={[
        {
          label: 'Таблица сравнения (CSV)',
          disabled: !compare?.rows?.length,
          onClick: () =>
            exportCsv(
              'muholet-compare.csv',
              'variant,miss_m,hit,t_guide_s,ref_dev_m,match_pct,n_peak,lock_frac',
              (compare?.rows ?? []).map((r) => [
                `"${r.label}${r.trained ? ' ✓' : ''}"`,
                r.miss_m.toFixed(1),
                r.hit ? 1 : 0,
                r.t_guide === null || r.t_guide === undefined ? '' : r.t_guide.toFixed(2),
                r.ref_dev_m === null || r.ref_dev_m === undefined ? '' : r.ref_dev_m.toFixed(1),
                r.ref_nrms === null || r.ref_nrms === undefined ? '' : (Math.max(0, 1 - r.ref_nrms) * 100).toFixed(1),
                r.n_peak.toFixed(1),
                r.lock_frac.toFixed(3),
              ]),
            ),
        },
      ]}
      status={compare?.error ? <span style={{ color: 'var(--status-danger)' }}>{compare.error}</span> : compareLoading ? <span>Считаю сценарий на всех мозгах и законах…</span> : undefined}
      note={
        <>
          Для сенсорных участников (био, «МПС через ГСН») отклонение показывает похожесть на эталонную траекторию; норм. СКО — рассогласование
          с эталоном, нормированное дальностью (меньше = ближе). Время наведения сравнивается с эталонным {fmtT(compare?.rows?.find((r) => r.kind === 'pn')?.t_guide ?? null)}.
        </>
      }
    >
      <section className="table-card">
        {compare?.rows ? (
          <table>
            <thead>
              <tr>
                <th>Вариант</th>
                <th>Промах, м</th>
                <th>Итог</th>
                <th>Время наведения</th>
                <th>Откл. от эталона, м</th>
                <th>норм. СКО</th>
                <th>Пик g</th>
                <th>Захват</th>
              </tr>
            </thead>
            <tbody>
              {compare.rows.map((r) => (
                <tr key={r.kind} className={r.kind === 'pn' ? 'ref' : ''}>
                  <td>
                    {r.label}
                    {r.trained === true ? ' ✓' : ''}
                  </td>
                  <td className={r.kind !== 'pn' && r.miss_m === Math.min(...compare.rows!.filter((x) => x.kind !== 'pn').map((x) => x.miss_m)) ? 'best' : ''}>{fmt(r.miss_m, 0)}</td>
                  <td>{r.hit ? 'перехват' : 'мимо'}</td>
                  <td>{fmtT(r.t_guide ?? null)}</td>
                  <td>{fmt(r.ref_dev_m ?? null, 0)}</td>
                  <td data-tip="СКО рассогласования с эталонным МПС, нормированное начальной дальностью. Меньше = ближе к эталону. Это не «процент совпадения».">
                    {r.ref_nrms === null || r.ref_nrms === undefined ? '—' : (r.ref_nrms * 100).toFixed(1)}
                  </td>
                  <td>{fmt(r.n_peak, 1)}</td>
                  <td>{(r.lock_frac * 100).toFixed(0)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="empty-state">
            {compareLoading
              ? 'Считаю…'
              : 'Нажмите «Прогнать сравнение»: строки — эталонный МПС, законы наведения, все мозги. Метрики — конечный промах, время наведения, похожесть на эталон, пик перегрузки, доля захвата.'}
          </p>
        )}
      </section>

      <div>
        <h3 className="lab-sub">Испытания мозгов</h3>
        <div className="lab-grid" style={{ marginTop: 6, alignItems: 'start' }}>
          <section className="lab-subcard">
            <h3 className="lab-sub" style={{ margin: 0 }}>Абляция зон коннектома</h3>
            <button type="button" data-tip="Поочерёдно выключаем зоны синаптических каналов мозга и меряем каноническое трио. Строится ~20 секунд." disabled={ablationBusy} onClick={onRunAblation}>
              {ablationBusy ? 'Испытываю зоны…' : 'Тест абляции зон'}
            </button>
            {ablation && (
              <table className="legend-table">
                <thead>
                  <tr>
                    <td>Зона</td>
                    <td>Каналов</td>
                    <td>Промах, м</td>
                    <td>Перехваты</td>
                    <td>Откл. от МПС, м</td>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>без абляции</td>
                    <td>128</td>
                    <td>{ablation.base.miss.toFixed(1)}</td>
                    <td>{Math.round(ablation.base.hit_rate * 100)}%</td>
                    <td>{ablation.base.ref_dev === null ? '—' : ablation.base.ref_dev.toFixed(1)}</td>
                  </tr>
                  {ablation.rows.map((r) => {
                    const worst = Math.max(...ablation.rows.map((x) => x.miss))
                    const isWorst = r.miss === worst && ablation.rows.length > 1
                    return (
                      <tr key={r.zone} className={isWorst ? 'abl-worst' : ''}>
                        <td>{r.zone}{isWorst ? ' ← критичнее всех' : ''}</td>
                        <td>{r.channels}</td>
                        <td>{r.miss.toFixed(1)}</td>
                        <td>{Math.round(r.hit_rate * 100)}%</td>
                        <td>{r.ref_dev === null ? '—' : r.ref_dev.toFixed(1)}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            )}
            <p className="lab-hint">
              Чем сильнее вырос промах после выключения зоны, тем важнее зона (подсвечена самая критичная). Прямые каналы
              (копии признаков) не выключаются — это врождённый каркас.
            </p>
          </section>

          <section className="lab-subcard">
            <h3 className="lab-sub" style={{ margin: 0 }}>Карта отказов сетчатки</h3>
            <button type="button" data-tip="Деградация при «умирании» части омматидиев: до 75% сетчатки слепнет; считается ~4-5 минут." disabled={faultsBusy} onClick={onRunFaults}>
              {faultsBusy ? 'Отказываю омматидиям…' : 'Карта отказов сетчатки'}
            </button>
            {faults && (
              <>
                <svg viewBox="0 0 300 90" className="coev-ladder" role="img" aria-label="Промах при отказе омматидиев">
                  <polyline
                    fill="none"
                    stroke={palette().amber}
                    strokeWidth="1.5"
                    points={faults.rows
                      .map((r, i) => {
                        const maxMiss = Math.max(...faults.rows.map((x) => x.miss), 10)
                        const x = 5 + (i / (faults.rows.length - 1)) * 290
                        const y = 80 - (r.miss / maxMiss) * 70
                        return `${x.toFixed(1)},${y.toFixed(1)}`
                      })
                      .join(' ')}
                  />
                  {faults.rows.map((r, i) => (
                    <text key={i} x={5 + (i / (faults.rows.length - 1)) * 290 - 8} y="89" fill={palette().dim} fontSize="8">
                      {Math.round(r.fraction * 100)}%
                    </text>
                  ))}
                </svg>
                <p className="lab-hint">
                  Промах (м) при отказе 0–80% омматидиев. Пологая кривая — зрение робастно; резкий рост — мозг держится на
                  конкретных каналах.
                </p>
              </>
            )}
          </section>

          <section className="lab-subcard">
            <h3 className="lab-sub" style={{ margin: 0 }}>Дистилляция: большой мозг → схема</h3>
            <div className="row">
              <label data-tip="Кто учит маленькую схему: коннектом (FlyWire), полный мозг или сама схема.">
                Учитель
                <select value={distillTeacher} onChange={(e) => onDistillTeacher(e.target.value as 'connectome' | 'full' | 'stub')}>
                  <option value="connectome">коннектом</option>
                  <option value="full">полный</option>
                  <option value="stub">схема</option>
                </select>
              </label>
              <button type="button" data-tip="Схема (16 весов) учится повторять команды учителя. Основную схему не трогаем — до кнопки «Применить»." disabled={distillBusy} onClick={onRunDistill}>
                {distillBusy ? 'Дистиллирую…' : 'Дистиллировать'}
              </button>
            </div>
            {distill && (
              <table className="legend-table">
                <thead>
                  <tr>
                    <td>Вариант</td>
                    <td>Промах, м</td>
                    <td>Перехваты</td>
                    <td>Откл. от МПС, м</td>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>учитель: {distill.teacher_kind ?? 'коннектом'}</td>
                    <td>{distill.teacher.miss.toFixed(1)}</td>
                    <td>{Math.round(distill.teacher.hit_rate * 100)}%</td>
                    <td>—</td>
                  </tr>
                  <tr>
                    <td>дистиллят (схема 2×8)</td>
                    <td className="best">{distill.distilled.miss.toFixed(1)}</td>
                    <td>{Math.round(distill.distilled.hit_rate * 100)}%</td>
                    <td>{distill.distilled.ref_dev.toFixed(1)}</td>
                  </tr>
                  <tr>
                    <td>схема с нуля (рефлекс)</td>
                    <td>{distill.scratch.miss.toFixed(1)}</td>
                    <td>{Math.round(distill.scratch.hit_rate * 100)}%</td>
                    <td>{distill.scratch.ref_dev.toFixed(1)}</td>
                  </tr>
                </tbody>
              </table>
            )}
            <div className="row">
              <button type="button" className="go" data-tip="Записать веса дистиллята в схему стенда (с сохранением)." disabled={!distill} onClick={onApplyDistill}>
                Применить к схеме
              </button>
              <button type="button" data-tip="Сохранить дистиллят в data/weights_distilled.npz." disabled={!distill} onClick={onDistillSave}>
                Сохранить в файл
              </button>
              <button type="button" data-tip="Загрузить сохранённый дистиллят обратно." onClick={onDistillLoad}>
                Загрузить из файла
              </button>
              {distillSaved && <span className="lab-hint">сохранён ✓</span>}
            </div>
            <p className="lab-hint">
              Конвейер: учитель → дистиллят → файл → применение → дообучение «по результату». Откл. от МПС у дистиллята
              выше, чем у рефлекса, — он летит «как учитель», так и задумано.
            </p>
          </section>
        </div>
      </div>
    </LabScreen>
  )
}

/** Экран «N_экв»: диагностика адаптивности навигационного коэффициента. */
export function NeffScreen({ data, onExport }: { data: LabData; onExport: () => void }) {
  const c1 = useRef<HTMLCanvasElement>(null)
  const c2 = useRef<HTMLCanvasElement>(null)
  const c3 = useRef<HTMLCanvasElement>(null)
  const c4 = useRef<HTMLCanvasElement>(null)

  const pts = data.neff
  const valid = pts.filter((p) => p.valid && p.nEff !== null && Number.isFinite(p.nEff))
  const vals = valid.map((p) => p.nEff as number)
  const sorted = [...vals].sort((a, b) => a - b)
  const med = pctSorted(sorted, 0.5)
  const q25 = pctSorted(sorted, 0.25)
  const q75 = pctSorted(sorted, 0.75)
  const mn = sorted.length ? sorted[0] : null
  const mx = sorted.length ? sorted[sorted.length - 1] : null
  const satFrac = pts.length ? pts.filter((p) => p.sat).length / pts.length : 0
  const validFrac = pts.length ? valid.length / pts.length : 0
  const corr = pearson(vals, valid.map((p) => p.rho))
  const modes: Record<string, number[]> = {}
  valid.forEach((p) => {
    ;(modes[p.speedMode] ??= []).push(p.nEff as number)
  })

  useAutoRedraw(() => {
    const medC = med ?? NaN
    const q25C = q25 ?? NaN
    const q75C = q75 ?? NaN
    if (c1.current)
      drawChart(
        c1.current,
        [
          { data: vals, color: CHART.accent(), label: 'N_экв' },
          { data: vals.map(() => medC), color: CHART.amber(), width: 2, label: 'медиана' },
          { data: vals.map(() => q25C), color: CHART.minmax(), dash: [4, 3], label: 'Q25' },
          { data: vals.map(() => q75C), color: CHART.minmax(), dash: [4, 3], label: 'Q75' },
        ],
        'N_экв по времени прогона (только валидные отсчёты; медиана и квартили — пунктир)',
      )
    const byRho = [...valid].sort((a, b) => a.rho - b.rho)
    if (c2.current)
      drawChart(
        c2.current,
        [
          { data: byRho.map((p) => p.nEff as number), color: CHART.accent(), label: 'N_экв' },
          { data: byRho.map(() => medC), color: CHART.amber(), width: 2, label: 'медиана' },
        ],
        'N_экв против ρ (кадры по возрастанию ρ): наклон/разлёт = адаптивность коэффициента',
      )
    const byTgo = [...valid].sort((a, b) => (a.tgo ?? 1e9) - (b.tgo ?? 1e9))
    if (c3.current)
      drawChart(
        c3.current,
        [
          { data: byTgo.map((p) => p.nEff as number), color: CHART.accent(), label: 'N_экв' },
          { data: byTgo.map(() => medC), color: CHART.amber(), width: 2, label: 'медиана' },
        ],
        'N_экв против t_cpa (кадры по возрастанию t_cpa): рост усиления к контакту виден слева',
      )
    const byVc = [...valid].sort((a, b) => a.vc - b.vc)
    if (c4.current)
      drawChart(
        c4.current,
        [
          { data: byVc.map((p) => p.nEff as number), color: CHART.accent(), label: 'N_экв' },
          { data: byVc.map(() => medC), color: CHART.amber(), width: 2, label: 'медиана' },
        ],
        'N_экв против скорости сближения Vc (кадры по возрастанию Vc) — операторская диагностика',
      )
  })

  return (
    <LabScreen
      title="N_экв"
      about="Эквивалентный навигационный коэффициент N_экв = ⟨a, q⟩/⟨q, q⟩ — «какому постоянному N эквивалентна команда». Истинная геометрия используется постфактум для анализа, но НЕ для управления БИО."
      exports={[{ label: 'Диагностика N_экв (CSV)', disabled: !data.neff.length, onClick: onExport }]}
      status={
        <>
          <span>
            медиана <b>{fmtNum(med)}</b>
          </span>
          <span>
            квартили <b>{med !== null && q25 !== null && q75 !== null ? `${fmtNum(q25)}…${fmtNum(q75)}` : '—'}</b>
          </span>
          <span>
            мин/макс <b>{mn !== null && mx !== null ? `${fmtNum(mn)} / ${fmtNum(mx)}` : '—'}</b>
          </span>
          <span>
            валидных <b>{Math.round(validFrac * 100)}%</b>
          </span>
          <span>
            насыщение <b>{Math.round(satFrac * 100)}%</b>
          </span>
          <span>
            корр(N_экв, ρ) <b>{corr !== null ? corr.toFixed(2) : '—'}</b>
          </span>
        </>
      }
      note="Постоянный МПС даёт N_экв ≡ N; если у БИО медиана постоянна, а квартили узки при любом ρ — политика близка к классическому МПС; если N_экв растёт с ρ (при уменьшении t_cpa) — коэффициент адаптивен. На насыщении и при ω≈0 оценка помечается невалидной и в статистику не идёт."
    >
      <div className="lab-grid">
        <div className="chart-card"><canvas ref={c1} /></div>
        <div className="chart-card"><canvas ref={c2} /></div>
        <div className="chart-card"><canvas ref={c3} /></div>
        <div className="chart-card"><canvas ref={c4} /></div>
      </div>
      {Object.keys(modes).length > 0 && (
        <section className="table-card">
          <table>
            <thead>
              <tr>
                <th>Профиль скорости</th>
                <th>Отсчётов</th>
                <th>Медиана N_экв</th>
                <th>Q25</th>
                <th>Q75</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(modes).map(([mode, arr]) => {
                const srt = [...arr].sort((a, b) => a - b)
                return (
                  <tr key={mode}>
                    <td>{SPEED_MODE_RU[mode] ?? mode}</td>
                    <td>{arr.length}</td>
                    <td>
                      <b>{fmtNum(pctSorted(srt, 0.5))}</b>
                    </td>
                    <td>{fmtNum(pctSorted(srt, 0.25))}</td>
                    <td>{fmtNum(pctSorted(srt, 0.75))}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </section>
      )}
    </LabScreen>
  )
}

/** Экран «Робастность»: серия прогонов с нарастающим шумом. */
export function RobScreen({
  rob,
  robBusy,
  onRunRobustness,
}: {
  rob: { levels: number[]; series: { kind: string; label: string; miss: number[]; nrms: (number | null)[] }[] } | null
  robBusy: boolean
  onRunRobustness: () => void
}) {
  const c1 = useRef<HTMLCanvasElement>(null)
  const c2 = useRef<HTMLCanvasElement>(null)

  useAutoRedraw(() => {
    const lineColors = [CHART.accent(), CHART.amber(), CHART.red()]
    if (c1.current)
      drawChart(
        c1.current,
        (rob?.series ?? []).map((sr, i) => ({ data: sr.miss, color: lineColors[i % lineColors.length], width: 2, label: sr.label })),
        'Промах vs шум пеленга, м (серии — мозги)',
      )
    if (c2.current)
      drawChart(
        c2.current,
        (rob?.series ?? []).map((sr, i) => ({
          data: sr.nrms.map((v) => (v === null ? NaN : v)).filter((v) => Number.isFinite(v)),
          color: lineColors[i % lineColors.length],
          width: 2,
          label: sr.label,
        })),
        'Совпадение с эталоном 1−СКО vs шум, отн. ед.',
      )
  })

  return (
    <LabScreen
      title="Робастность"
      about="Серия прогонов с нарастающим шумом пеленга (0–4°): сравнение мозгов по деградации промаха и похожести на эталон."
      primary={
        <button type="button" className="primary" disabled={robBusy} onClick={onRunRobustness}>
          {robBusy ? 'Считаю…' : 'Прогнать серию'}
        </button>
      }
      exports={[
        {
          label: 'Робастность (CSV)',
          disabled: !rob,
          onClick: () => {
            if (!rob) return
            download(
              'muholet-robust.csv',
              'noise_deg,' + rob.series.map((sr) => '"' + sr.label + '"').join(',') + '\n' +
                rob.levels.map((lvl, i) => lvl + ',' + rob.series.map((sr) => sr.miss[i]).join(',')).join('\n'),
            )
          },
        },
      ]}
      status={
        <>
          <span>
            уровни шума, °: <b>{rob?.levels.join(' → ') ?? '—'}</b>
          </span>
          <span>
            серий <b>{rob?.series.length ?? 0}</b>
          </span>
        </>
      }
      note="Робастность считается только на работающем стенде. Серии идут по всем мозгам: схема, полный мозг, коннектом."
    >
      <div className="lab-grid">
        <div className="chart-card"><canvas ref={c1} /></div>
        <div className="chart-card"><canvas ref={c2} /></div>
      </div>
    </LabScreen>
  )
}

/** Экран «Разброс»: Monte-Carlo серия одного сценария с разными seed. */
export function McScreen({
  mc,
  mcBusy,
  mcRuns,
  onMcRuns,
  onRunMc,
}: {
  mc: MonteCarloData | null
  mcBusy: boolean
  mcRuns: number
  onMcRuns: (n: number) => void
  onRunMc: () => void
}) {
  const cv = useRef<HTMLCanvasElement>(null)

  useAutoRedraw(() => {
    if (!cv.current) return
    const vals = (mc?.per_run ?? []).map((p) => p.r_min_m).filter((v) => Number.isFinite(v))
    if (!vals.length) {
      drawHist(cv.current, [], 'R_min по прогонам, м')
      return
    }
    const lo = Math.min(...vals)
    const hi = Math.max(...vals)
    const nb = Math.min(24, Math.max(6, Math.ceil(Math.sqrt(vals.length))))
    const step = Math.max((hi - lo) / nb, 1e-9)
    const bins = Array.from({ length: nb }, (_, i) => ({ lo: lo + i * step, hi: lo + (i + 1) * step, n: 0 }))
    for (const v of vals) bins[Math.min(nb - 1, Math.floor((v - lo) / step))].n += 1
    drawHist(
      cv.current,
      bins,
      `Гистограмма R_min (${vals.length} прогонов), м`,
      mc ? { x: mc.trigger_range_m, label: `сфера срабатывания ${Math.round(mc.trigger_range_m)} м` } : undefined,
    )
  })

  const stat = (label: string, value: string) => (
    <span>
      {label}: <b>{value}</b>
    </span>
  )

  return (
    <LabScreen
      title="Разброс"
      about="Monte-Carlo рассеивания: один сценарий летит n раз с разными seed — шум измерений, срывы сопровождения и отказы сетчатки дают разброс R_min. Классическая оценка точки прицела (dispersion of aim point)."
      primary={
        <button type="button" className="primary" disabled={mcBusy} onClick={onRunMc}>
          {mcBusy ? 'Считаю…' : 'Прогнать серию'}
        </button>
      }
      actions={
        <label data-tip="Число независимых прогонов серии (каждый со своим seed). Больше — уже ДИ вероятности перехвата, но медленнее: ~1 с на прогон." style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          Прогонов
          <input type="number" min={4} max={500} step={1} value={mcRuns} style={{ width: 70 }} onChange={(e) => onMcRuns(Math.max(4, Math.min(500, Number(e.target.value) || 4)))} />
        </label>
      }
      exports={[
        {
          label: 'Серия (CSV)',
          disabled: !mc,
          onClick: () => exportCsv('muholet-monte-carlo.csv', 'seed,r_min_m,hit,reason,h0_m', (mc?.per_run ?? []).map((p) => [p.seed, p.r_min_m, p.hit ? 1 : 0, p.reason, p.h0_m ?? ''])),
        },
      ]}
      status={
        mcBusy ? (
          <span>Летят {mcRuns} прогонов…</span>
        ) : !mc ? (
          <span>Серия ещё не считана — нажмите «Прогнать серию» (нужен работающий стенд).</span>
        ) : (
          <>
            {stat('p_hit', `${(mc.p_hit * 100).toFixed(0)} %`)}
            {stat('95 % ДИ', `${(mc.p_hit_ci95[0] * 100).toFixed(0)}…${(mc.p_hit_ci95[1] * 100).toFixed(0)} %`)}
            {stat('R_min ср', `${Math.round(mc.r_min_mean_m)} м`)}
            {stat('СКО', mc.r_min_std_m === null ? '—' : `${Math.round(mc.r_min_std_m)} м`)}
            {stat('медиана', `${Math.round(mc.r_min_median_m ?? NaN)} м`)}
            {stat('P90', mc.r_min_p90_m === null ? '—' : `${Math.round(mc.r_min_p90_m)} м`)}
            {stat('R_95', mc.r_95_m === null ? '—' : `${Math.round(mc.r_95_m)} м`)}
            {stat('CEP_50', mc.cep50_m === null ? '—' : `${Math.round(mc.cep50_m)} м`)}
          </>
        )
      }
      note={
        mc?.deterministic
          ? 'СКО = 0: в текущем сценарии нет стохастики (нулевой шум/срывы) или закон не читает измерительный канал — все прогоны совпадают побитно. Добавьте шум пеленга или возьмите сенсорный закон (pn_gsn/БИО).'
          : 'p_hit — доля входов в сферу срабатывания (поражающее действие НЕ моделируется); ДИ — точный биномиальный (Уилсон). CEP_50 — медианный промах: половина выстрелов попадает в этот радиус.'
      }
    >
      <div className="lab-grid lab-grid--single">
        <div className="chart-card" style={{ height: 260 }}>
          <canvas ref={cv} />
        </div>
      </div>
    </LabScreen>
  )
}

/** Экран «Зона перехвата»: сетка «дальность пуска × постоянная перегрузка цели». */
export function CzScreen({
  cz,
  czBusy,
  czGmax,
  onCzGmax,
  czRuns,
  onCzRuns,
  onRunCz,
}: {
  cz: CaptureZoneData | null
  czBusy: boolean
  czGmax: number
  onCzGmax: (g: number) => void
  czRuns: number
  onCzRuns: (n: number) => void
  onRunCz: () => void
}) {
  const stat = (label: string, value: string) => (
    <span>
      {label}: <b>{value}</b>
    </span>
  )
  const km = (m: number) => `${(m / 1000).toFixed(1)} км`
  const rowNote = (r: CaptureZoneData['rows'][number]): string => {
    if (r.status === 'all_hit') return 'перехват всюду сетки'
    if (r.status === 'no_hit') return 'нет перехвата на сетке'
    if (r.status === 'lobed') return `долепестно: блоков перехвата ${r.hit_blocks}`
    const edges = [
      r.inner_boundary_m !== undefined ? `не успевает ближе ${km(r.inner_boundary_m)}` : '',
      r.boundary_m !== undefined
        ? `не достанет дальше ${km(r.boundary_m)}${r.boundary_time_limited ? '*' : ''}`
        : '',
    ].filter(Boolean)
    return edges.join(' · ') || 'один блок перехвата'
  }

  return (
    <LabScreen
      title="Зона перехвата"
      about="Зона неубегаемого перехвата: цель уходит с ПОСТОЯННОЙ нормальной перегрузкой, для каждой перегрузки ищем ОТРЕЗОК дальностей пуска, где перехват возможен. Классическая задача уклонения (Дмитрий, Ровинский, Юрьев); границы — сетка прогонов + бисекция скобок «промах → попал» (ближняя: ракета не успевает развернуться) и «попал → промах» (дальняя: цель уходит). При «прогонов на ячейку» больше 1 зона становится вероятностной: каждая ячейка — серия Monte-Carlo, цвет и число — p_hit с 95 % ДИ (Уилсон)."
      primary={
        <button type="button" className="primary" disabled={czBusy} onClick={onRunCz}>
          {czBusy ? 'Считаю…' : 'Построить зону'}
        </button>
      }
      actions={
        <>
        <label data-tip="Верх сетки по перегрузке цели, g: строки 0…G равномерно (6 шт). Дальняя граница для ПН — учебная опорная точка (N−1)/N·n_max: при N=4 и 30 g цель убегает от 22.5 g. Ближняя — время разворота: граница растёт как √смещения цели и падает как 1/√n_max, то есть слабой ракете нужен больший запас дальности. Полный скан ~минута: до ~50 прогонов замкнутого контура." style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          Перегрузка цели до, g
          <input type="number" min={5} max={40} step={1} value={czGmax} style={{ width: 70 }} onChange={(e) => onCzGmax(Math.max(5, Math.min(40, Number(e.target.value) || 5)))} />
        </label>
        <label data-tip="Прогонов Monte-Carlo на каждую ячейку сетки (seed = 1000, 1001, …). 1 — прежний одиночный прогон с бисекцией границ; больше 1 — ячейка показывает долю попаданий p_hit с 95 % ДИ (Уилсон), бисекция границ пропускается, а счёт идёт в 6 × 6 × N прогонов (минуты на 8). Для дробных долей нужен источник стохастики: шум измерений, срывы захвата или сенсорный закон.">
          Прогонов на ячейку
          <input type="number" min={1} max={25} step={1} value={czRuns} style={{ width: 60 }} onChange={(e) => onCzRuns(Math.max(1, Math.min(25, Math.round(Number(e.target.value) || 1))))} />
        </label>
        </>
      }
      exports={[
        {
          label: 'Сетка (CSV)',
          disabled: !cz,
          onClick: () => {
            if (!cz) return
            const mc = (cz.n_runs ?? 1) > 1
            exportCsv(
              'muholet-capture-zone.csv',
              mc
                ? 'n_target_g,range_m,hit_majority,p_hit,ci95_lo,ci95_hi,r_min_median_m,t_end_s,time_limited,deterministic_cell'
                : 'n_target_g,range_m,hit,r_min_m,t_end_s,time_limited',
              cz.rows.flatMap((r) =>
                r.cells.map((c) =>
                  mc
                    ? [r.n_target_g, c.range_m, c.hit ? 1 : 0, c.p_hit ?? '', c.p_hit_ci95?.[0] ?? '', c.p_hit_ci95?.[1] ?? '', c.median_cpa_m ?? c.r_min_m, c.t_end_s, c.time_limited ? 1 : 0, c.deterministic_cell ? 1 : 0]
                    : [r.n_target_g, c.range_m, c.hit ? 1 : 0, c.r_min_m, c.t_end_s, c.time_limited ? 1 : 0],
                ),
              ),
            )
          },
        },
        {
          label: 'Границы зоны (CSV)',
          disabled: !cz || (cz.n_runs ?? 1) > 1,
          onClick: () => {
            if (!cz) return
            exportCsv(
              'muholet-capture-zone-boundaries.csv',
              'n_target_g,status,hit_blocks,inner_boundary_m,inner_bracket_lo_m,inner_bracket_hi_m,boundary_m,bracket_lo_m,bracket_hi_m,boundary_time_limited',
              cz.rows.map((r) => [
                r.n_target_g,
                r.status,
                r.hit_blocks,
                r.inner_boundary_m ?? '',
                r.inner_boundary_bracket_m?.[0] ?? '',
                r.inner_boundary_bracket_m?.[1] ?? '',
                r.boundary_m ?? '',
                r.boundary_bracket_m?.[0] ?? '',
                r.boundary_bracket_m?.[1] ?? '',
                r.boundary_time_limited === undefined ? '' : r.boundary_time_limited ? 1 : 0,
              ]),
            )
          },
        },
      ]}
      status={
        czBusy ? (
          <span>Летит сетка {6 * 6 * Math.max(1, czRuns)} прогонов{czRuns > 1 ? ' (серии Monte-Carlo, бисекция пропущена)' : ' + бисекция границ'}…</span>
        ) : !cz ? (
          <span>Зона ещё не считана — нажмите «Построить зону» (нужен работающий стенд).</span>
        ) : (
          <>
            {stat('режим', cz.model === 'point_mass_3dof' ? 'трёхстепенная' : 'кинематическая')}
            {(cz.n_runs ?? 1) > 1 && stat('прогонов/ячейку', `${cz.n_runs}`)}
            {stat('N', fmt(cz.pn_n, 1))}
            {stat('n_max', `${fmt(cz.n_max_g, 0)} g`)}
            {stat('окно', `${fmt(cz.t_max_s, 0)} с`)}
            {stat('время', `${fmt(cz.seconds, 0)} с`)}
          </>
        )
      }
      note={
        cz
          ? cz.series_deterministic
            ? 'Все серии ячеек совпали побитно: в текущем сценарии нет стохастики (нулевой шум/срывы) или закон не читает измерительный канал — вероятностная зона вырождается в бинарную. Добавьте шум пеленга или возьмите сенсорный закон (pn_gsn/БИО).'
            : (cz.n_runs ?? 1) > 1
              ? `Ячейки — доли серии из ${cz.n_runs} прогонов Monte-Carlo (seed от ${cz.seed_start}): p_hit с 95 % ДИ (Уилсон), раскраска по p_hit; бинарный статус строки — большинство (p_hit ≥ 0.5). Бисекция численных границ в этом режиме не выполняется — скобка «попал→промах» на majority-исходах была бы статистикой из одного выстрела; поставьте 1 прогон, чтобы получить её.`
              : 'Зона — отрезок дальностей, если перехват на сетке образует ОДИН связный блок: бисекцией находят ближнюю границу (на малых дальностях ракете не хватает времени развернуться на упреждение) и дальнюю (цель уходит). Блоков перехвата несколько — это долепестность (исходы зависят от фазы разворота цели), численных границ для такой строки не ищем. Промах при 0 g — предел манёвренности, а не убегание цели. * у дальней границы — промах по истечении окна прогона (time_limited): это граница окна t_max, а не физики.'
          : 'Скан прогоняет текущий сценарий (закон, скорость, сфера срабатывания) на сетке дальностей и перегрузок; для строк с перегрузкой больше нуля цель разворачивается постоянно («turn»).'
      }
    >
      {cz && (
        <section className="table-card">
          <table>
            <thead>
              <tr>
                <th>Перегрузка цели</th>
                {cz.ranges_m.map((r) => (
                  <th key={r}>{Math.round(r / 100) / 10} км</th>
                ))}
                <th>{(cz.n_runs ?? 1) > 1 ? 'Статус строки' : 'Границы зоны'}</th>
              </tr>
            </thead>
            <tbody>
              {cz.rows.map((row) => (
                <tr key={row.n_target_g}>
                  <td>{fmt(row.n_target_g, 1)} g</td>
                  {row.cells.map((c) => (
                    <td
                      key={c.range_m}
                      className={c.hit ? 'map-win' : 'map-lose'}
                      style={
                        c.p_hit === undefined
                          ? undefined
                          : {
                              // непрерывная шкала p_hit: красный 0 → янтарь → фосфор 1
                              background: `color-mix(in srgb, var(--status-success) ${Math.round(c.p_hit * 100)}%, var(--status-danger-soft))`,
                              color: c.p_hit >= 0.5 ? 'var(--status-success)' : 'var(--status-danger)',
                            }
                      }
                      title={
                        c.p_hit === undefined
                          ? `R_min ${Math.round(c.r_min_m)} м · ${c.t_end_s.toFixed(2)} с${c.time_limited ? ' · промах по истечении окна' : ''}`
                          : `p_hit ${(c.p_hit * 100).toFixed(0)} % [${((c.p_hit_ci95?.[0] ?? 0) * 100).toFixed(0)}…${((c.p_hit_ci95?.[1] ?? 0) * 100).toFixed(0)}] ДИ Уилсона · медиана R_min ${Math.round(c.median_cpa_m ?? c.r_min_m)} м · серия ${cz.n_runs} прогонов${c.deterministic_cell ? ' · все прогоны совпали' : ''}`
                      }
                    >
                      {c.p_hit === undefined
                        ? c.hit
                          ? 'перехват'
                          : `промах${c.time_limited ? '*' : ''}`
                        : `${Math.round(c.p_hit * 100)} % ±${Math.round(((c.p_hit_ci95?.[1] ?? 0) - (c.p_hit_ci95?.[0] ?? 0)) * 50)}${c.time_limited ? ' *' : ''}`}
                    </td>
                  ))}
                  <td>
                    <small>{rowNote(row)}</small>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </LabScreen>
  )
}

type LawDataBundle = { neff: any; hyp: any; loop: any }

/** Экран «Закон»: разложение команды, гипотезы, closed-loop с суррогатом. */
export function LawScreen({ brain }: { brain: BrainKind }) {
  const [lawData, setLawData] = useState<LawDataBundle | null>(null)
  const [busy, setBusy] = useState(false)
  const [tick, setTick] = useState(0)

  useEffect(() => {
    let cancelled = false
    setBusy(true)
    setLawData(null)
    Promise.all([
      fetch('/api/science/neff-decomposition', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ kind: brain }) }).then((r) => r.json()),
      fetch('/api/science/law-hypotheses', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ kind: brain }) }).then((r) => r.json()),
      fetch('/api/science/closed-loop', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ kind: brain }) }).then((r) => r.json()),
    ])
      .then(([neff, hyp, loop]) => {
        if (!cancelled) setLawData({ neff, hyp, loop })
      })
      .catch(() => {
        if (!cancelled) setLawData(null)
      })
      .finally(() => {
        if (!cancelled) setBusy(false)
      })
    return () => {
      cancelled = true
    }
  }, [brain, tick])

  return (
    <LabScreen
      title="Закон"
      about="Научный разбор закона наведения: из чего состоит команда мозга, какая гипотеза закона лучше её описывает и летит ли суррогат вместо мозга в замкнутом контуре."
      primary={
        <button type="button" className="primary" disabled={busy} onClick={() => setTick((t) => t + 1)}>
          {busy ? 'Считаю…' : 'Обновить отчёт'}
        </button>
      }
      status={busy ? <span>Считаю диагностику закона (разложение → гипотезы → closed-loop)…</span> : !lawData ? <span>Не удалось получить отчёт — проверьте, что стенд запущен.</span> : undefined}
    >
      {lawData && (
        <div className="lab-grid lab-grid--single" style={{ gap: 14 }}>
          <section className="lab-subcard">
            <h3 className="lab-sub" style={{ margin: 0 }}>Разложение команды: a = N·q + остаток (истинная геометрия — только постфактум)</h3>
            <p className="lab-hint">
              Вердикт: <b>{String(lawData.neff?.verdict ?? '—')}</b> — {String(lawData.neff?.interpretation ?? 'нет данных')} · N_wls {fmt(lawData.neff?.medians?.n_wls_median, 3)}
              {lawData.neff?.bootstrap_ci95?.n_wls_median && ` (CI95 ${fmt(lawData.neff.bootstrap_ci95.n_wls_median.low, 3)}…${fmt(lawData.neff.bootstrap_ci95.n_wls_median.high, 3)})`}
              {`, alignment ${fmt(lawData.neff?.medians?.alignment_median, 3)}, остаток ${fmt(lawData.neff?.medians?.residual_median, 3)}`}. Единицы: ρ — 1/с, k_ρ — с.
            </p>
          </section>
          <section className="lab-subcard">
            <h3 className="lab-sub" style={{ margin: 0 }}>Сравнение гипотез закона (аппроксимация команды ≠ закон)</h3>
            <table className="legend-table">
              <thead>
                <tr>
                  <td>гипотеза</td>
                  <td>R² (проверка)</td>
                  <td>ошибка напр., °</td>
                  <td>параметров</td>
                  <td>форма</td>
                </tr>
              </thead>
              <tbody>
                {Object.entries((lawData.hyp?.hypotheses ?? {}) as Record<string, any>).map(([k, h]) => (
                  <tr key={k}>
                    <td>{String(h.label)}</td>
                    <td>{fmt(h.r2_val, 3)}</td>
                    <td>{fmt(h.direction_err_deg, 1)}</td>
                    <td>{String(h.n_params)}</td>
                    <td>{h.model_kind === 'deployable_sensor_model' ? 'можно лететь' : 'только анализ'}</td>
                  </tr>
                ))}
                {(!lawData.hyp?.hypotheses || Object.keys(lawData.hyp.hypotheses).length === 0) && (
                  <tr>
                    <td colSpan={5}>нет данных — отчёт гипотез недоступен на этой версии стенда</td>
                  </tr>
                )}
              </tbody>
            </table>
            {lawData.hyp?.caveat && <p className="lab-hint">{String(lawData.hyp.caveat)}</p>}
          </section>
          <section className="lab-subcard">
            <h3 className="lab-sub" style={{ margin: 0 }}>Проверка в замкнутом контуре: суррогат летит вместо БИО (закреплённое трио)</h3>
            <table className="legend-table">
              <thead>
                <tr>
                  <td>модель</td>
                  <td>перехваты</td>
                  <td>кратчайшее (медиана), м</td>
                  <td>90-й процентиль, м</td>
                  <td>усилие, g·с</td>
                </tr>
              </thead>
              <tbody>
                {((lawData.loop?.rows ?? []) as Array<Record<string, any>>).map((r) => (
                  <tr key={String(r.name)}>
                    <td>{String(r.label)}</td>
                    <td>{((r.hit_rate ?? 0) * 100).toFixed(0)}%</td>
                    <td>{fmt(r.cpa_median_m, 1)}</td>
                    <td>{fmt(r.cpa_p90_m, 1)}</td>
                    <td>{fmt(r.effort_median_gs, 1)}</td>
                  </tr>
                ))}
                {(!lawData.loop?.rows || lawData.loop.rows.length === 0) && (
                  <tr>
                    <td colSpan={5}>нет данных — закрытый контур недоступен на этой версии стенда</td>
                  </tr>
                )}
              </tbody>
            </table>
            {lawData.loop?.rule && <p className="lab-hint">{String(lawData.loop.rule)}</p>}
          </section>
        </div>
      )}
    </LabScreen>
  )
}
