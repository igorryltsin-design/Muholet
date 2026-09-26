import { useEffect, useState } from 'react'
import { LabScreen } from '../LabScreen'
import type { BrainKind } from '../types'

const BRAIN_RU: Record<string, string> = { stub: 'Схема', full: 'Полный мозг', connectome: 'Коннектом (FlyWire)' }

type FormulaEntry = { kind: BrainKind; data: any }

/** Экран «Формула»: точные формулы команд и полиномиальные приближения. */
export function FormulaScreen() {
  const [formulaAll, setFormulaAll] = useState<FormulaEntry[] | null>(null)
  const [surDegree, setSurDegree] = useState(0)

  useEffect(() => {
    let cancelled = false
    setFormulaAll(null)
    const dq = surDegree ? `&degree=${surDegree}` : ''
    Promise.all(
      (['stub', 'full', 'connectome'] as BrainKind[]).map((kind) =>
        fetch(`/api/brain/formula?kind=${kind}${dq}`)
          .then((r) => r.json())
          .then((data) => ({ kind, data })),
      ),
    )
      .then((res) => {
        if (!cancelled) setFormulaAll(res)
      })
      .catch(() => {
        if (!cancelled) setFormulaAll([])
      })
    return () => {
      cancelled = true
    }
  }, [surDegree])

  return (
    <LabScreen
      title="Формула"
      about="Для каждого мозга — его точная формула (как команды рождаются из признаков) и упрощение полиномом. Погрешность приближения считается на отложенной сетке."
      actions={
        <>
          <label data-tip="Степень полинома-приближения: «авто» растёт со сложностью мозга (схема 1, полный 2, коннектом 3). Выше степень — точнее, но формула длиннее; 4 обычно переобучается." className="chip">
            Степень приближения
            <select value={surDegree} onChange={(e) => setSurDegree(Number(e.target.value))} style={{ width: 'auto' }}>
              <option value={0}>авто</option>
              <option value={1}>1 — линейная</option>
              <option value={2}>2 — квадратичная</option>
              <option value={3}>3 — кубическая</option>
              <option value={4}>4 (эксперим.)</option>
            </select>
          </label>
          {formulaAll && formulaAll.length > 0 && (
            <button
              type="button"
              className="primary"
              data-tip="Скачать все три формулы: исполняемый Python-файл с формулой мозга и легендой признаков для каждого."
              onClick={() => {
                ;(formulaAll ?? []).forEach(({ kind }, i) => {
                  window.setTimeout(() => {
                    const a = document.createElement('a')
                    a.href = `/api/brain/formula/python?kind=${kind}${surDegree ? `&degree=${surDegree}` : ''}`
                    a.download = `muholet-mozg-${kind}.py`
                    a.click()
                  }, i * 300)
                })
              }}
            >
              Скачать формулы: Python
            </button>
          )}
        </>
      }
      status={!formulaAll ? <span>Считаю формулы трёх мозгов…</span> : formulaAll.length === 0 ? <span>Не удалось получить формулы — проверьте, что стенд запущен.</span> : undefined}
    >
      {formulaAll && formulaAll.length > 0 && (
        <section className="lab-subcard">
          <h3 className="lab-sub" style={{ margin: 0 }}>Легенда признаков (входы закона наведения)</h3>
          <table className="legend-table">
            <tbody>
              {(formulaAll[0].data.feature_spec as Array<{ key: string; symbol: string; ru: string; term: string }>).map((f) => (
                <tr key={f.key}>
                  <td className="legend-sym">{f.symbol}</td>
                  <td className="legend-ru">{f.ru}</td>
                  <td className="legend-term">{f.term}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
      <div className="lab-grid lab-grid--single" style={{ gap: 14 }}>
        {(formulaAll ?? []).map(({ kind, data }) => (
          <section key={kind} className="lab-subcard">
            <h3 className="lab-sub" style={{ margin: 0 }}>
              {BRAIN_RU[kind]} · {data.n_cells.toLocaleString('ru')} нейронов · {data.trained ? 'обучен' : 'не обучен'} · приближение: степень {data.poly.degree}
            </h3>
            {data.exact.вид && <p className="lab-hint">{data.exact.вид}</p>}
            {Array.isArray(data.exact.dn_формула) ? (
              data.exact.dn_формула.map((row: string, i: number) => (
                <pre key={i} className="lab-formula">
                  {row}
                </pre>
              ))
            ) : (
              <pre className="lab-formula">{data.exact.dn_формула}</pre>
            )}
            <p className="lab-hint">
              Погрешность приближения (отложенная сетка): R² {data.poly.r2_pitch} / {data.poly.r2_yaw} · средняя {data.poly.err_mean_pitch} · макс {data.poly.err_max_pitch} (единицы команды −1…1).
            </p>
            <pre className="lab-formula">вертикальная команда ≈ {data.poly.formula_ru_pitch}</pre>
            <pre className="lab-formula">боковая команда ≈ {data.poly.formula_ru_yaw}</pre>
            <div className="row">
              <button
                type="button"
                data-tip="Исполняемый Python-файл: точная формула мозга, легенда признаков и пример использования."
                onClick={() => {
                  const a = document.createElement('a')
                  a.href = `/api/brain/formula/python?kind=${kind}${surDegree ? `&degree=${surDegree}` : ''}`
                  a.download = `muholet-mozg-${kind}.py`
                  a.click()
                }}
              >
                Скачать: Python (.py)
              </button>
              <button
                type="button"
                data-tip="Чистый C99 (только math.h): точная сеть dn() + полином dn_approx() — для внешнего симулятора или железа."
                onClick={() => {
                  const a = document.createElement('a')
                  a.href = `/api/brain/formula/c?kind=${kind}${surDegree ? `&degree=${surDegree}` : ''}`
                  a.download = `muholet-mozg-${kind}.c`
                  a.click()
                }}
              >
                Скачать: C (.c)
              </button>
            </div>
          </section>
        ))}
      </div>
    </LabScreen>
  )
}

/** Экран «Параметры»: настройки экспериментов, обучения, лаборатории, звука. */
export function ParamsScreen({
  tune,
  onTune,
  onRebuild,
  trainCfg,
  onTrainCfg,
  egg,
  onEgg,
  soundOn,
  onSound,
  voiceKind,
  onVoiceKind,
  humorOn,
  onHumor,
  smooth,
  showFact,
  onLabCfg,
  serverOnline,
}: {
  tune: { pool: number; channels: number }
  onTune: (patch: { pool?: number; channels?: number }) => void
  onRebuild: (pool: number, channels: number) => void
  trainCfg: { mode: 'scratch' | 'finetune' | 'result'; lr: number; episodes: number }
  onTrainCfg: (patch: Partial<{ mode: 'scratch' | 'finetune' | 'result'; lr: number; episodes: number }>) => void
  egg: boolean
  onEgg: () => void
  soundOn: boolean
  onSound: (v: boolean) => void
  voiceKind: 'male' | 'female'
  onVoiceKind: (v: 'male' | 'female') => void
  humorOn: boolean
  onHumor: (v: boolean) => void
  smooth: number
  showFact: boolean
  onLabCfg: (patch: Partial<{ smooth: number; showFact: boolean }>) => void
  serverOnline: boolean | null
}) {
  const offline = serverOnline === false
  return (
    <LabScreen
      title="Параметры"
      about="Исследовательские настройки: размеры скрытых слоёв, режимы обучения, сглаживание графиков, звук и пасхалка."
    >
      <div className="params-grid">
        <section className="lab-subcard">
          <h3 className="lab-sub" style={{ margin: 0 }}>Скрытые слои — исследование ёмкости</h3>
          <label>
            Каналы коннектома 8→N
            <input type="number" min={16} max={512} value={tune.channels} onChange={(e) => onTune({ channels: Math.max(16, Math.min(512, Number(e.target.value) || 16)) })} />
          </label>
          <label>
            Пул полного мозга
            <input type="number" min={8} max={128} value={tune.pool} onChange={(e) => onTune({ pool: Math.max(8, Math.min(128, Number(e.target.value) || 32)) })} />
          </label>
          <button type="button" data-tip="Пересоздать мозг с новыми размерами. Обучение сбрасывается." onClick={() => onRebuild(tune.pool, tune.channels)}>
            Пересоздать мозг
          </button>
          <p className="lab-hint">Ось исследования: как число скрытых параметров меняет точность наведения. Пересозданный мозг — необученный.</p>
        </section>

        <section className="lab-subcard">
          <h3 className="lab-sub" style={{ margin: 0 }}>Обучение</h3>
          <label>
            Режим
            <select
              value={trainCfg.mode}
              onChange={(e) =>
                onTrainCfg(
                  e.target.value === 'finetune'
                    ? { mode: 'finetune', lr: 0.006, episodes: 12 }
                    : e.target.value === 'result'
                      ? { mode: 'result', lr: 0.04, episodes: 10 }
                      : { mode: 'scratch', lr: 0.04, episodes: 24 },
                )
              }
            >
              <option value="scratch">с нуля</option>
              <option value="finetune">дообучение</option>
              <option value="result" disabled={offline}>
                по результату (довести до попадания){offline ? ' — только на стенде' : ''}
              </option>
            </select>
          </label>
          {offline && trainCfg.mode === 'result' && (
            <p className="lab-hint">Доводка по результату считается только на стенде — сейчас сервер недоступен. Выбран режим «по результату», но запуск обучения не сработает, пока стенд не поднимется.</p>
          )}
          <label>
            Шаг обучения
            <input type="number" step={0.005} min={0.001} max={0.2} value={trainCfg.lr} onChange={(e) => onTrainCfg({ lr: Math.max(0.001, Number(e.target.value) || 0.04) })} />
          </label>
          <label>
            Эпизодов
            <input type="number" min={4} max={80} value={trainCfg.episodes} onChange={(e) => onTrainCfg({ episodes: Math.max(4, Math.min(80, Number(e.target.value) || 24)) })} />
          </label>
          <p className="lab-hint">
            Кнопка «Запустить обучение» — в пространстве «Мозг» и на экране «Обучение». Эпизоды разнообразны: все аспекты,
            манёвры, дальности 3,5–11 км, шумы и срывы захвата.
          </p>
        </section>

        <section className="lab-subcard">
          <h3 className="lab-sub" style={{ margin: 0 }}>Лаборатория</h3>
          <label>
            Окно сглаживания, эпизодов
            <input type="number" min={3} max={21} value={smooth} onChange={(e) => onLabCfg({ smooth: Math.max(3, Math.min(21, Number(e.target.value) || 7)) })} />
          </label>
          <button type="button" className={showFact ? 'on' : ''} data-tip="Тонкая линия фактических значений поверх скользящего среднего." onClick={() => onLabCfg({ showFact: !showFact })}>
            {showFact ? 'показывать факт: вкл' : 'показывать факт: выкл'}
          </button>
          <p className="lab-hint">Применяется ко всем графикам обучения сразу: тонкая линия — факт, яркая — среднее за окно.</p>
        </section>

        <section className="lab-subcard">
          <h3 className="lab-sub" style={{ margin: 0 }}>Пасхалка и звук</h3>
          <button type="button" className={egg ? 'on' : ''} data-tip="Разрез мухолёта на 3D-сцене: внутри муха-пилот, рычаги — реальные команды DN. Клавиша X — переключатель." onClick={onEgg}>
            {egg ? 'муха за штурвалом: включена ✕' : 'муха за штурвалом 🪰'}
          </button>
          <div className="row">
            <button type="button" className={soundOn ? 'on' : ''} data-tip="Жужжание мухи в тон манёвра и голосовые фразы (Silero TTS). При включённом звуке полёт идёт вдвое медленнее." onClick={() => onSound(!soundOn)}>
              {soundOn ? 'озвучка: включена 🔊' : 'озвучка: выключена 🔇'}
            </button>
            <button type="button" className={voiceKind === 'female' ? 'on' : ''} data-tip="Голос озвучки: мужской (eugene) или женский (xenia)." onClick={() => onVoiceKind(voiceKind === 'female' ? 'male' : 'female')}>
              {voiceKind === 'female' ? 'голос: женский' : 'голос: мужской'}
            </button>
          </div>
          <div className="row">
            <button type="button" className={humorOn ? 'on' : ''} onClick={() => onHumor(!humorOn)} data-tip="Вторая муха противоположного полу комментирует полёт субтитрами и голосом — физика и мозг не трогаются, все реплики выводятся из настоящей телеметрии прогона.">
              {humorOn ? (voiceKind === 'female' ? 'юмор: Штруман ♂ 🪰' : 'юмор: Штрумана ♀ 🪰') : 'юмор: выключен'}
            </button>
          </div>
          <p className="lab-hint">
            Муха жужжит в тон манёвру и проговаривает этапы: пуск, захват, потеря, перехват, промах. Записи — Silero TTS,
            регенерируются tools/make_voice.py.
          </p>
        </section>
      </div>
    </LabScreen>
  )
}
