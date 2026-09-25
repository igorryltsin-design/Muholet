import { useEffect } from 'react'
import { FlyRig } from '../ui'
import { SeekerView } from '../SeekerView'
import { controlOutputs, navMetrics } from '../metrics'
import { fmt, LAYERS, LAYER_TIP, SPEED_MODE_LABEL } from './labels'
import type { Frame, Scenario } from '../types'

/**
 * Компактная полоса телеметрии: четыре ключевые метрики — дальность R,
 * скорость сближения Vc, радиальная оценка времени R/Vc, прогноз промаха h_cv.
 * Полная телеметрия — в drawer по кнопке.
 */
export function TelemetryBar({
  frame,
  scenario,
  statusText,
  statusKind,
  onOpenFull,
}: {
  frame: Frame | null
  scenario: Scenario
  statusText: string
  statusKind: '' | 'is-ok' | 'is-bad'
  onOpenFull: () => void
}) {
  const nm = frame ? navMetrics(frame) : null
  return (
    <div className="telemetrybar">
      <div className="telemetry">
        <div data-tip="Дистанция ракета–цель в этот момент.">
          <small>Дальность</small>
          <b>{fmt((frame?.range_m ?? scenario.range_m) / 1000, 2)} км</b>
        </div>
        <div data-tip="Как быстро сокращается дистанция. Отрицательная — цель начинает уходить.">
          <small>Сближение</small>
          <b>{fmt(frame?.v_c ?? 0, 0)} м/с</b>
        </div>
        <div data-tip="Радиальная оценка оставшегося времени t_radial = R/Vc (не точное время до встречи).">
          <small>R/Vc</small>
          <b>{fmt(nm?.tgo ?? 0, 2)} с</b>
        </div>
        <div data-tip="h_cv — прогноз промаха при неизменных текущих скоростях (прямая экстраполяция). Работа наведения — стянуть к нулю.">
          <small>Прогноз h_cv</small>
          <b>{fmt(nm?.zem ?? 0, 0)} м</b>
        </div>
        {statusText && <div className={`telemetry__status ${statusKind}`}>{statusText}</div>}
      </div>
      <button type="button" onClick={onOpenFull} data-tip="Полная телеметрия: все метрики кадра, активности, журнал.">
        Полная телеметрия
      </button>
    </div>
  )
}

/** Нижний drawer полной телеметрии: все метрики, кадр головки, рули, активности, журнал. */
export function TelemetryDrawer({
  frame,
  scenario,
  day,
  log,
  onClearLog,
  onClose,
}: {
  frame: Frame | null
  scenario: Scenario
  day: boolean
  log: string[]
  onClearLog: () => void
  onClose: () => void
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const nm = frame ? navMetrics(frame) : null
  const { pitchAcc, yawAcc, gLoad, loomSig } = controlOutputs(frame, scenario.n_max)
  const layers = frame?.layers || {}
  const dn = frame?.circuit.dn

  return (
    <>
      <div className="drawer-scrim" onClick={onClose} />
      <div className="telemetry-drawer" role="dialog" aria-label="Полная телеметрия">
        <div className="drawer__head">
          <h3>
            Полная телеметрия · t = {fmt(frame?.t ?? 0, 2)} с
          </h3>
          <button type="button" onClick={onClose}>
            Закрыть
          </button>
        </div>
        <div className="drawer__body">
          <div className="drawer__col">
            <h4>Кадр головки</h4>
            <SeekerView frame={frame} day={day} bioFovDeg={scenario.bio_fov_deg} />
            <h4>Выход на рули</h4>
            <div className="fly" style={{ gridTemplateColumns: '1fr', padding: 0 }}>
              <FlyRig pitchAcc={pitchAcc} yawAcc={yawAcc} gLoad={gLoad} loom={loomSig} />
            </div>
            <div className="channels">
              <span data-tip="Вертикальная команда нормального ускорения (не угол тангажа корпуса): −1…1.">
                верт. команда <b>{fmt(pitchAcc, 2)}</b>
              </span>
              <span data-tip="Боковая команда нормального ускорения (не угол рыскания корпуса): −1…1.">
                боков. команда <b>{fmt(yawAcc, 2)}</b>
              </span>
              <span data-tip="Скорость оптического расширения изображения цели (LPLC2): чем ближе к 1, тем сильнее тревога.">
                оптическ. расширение <b>{fmt(frame?.circuit.lplc2 ?? 0, 2)}</b>
              </span>
            </div>
          </div>

          <div className="drawer__col">
            <h4>Метрики кадра</h4>
            <dl className="kv">
              <div data-tip="Нормальное ускорение в этом кадре, g (поворот вектора скорости).">
                <dt>Норм. ускорение</dt>
                <dd>{fmt(frame?.n_req ?? 0, 1)}</dd>
              </div>
              <div data-tip="ИСТИННЫЙ угол на цель (оператору; БИО его не видит — только изображение).">
                <dt>Азимут (истина)</dt>
                <dd>{fmt(((frame?.az ?? 0) * 180) / Math.PI, 2)}°</dd>
              </div>
              <div data-tip="ИСТИННЫЙ угол на цель по вертикали (оператору; БИО его не видит).">
                <dt>Угол места (истина)</dt>
                <dd>{fmt(((frame?.el ?? 0) * 180) / Math.PI, 2)}°</dd>
              </div>
              <div data-tip="Вертикальная команда нормального ускорения контура: −1…1 (не угол тангажа корпуса).">
                <dt>Верт. команда</dt>
                <dd>{fmt(dn?.pitch ?? 0, 3)}</dd>
              </div>
              <div data-tip="Боковая команда нормального ускорения контура: −1…1 (не угол рыскания корпуса).">
                <dt>Боков. команда</dt>
                <dd>{fmt(dn?.yaw ?? 0, 3)}</dd>
              </div>
              <div data-tip="Радиальная оценка оставшегося времени t_radial = R/Vc при текущих скоростях (не точное время до встречи).">
                <dt>R/Vc, с</dt>
                <dd>{fmt(nm?.tgo ?? 0, 2)}</dd>
              </div>
              <div data-tip="h_cv — прогноз промаха при неизменных текущих скоростях (прямая экстраполяция до t_cpa).">
                <dt>Прогноз h_cv</dt>
                <dd>{fmt(nm?.zem ?? 0, 0)}</dd>
              </div>
              <div data-tip="Угол между вектором скорости ракеты и направлением на точку встречи.">
                <dt>Упреждение, °</dt>
                <dd>{fmt(nm?.gamma ?? 0, 1)}</dd>
              </div>
              <div data-tip="Измеренный угловой размер цели (рад), декодирован из изображения.">
                <dt>θ (угловой размер)</dt>
                <dd>{fmt((frame?.theta ?? 0) * 1000, 1)} мрад</dd>
              </div>
              <div data-tip="ρ = θ̇/θ — нормированная скорость расширения изображения, оптическая оценка Vc/R.">
                <dt>ρ (фаза сближения)</dt>
                <dd>{fmt(frame?.rho ?? 0, 2)} 1/с</dd>
              </div>
              <div data-tip="τ = θ/θ̇ — ограниченная оценка времени до контакта из изображения.">
                <dt>τ контакта (оптика)</dt>
                <dd>{fmt(frame?.tau_contact ?? 0, 2)} с</dd>
              </div>
              <div data-tip="Скорость цели: профиль и фактический модуль в этом кадре.">
                <dt>Скорость цели ({SPEED_MODE_LABEL[frame?.speed_mode ?? 'constant']})</dt>
                <dd>{fmt(frame?.target_speed ?? 0, 0)} м/с</dd>
              </div>
              <div data-tip="Эквивалентный навигационный коэффициент N_экв: какому постоянному N эквивалентна команда.">
                <dt>
                  N_экв{' '}
                  {frame?.n_eff_valid === false && frame?.n_eff_reason ? (
                    <em style={{ color: 'var(--status-warning)' }}>({frame.n_eff_reason === 'saturation' ? 'насыщение' : frame.n_eff_reason === 'no_geom' ? 'ω≈0' : 'неопр.'})</em>
                  ) : (
                    ''
                  )}
                </dt>
                <dd>{frame?.n_eff !== null && frame?.n_eff !== undefined ? fmt(frame.n_eff, 2) : '—'}</dd>
              </div>
              <div data-tip="R_min — минимальное расстояние сближения за полёт. Если меньше радиуса срабатывания — засчитан перехват.">
                <dt>R_min, м</dt>
                <dd>{fmt(frame?.miss ?? scenario.range_m, 0)}</dd>
              </div>
              <div data-tip="Команда на пределе n_max?">
                <dt>Насыщение команды</dt>
                <dd>{frame?.sat ? 'да' : 'нет'}</dd>
              </div>
            </dl>
          </div>

          <div className="drawer__col">
            <h4>Активность контура</h4>
            <div className="bars">
              {LAYERS.map(([key, name]) => (
                <div key={key} data-tip={LAYER_TIP[name]}>
                  <small>{name}</small>
                  <span>
                    <i style={{ width: `${Math.min(100, (layers[key] || 0) * 140)}%` }} />
                  </span>
                </div>
              ))}
            </div>
            <h4>
              Журнал
              {log.length > 0 && (
                <button type="button" className="log__clear" onClick={onClearLog} data-tip="Очистить журнал событий этого окна.">
                  очистить
                </button>
              )}
            </h4>
            <ul className="log">
              {log.map((line, i) => (
                <li key={`${i}-${line}`}>{line}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </>
  )
}
