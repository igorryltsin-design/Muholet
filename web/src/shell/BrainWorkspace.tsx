import { useState } from 'react'
import { Brain3DView } from '../lazyViews'
import { Dialog } from '../ui'
import { controlOutputs } from '../metrics'
import { fmt, LAYERS, LAYER_TIP, nCellsLabel } from './labels'
import type { BrainKind, Frame } from '../types'

type TrainCfg = { mode: 'scratch' | 'finetune' | 'result'; lr: number; episodes: number }

/** Рабочее пространство «Мозг»: большая визуализация + состояние обучения + команды. */
export function BrainWorkspace({
  frame,
  brain,
  onBrain,
  day,
  trained,
  trainNote,
  training,
  trainProgress,
  trainCfg,
  onTrainCfg,
  busy,
  onTrain,
  onTrainStop,
  onSaveBrain,
  onLoadBrainFile,
  onResetBrain,
  tune,
  onTune,
  onRebuild,
}: {
  frame: Frame | null
  brain: BrainKind
  onBrain: (b: BrainKind) => void
  day: boolean
  trained: boolean
  trainNote: string
  training: boolean
  trainProgress: { ep: number; total: number } | null
  trainCfg: TrainCfg
  onTrainCfg: (patch: Partial<TrainCfg>) => void
  busy: boolean
  onTrain: () => void
  onTrainStop: () => void
  onSaveBrain: () => void
  onLoadBrainFile: (file: File) => void
  onResetBrain: () => void
  tune: { pool: number; channels: number }
  onTune: (patch: { pool?: number; channels?: number }) => void
  onRebuild: (pool: number, channels: number) => void
}) {
  const [paramsOpen, setParamsOpen] = useState(false)
  const { pitchAcc, yawAcc, loomSig } = controlOutputs(frame, 12)
  const layers = frame?.layers || {}
  const dn = frame?.circuit.dn

  return (
    <div className="twocol-ws">
      <div className="ws-main">
        <Brain3DView frame={frame} kind={brain} day={day} />
      </div>

      <div className="ws-side">
        <h2 className="ws-side__title">Мозг мухи</h2>

        <section className="ws-card">
          <h3>Тип мозга</h3>
          <label data-tip="Коннектом-модель: проводка регионов — из реальных синапсов FlyWire; моторный выход — обучаемый считывающий слой.">
            Выбор мозга
            <select value={brain} onChange={(e) => onBrain(e.target.value as BrainKind)} disabled={busy}>
              <option value="stub">схема (279 клеток)</option>
              <option value="full">полный (4 439 клеток)</option>
              <option value="connectome">коннектом-модель (~109 тыс)</option>
            </select>
          </label>
          <div className="stat-grid">
            <span>
              нейронов <b>{nCellsLabel(brain).toLocaleString('ru')}</b>
            </span>
            <span>
              состояние <b style={{ color: trained ? 'var(--status-success)' : 'var(--text-muted)' }}>{trained ? 'обучен' : 'не обучен'}</b>
            </span>
          </div>
        </section>

        <section className="ws-card">
          <h3>Обучение</h3>
          <p className="lab-hint" style={{ margin: 0 }} data-tip="Учим выход DN подражать эталону ПН на случайных пусках.">
            {trainNote}
          </p>
          {trainProgress && (
            <div className="lab-progress">
              <div className="lab-progress__bar" style={{ width: `${Math.round((trainProgress.ep / Math.max(trainProgress.total, 1)) * 100)}%` }} />
              <span>
                эпизод {trainProgress.ep}/{trainProgress.total}
              </span>
            </div>
          )}
          {training ? (
            <button type="button" className="primary" onClick={onTrainStop}>
              Остановить обучение
            </button>
          ) : (
            <button type="button" className="primary" disabled={busy} onClick={onTrain}>
              Запустить обучение
            </button>
          )}
          <button type="button" onClick={() => setParamsOpen(true)}>
            Параметры обучения…
          </button>
        </section>

        <section className="ws-card">
          <h3>Активность регионов</h3>
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
          <div className="stat-grid">
            <span data-tip="Вертикальная команда нормального ускорения (не угол тангажа корпуса): −1…1.">
              верт. команда <b>{fmt(dn?.pitch ?? pitchAcc, 2)}</b>
            </span>
            <span data-tip="Боковая команда нормального ускорения (не угол рыскания корпуса): −1…1.">
              боков. команда <b>{fmt(dn?.yaw ?? yawAcc, 2)}</b>
            </span>
            <span data-tip="Скорость оптического расширения изображения цели (LPLC2).">
              оптическ. расширение <b>{fmt(frame?.circuit.lplc2 ?? loomSig, 2)}</b>
            </span>
          </div>
        </section>

        <section className="ws-card">
          <h3>Файлы мозга</h3>
          <button type="button" data-tip="В файл: веса выхода, усиление и рецепт проводки (зерно + хеш всех синапсов)." onClick={onSaveBrain}>
            Сохранить в файл
          </button>
          <label className="filelabel" data-tip="Загрузить веса из файла, сохранённого этим стендом.">
            Загрузить из файла
            <input
              type="file"
              accept="application/json,.json"
              aria-label="Файл мозга"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) onLoadBrainFile(f)
                e.target.value = ''
              }}
            />
          </label>
          <button type="button" className="danger" data-tip="Сбросить выбранный мозг к заводскому состоянию. Действие необратимо." onClick={onResetBrain}>
            Сбросить к дефолту
          </button>
        </section>
      </div>

      {paramsOpen && (
        <Dialog title="Параметры обучения" onClose={() => setParamsOpen(false)}>
          <div className="params-grid">
            <section className="lab-subcard">
              <h3 className="lab-sub" style={{ margin: 0 }}>Режим</h3>
              <label>
                Режим обучения
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
                  <option value="result">по результату (довести до попадания)</option>
                </select>
              </label>
              <label>
                Шаг обучения
                <input type="number" step={0.005} min={0.001} max={0.2} value={trainCfg.lr} onChange={(e) => onTrainCfg({ lr: Math.max(0.001, Number(e.target.value) || 0.04) })} />
              </label>
              <label>
                Эпизодов
                <input type="number" min={4} max={80} value={trainCfg.episodes} onChange={(e) => onTrainCfg({ episodes: Math.max(4, Math.min(80, Number(e.target.value) || 24)) })} />
              </label>
              <p className="lab-hint">
                Эпизоды разнообразны: все аспекты, манёвры, дальности 3,5–11 км, шумы и срывы захвата — режим «по
                результату» отбирает веса по медиане промаха.
              </p>
            </section>
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
              <p className="lab-hint">Пересозданный мозг — необученный. Кривая «размер ↔ точность» — в лаборатории, «Масштаб мозга».</p>
            </section>
          </div>
        </Dialog>
      )}
    </div>
  )
}
