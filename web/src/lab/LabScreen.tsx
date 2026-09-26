import type { ReactNode } from 'react'
import { Menu, type MenuItem } from '../ui'

/**
 * Единый шаблон экрана лаборатории: заголовок, описание цели анализа,
 * главное действие, вторичные действия (меню «Экспорт»), состояние, контент.
 */
export function LabScreen({
  title,
  about,
  primary,
  actions,
  exports,
  status,
  note,
  children,
}: {
  title: string
  about: string
  /** Главное действие экрана (готовая кнопка) — опционально. */
  primary?: ReactNode
  /** Дополнительные вторичные кнопки рядом с главным действием. */
  actions?: ReactNode
  /** Пункты меню «Экспорт» (CSV/MATLAB/снимки). */
  exports?: MenuItem[]
  /** Строка состояния: сводки, прогресс, ошибки. */
  status?: ReactNode
  /** Примечание под контентом. */
  note?: ReactNode
  children?: ReactNode
}) {
  return (
    <div className="lab-screen">
      <header className="lab-screen__head">
        <div className="lab-screen__intro">
          <h2>{title}</h2>
          <p>{about}</p>
        </div>
        <div className="lab-screen__actions">
          {primary}
          {actions}
          {exports && exports.length > 0 && <Menu label="Экспорт" items={exports} title="Выгрузить данные" />}
        </div>
      </header>
      {status != null && <div className="lab-screen__status">{status}</div>}
      {children}
      {note != null && <p className="lab-screen__note">{note}</p>}
    </div>
  )
}
