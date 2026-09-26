import { useEffect, useRef } from 'react'
import { Menu, type MenuItem } from '../ui'

/** Рабочие пространства: одновременно виден только один рабочий контекст. */
export type Workspace = 'flight' | 'brain' | 'swarm' | 'duel' | 'lab'

export const WORKSPACES: [Workspace, string, string][] = [
  ['flight', 'Полёт', 'Настройка и запуск перехвата'],
  ['brain', 'Мозг', 'Визуализация и обучение мозга'],
  ['swarm', 'Рой', 'Эволюция популяции мух'],
  ['duel', 'Дуэль', 'Муха-ракета против мухи-самолёта'],
  ['lab', 'Лаборатория', 'Анализы, графики и экспорт'],
]

/** Постоянный переключатель пространств в верхней навигации. */
export function WorkspaceNavigation({ value, onChange }: { value: Workspace; onChange: (w: Workspace) => void }) {
  // на телефоне полоска вкладок прокручивается вбок: активная вкладка всегда в поле зрения
  const navRef = useRef<HTMLElement>(null)
  useEffect(() => {
    const nav = navRef.current
    const btn = nav?.querySelector<HTMLElement>('button.on')
    if (!nav || !btn || nav.scrollWidth <= nav.clientWidth) return
    const n = nav.getBoundingClientRect()
    const b = btn.getBoundingClientRect()
    if (b.left < n.left) nav.scrollLeft += b.left - n.left - 12
    else if (b.right > n.right) nav.scrollLeft += b.right - n.right + 12
  }, [value])
  return (
    <nav className="workspace-nav" role="tablist" aria-label="Рабочие пространства" ref={navRef} data-tour="workspaces">
      {WORKSPACES.map(([key, label, tip]) => (
        <button
          key={key}
          type="button"
          role="tab"
          aria-selected={value === key}
          className={value === key ? 'on' : ''}
          data-tip={tip}
          data-tip-pos="down"
          data-tour={key === 'lab' ? 'lab-nav' : undefined}
          onClick={() => onChange(key)}
        >
          {label}
        </button>
      ))}
    </nav>
  )
}

/**
 * Верхняя панель: только глобальное — имя, пространства, связь, статус расчёта,
 * главный «Пуск/Останов», тема, справка и меню вторичных действий.
 * Аспект, закон и захват живут на сцене, не здесь.
 */
export function TopBar({
  subtitle,
  workspace,
  onWorkspace,
  serverOnline,
  statusText,
  statusKind,
  running,
  canLaunch,
  onLaunch,
  onStop,
  day,
  onToggleDay,
  onHelp,
  menuItems,
}: {
  subtitle: string
  workspace: Workspace
  onWorkspace: (w: Workspace) => void
  serverOnline: boolean | null
  statusText: string
  statusKind: '' | 'is-ok' | 'is-bad' | 'is-warn'
  /** Идёт расчёт/обучение/рой — кнопка становится «Останов». */
  running: boolean
  /** Разрешён ли пуск сейчас (кнопка «Пуск» заблокирована во время расчёта). */
  canLaunch: boolean
  onLaunch: () => void
  onStop: () => void
  day: boolean
  onToggleDay: () => void
  onHelp: () => void
  menuItems: MenuItem[]
}) {
  return (
    <header className="topbar">
      <div className="topbar__brand">
        <strong>МУХОЛЕТ</strong>
        <span data-tip="Версия сборки и состояние выбранного мозга.">{subtitle}</span>
      </div>
      <WorkspaceNavigation value={workspace} onChange={onWorkspace} />
      <div className="topbar__spacer" />
      <div className="topbar__status">
        <span
          className={`chip ${serverOnline === null ? '' : serverOnline ? 'is-ok' : 'is-warn'}`}
          data-tip={serverOnline ? 'Стенд отвечает — расчёт идёт на нём, это быстрее.' : 'Стенд не отвечает: расчёт идёт здесь, в окне браузера.'}
        >
          {serverOnline === null ? 'связь…' : serverOnline ? 'на стенде' : 'в окне'}
        </span>
        <span className={`status-text ${statusKind}`} role="status">
          {statusText}
        </span>
      </div>
      {running ? (
        <button type="button" className="launch-btn is-stop" onClick={onStop} data-tip="Остановить текущий расчёт (работающий расчёт всегда можно остановить)." data-tour="launch">
          Останов
        </button>
      ) : (
        <button type="button" className="launch-btn go" onClick={onLaunch} disabled={!canLaunch} data-tip="Прогнать перехват с текущими условиями (Пробел)." data-tour="launch">
          Пуск
        </button>
      )}
      <div className="topbar__tools">
        <button type="button" className="icon-btn" onClick={onToggleDay} data-tip={day ? 'Ночная тема: фосфорная сцена.' : 'Дневная тема: светлое небо и светлый интерфейс.'} aria-label={day ? 'Включить ночную тему' : 'Включить дневную тему'}>
          {day ? '☾' : '☀'}
        </button>
        <button type="button" className="icon-btn" onClick={onHelp} data-tip="Справка: что за мозги, что они учат, метрики и управление." aria-label="Справка">
          ?
        </button>
        <Menu label="Ещё" items={menuItems} />
      </div>
    </header>
  )
}
