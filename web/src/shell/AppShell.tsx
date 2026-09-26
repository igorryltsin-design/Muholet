import type { ReactNode } from 'react'
import { TopBar, type Workspace } from './TopBar'
import { WorkspaceTransition } from './WorkspaceTransition'
import type { MenuItem } from '../ui'

/** Каркас приложения: верхняя панель + активное рабочее пространство. */
export function AppShell({
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
  cinema,
  onToggleCinema,
  menuItems,
  children,
}: {
  subtitle: string
  workspace: Workspace
  onWorkspace: (w: Workspace) => void
  serverOnline: boolean | null
  statusText: string
  statusKind: '' | 'is-ok' | 'is-bad' | 'is-warn'
  running: boolean
  canLaunch: boolean
  onLaunch: () => void
  onStop: () => void
  day: boolean
  onToggleDay: () => void
  onHelp: () => void
  /** сцена во весь экран без шапки/панелей — усилитель «вау» для показа, не для каждого пуска */
  cinema: boolean
  onToggleCinema: () => void
  menuItems: MenuItem[]
  children: ReactNode
}) {
  // кнопка кино имеет смысл только там, где есть 3D-сцена перехвата
  const showCinemaBtn = workspace === 'flight' || workspace === 'swarm' || workspace === 'duel'
  return (
    <div className={cinema ? 'shell shell--cinema' : 'shell'}>
      <TopBar
        subtitle={subtitle}
        workspace={workspace}
        onWorkspace={onWorkspace}
        serverOnline={serverOnline}
        statusText={statusText}
        statusKind={statusKind}
        running={running}
        canLaunch={canLaunch}
        onLaunch={onLaunch}
        onStop={onStop}
        day={day}
        onToggleDay={onToggleDay}
        onHelp={onHelp}
        menuItems={menuItems}
      />
      <main className="workspace">
        <WorkspaceTransition transitionKey={workspace}>{children}</WorkspaceTransition>
        {showCinemaBtn && (
          <button
            type="button"
            className={`cinema-btn${cinema ? ' on' : ''}`}
            onClick={onToggleCinema}
            data-tip={cinema ? 'Выйти из кинорежима (Esc или K).' : 'Кинорежим: сцена во весь экран без панелей (K).'}
          >
            {cinema ? '✕ кино' : '⛶ кино'}
          </button>
        )}
      </main>
    </div>
  )
}
