import type { ReactNode } from 'react'
import { TopBar, type Workspace } from './TopBar'
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
  menuItems: MenuItem[]
  children: ReactNode
}) {
  return (
    <div className="shell">
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
      <main className="workspace">{children}</main>
    </div>
  )
}
