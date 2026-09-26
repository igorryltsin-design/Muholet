/** Метка у кнопок, которые считаются только на работающем Python-стенде (не на GitHub Pages). */
export function ServerOnlyBadge() {
  return (
    <span className="chip is-warn server-only-badge" title="Нужен запущенный локальный стенд — недоступно на статическом хостинге (GitHub Pages)">
      только на стенде
    </span>
  )
}
