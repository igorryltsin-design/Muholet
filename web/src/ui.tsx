import { useEffect, useRef, useState, type ReactNode } from 'react'

/** Мелкая UI-фурнитура: подсказка «?», сворачиваемые панель/секция, меню, диалог и муха «Выход на рули». */

/** Событие «сбросить раскладку»: панели и секции возвращаются к состоянию по умолчанию. */
export const RESET_LAYOUT_EVENT = 'muholet-reset-layout'

export function resetLayout() {
  try {
    const kill: string[] = []
    for (let i = 0; i < localStorage.length; i += 1) {
      const k = localStorage.key(i)
      if (k && (k.startsWith('muholet-panel-') || k.startsWith('muholet-acc-'))) kill.push(k)
    }
    kill.forEach((k) => localStorage.removeItem(k))
  } catch {
    /* приватный режим */
  }
  window.dispatchEvent(new CustomEvent(RESET_LAYOUT_EVENT))
}

/** Состояние open с сохранением в localStorage и сбросом по RESET_LAYOUT_EVENT. */
function useDisclosure(id: string, prefix: string, defaultOpen: boolean) {
  const key = `${prefix}-${id}`
  const [open, setOpen] = useState(() => {
    try {
      const saved = localStorage.getItem(key)
      if (saved !== null) return saved === '1'
    } catch {
      /* приватный режим — просто не запоминаем */
    }
    return defaultOpen
  })
  useEffect(() => {
    const onReset = () => setOpen(defaultOpen)
    window.addEventListener(RESET_LAYOUT_EVENT, onReset)
    return () => window.removeEventListener(RESET_LAYOUT_EVENT, onReset)
    // defaultOpen намеренно не в зависимостях: сброс возвращается к состоянию на момент монтирования
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  const toggle = () =>
    setOpen((o) => {
      try {
        localStorage.setItem(key, o ? '0' : '1')
      } catch {
        /* приватный режим */
      }
      return !o
    })
  return { open, toggle }
}

function Chevron() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
      <path d="M2 4l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

/** Кнопка-шеврон: единственный способ свернуть/развернуть секцию. Шапка не кликабельна. */
function FoldButton({ open, onClick, label }: { open: boolean; onClick: () => void; label: string }) {
  return (
    <button
      type="button"
      className="fold-btn"
      aria-expanded={open}
      aria-label={open ? `Свернуть: ${label}` : `Развернуть: ${label}`}
      title={open ? 'Свернуть' : 'Развернуть'}
      onClick={onClick}
    >
      <Chevron />
    </button>
  )
}

export function Hint({ text }: { text: string }) {
  return (
    <i
      className="hint"
      data-tip={text}
      tabIndex={0}
      role="note"
      aria-label={text}
      onKeyDown={(e) => e.preventDefault()}
    >
      ?
    </i>
  )
}

type PanelProps = {
  /** Ключ для запоминания свёрнутости в localStorage. */
  id: string
  title: ReactNode
  /** Текст подсказки «?» рядом с заголовком. */
  hint?: string
  /** Краткий статус справа (виден и у свёрнутой панели). */
  extra?: ReactNode
  /** Кнопки-действия в шапке. Клик по ним не сворачивает панель. */
  actions?: ReactNode
  className?: string
  /** Начальное состояние, пока пользователь не свернул/развернул вручную. */
  defaultOpen?: boolean
  children: ReactNode
}

/**
 * Панель со сворачиваемым телом. Сворачивается ТОЛЬКО кнопкой-шевроном:
 * клик по шапке ничего не делает, кнопки в шапке не конфликтуют со сворачиванием.
 * Свёрнутое тело полностью исключается из раскладки. Состояние переживает перезагрузку.
 */
export function Panel({ id, title, hint, extra, actions, className, defaultOpen = true, children }: PanelProps) {
  const { open, toggle } = useDisclosure(id, 'muholet-panel', defaultOpen)
  const label = typeof title === 'string' ? title : id
  return (
    <section className={`panel${open ? '' : ' panel--closed'}${className ? ` ${className}` : ''}`}>
      <div className="panel__head panel__head--bare">
        <span className="panel__title">
          {title}
          {hint && <Hint text={hint} />}
        </span>
        <span className="panel__side">
          {extra !== undefined && extra !== null && <span className="panel__extra">{extra}</span>}
          {actions && <span className="actions">{actions}</span>}
          <FoldButton open={open} onClick={toggle} label={label} />
        </span>
      </div>
      <div className="panel__body" hidden={!open}>
        {children}
      </div>
    </section>
  )
}

type AccordionProps = {
  id: string
  title: ReactNode
  hint?: string
  /** Краткий статус справа. */
  extra?: ReactNode
  defaultOpen?: boolean
  children: ReactNode
}

/** Секция-аккордеон контекстного инспектора: та же механика, что у Panel. */
export function Accordion({ id, title, hint, extra, defaultOpen = true, children }: AccordionProps) {
  const { open, toggle } = useDisclosure(id, 'muholet-acc', defaultOpen)
  const label = typeof title === 'string' ? title : id
  return (
    <section className="acc">
      <div className="acc__head">
        <span className="acc__title">
          {title}
          {hint && <Hint text={hint} />}
        </span>
        {extra !== undefined && extra !== null && <span className="acc__extra">{extra}</span>}
        <FoldButton open={open} onClick={toggle} label={label} />
      </div>
      <div className="acc__body" hidden={!open}>
        {children}
      </div>
    </section>
  )
}

export type MenuItem = { label: string; onClick: () => void; disabled?: boolean; danger?: boolean }

/** Выпадающее меню вторичных действий: закрывается по клику вне, Escape и выбору пункта. */
export function Menu({ label, items, title, primary }: { label: string; items: MenuItem[]; title?: string; primary?: boolean }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.code === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    window.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      window.removeEventListener('keydown', onKey)
    }
  }, [open])
  return (
    <div className="menu" ref={ref}>
      <button
        type="button"
        className={primary ? 'primary' : ''}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        {label} ▾
      </button>
      {open && (
        <div className="menu__pop" role="menu">
          {title && <div className="menu__title">{title}</div>}
          {items.map((it) => (
            <button
              key={it.label}
              type="button"
              role="menuitem"
              disabled={it.disabled}
              className={it.danger ? 'danger' : ''}
              onClick={() => {
                setOpen(false)
                it.onClick()
              }}
            >
              {it.label}
            </button>
          ))}
          {!items.length && <div className="menu__title">нет доступных действий</div>}
        </div>
      )}
    </div>
  )
}

/** Модальный диалог: Escape закрывает, клик по фону — тоже. */
export function Dialog({ title, onClose, children, wide }: { title: string; onClose: () => void; children: ReactNode; wide?: boolean }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div className="dialog" role="dialog" aria-label={title} onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className={`dialog__window${wide ? ' dialog__window--wide' : ''}`}>
        <div className="dialog__head">
          <h2>{title}</h2>
          <button type="button" onClick={onClose}>
            Закрыть
          </button>
        </div>
        <div className="dialog__body">{children}</div>
      </div>
    </div>
  )
}

/**
 * Дрозофила сверху: живая привязка к телеметрии ракеты.
 * yawAcc/pitchAcc — проекции реальной команды a_cmd на оси корпуса (−1…1),
 * gLoad = n_req/n_lim — частота и амплитуда взмахов + дрожь на предельных перегрузках,
 * loom — сигнал LPLC2 «цель прёт»: глаза наливаются красным.
 */
export function FlyRig({ pitchAcc, yawAcc, gLoad, loom }: { pitchAcc: number; yawAcc: number; gLoad: number; loom: number }) {
  const flap = Math.max(0.14, 0.9 - Math.min(0.76, gLoad * 0.76))
  const amp = 0.5 + Math.min(1, gLoad) * 0.9 // размах крыльев растёт с перегрузкой
  const arrowUp = -Math.max(0, pitchAcc) * 26
  const arrowDown = -Math.min(0, pitchAcc) * 26
  const strain = gLoad > 0.75
  return (
    <div className={`fly__stage${strain ? ' fly__stage--strain' : ''}`} data-tip={strain ? 'Предельные перегрузки: конструкция на грани.' : undefined}>
      <svg viewBox="0 0 132 96" className="fly__svg" style={{ transform: `rotate(${yawAcc * 22}deg)` }}>
        <defs>
          <radialGradient id="flyBody" cx="38%" cy="40%" r="80%">
            <stop offset="0%" stopColor="#e0c093" />
            <stop offset="65%" stopColor="#a37f4f" />
            <stop offset="100%" stopColor="#66502f" />
          </radialGradient>
          <linearGradient id="flyWing" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="rgba(180,235,220,0.05)" />
            <stop offset="100%" stopColor="rgba(180,235,220,0.34)" />
          </linearGradient>
        </defs>
        {/* лапки */}
        <g stroke="#8a6a3f" strokeWidth="1.5" fill="none" strokeLinecap="round">
          <path d="M52 38 C42 30, 32 27, 20 29" />
          <path d="M52 56 C42 64, 32 67, 20 65" />
          <path d="M72 35 C64 24, 55 17, 44 13" />
          <path d="M72 59 C64 70, 55 77, 44 81" />
          <path d="M90 38 C85 30, 80 26, 73 24" />
          <path d="M90 56 C85 64, 80 68, 73 70" />
        </g>
        {/* крылья: частота от перегрузки, размах от --amp */}
        <g className="fly__wing fly__wing--l" style={{ animationDuration: `${flap}s`, transformOrigin: '64px 44px', ['--amp' as string]: String(amp) }}>
          <path d="M64 44 C44 10, 16 4, 7 15 C0 26, 22 42, 60 47 Z" fill="url(#flyWing)" stroke="rgba(215,245,235,0.45)" strokeWidth="0.8" />
          <path d="M62 44 C44 20, 24 12, 12 17" stroke="rgba(220,250,240,0.4)" strokeWidth="0.7" fill="none" />
        </g>
        <g className="fly__wing fly__wing--r" style={{ animationDuration: `${flap}s`, transformOrigin: '64px 50px', ['--amp' as string]: String(amp) }}>
          <path d="M64 50 C44 84, 16 90, 7 79 C0 68, 22 52, 60 47 Z" fill="url(#flyWing)" stroke="rgba(215,245,235,0.45)" strokeWidth="0.8" />
          <path d="M62 50 C44 74, 24 82, 12 77" stroke="rgba(220,250,240,0.4)" strokeWidth="0.7" fill="none" />
        </g>
        {/* брюшко с сегментами */}
        <ellipse cx="38" cy="47" rx="23" ry="11.5" fill="url(#flyBody)" />
        <g stroke="rgba(60,44,22,0.55)" strokeWidth="1.2" fill="none">
          <path d="M46 37.5 C46 43, 46 51, 46 56.5" />
          <path d="M36 36.5 C36 43, 36 51, 36 57.5" />
          <path d="M26 39 C26 44, 26 50, 26 55" />
        </g>
        {/* грудь */}
        <ellipse cx="74" cy="47" rx="17" ry="13.5" fill="url(#flyBody)" />
        <ellipse cx="70" cy="42" rx="8" ry="4.5" fill="rgba(255,240,210,0.14)" />
        {/* голова: глаза наливаются красным от сигнала приближения */}
        <circle cx="98" cy="47" r="9" fill="#b08a55" />
        <g style={{ filter: `drop-shadow(0 0 ${(loom * 7).toFixed(1)}px rgba(255,42,26,0.9))` }}>
          <ellipse cx="102" cy="40.5" rx="5" ry="6.5" fill="#cf3a2b" />
          <ellipse cx="102" cy="53.5" rx="5" ry="6.5" fill="#cf3a2b" />
        </g>
        <ellipse cx="103.5" cy="39" rx="1.8" ry="2.6" fill="rgba(255,180,170,0.75)" />
        <ellipse cx="103.5" cy="52" rx="1.8" ry="2.6" fill="rgba(255,180,170,0.75)" />
        <path d="M105 44 C110 42, 112 40, 114 37 M105 50 C110 52, 112 54, 114 57" stroke="#8a6a3f" strokeWidth="1.1" fill="none" strokeLinecap="round" />
        {/* команда тангажа: стрелка от брюшка */}
        {Math.abs(pitchAcc) > 0.03 && (
          <g stroke="#7dffc8" strokeWidth="1.6" strokeLinecap="round">
            {arrowUp !== 0 && <path d={`M38 34 L38 ${34 + arrowUp} M35 ${39 + arrowUp} L38 ${34 + arrowUp} L41 ${39 + arrowUp}`} fill="none" />}
            {arrowDown !== 0 && <path d={`M38 60 L38 ${60 + arrowDown} M35 ${55 + arrowDown} L38 ${60 + arrowDown} L41 ${55 + arrowDown}`} fill="none" />}
          </g>
        )}
      </svg>
    </div>
  )
}
