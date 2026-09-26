import { useEffect, useState, type ReactNode } from 'react'

/**
 * Мягкий переход при смене активной панели: входящая появляется через
 * плавный fade + едва заметный сдвиг, а не мгновенным хардкатом смены DOM.
 *
 * Осознанно НЕ держит старую панель смонтированной ради настоящего кроссфейда:
 * у «Полёта»/«Роя»/«Дуэли» и «Мозга» — свой живой WebGL-контекст
 * (EngagementView/Brain3DView), и держать два одновременно ради красивой
 * анимации — риск на пустом месте (двойная нагрузка на GPU, гонка teardown).
 * Переход односторонний (только вход), но полностью снимает ощущение
 * «щелчка» при переключении рабочих пространств и вкладок лаборатории.
 * `prefers-reduced-motion` гасится глобальным CSS-правилом (styles.css).
 */
export function WorkspaceTransition({ transitionKey, children }: { transitionKey: string | number; children: ReactNode }) {
  const [entered, setEntered] = useState(false)
  useEffect(() => {
    setEntered(false)
    const raf = requestAnimationFrame(() => setEntered(true))
    return () => cancelAnimationFrame(raf)
  }, [transitionKey])
  return <div className={`ws-pane${entered ? ' ws-pane--in' : ''}`}>{children}</div>
}
